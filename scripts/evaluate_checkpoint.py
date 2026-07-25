"""Run executable model rollouts against the deterministic ToolAlign tasks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs import build_standard_environment, parse_action  # noqa: E402
from eval.metrics import summarize, write_csv  # noqa: E402
from rewards import calculate_reward  # noqa: E402


def clean_generation(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = text.replace("<|im_end|>", "").strip()
    return text


def make_prompt(tokenizer: Any, task: dict[str, Any], manifest: list[dict[str, Any]], history: list[dict[str, str]]) -> str:
    allowed = set(task.get("allowed_tools", []))
    task_manifest = [tool for tool in manifest if tool.get("name") in allowed] if "allowed_tools" in task else manifest
    system = "You are a tool-use agent. Emit exactly one JSON action per turn. Use exactly this schema: {\"action\":\"tool_call\",\"tool\":\"<name>\",\"arguments\":{}} or {\"action\":\"final\",\"answer\":\"<text>\"}. Tools=" + json.dumps(task_manifest, sort_keys=True)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": f"[task_id={task['task_id']}] {task['user']}"}]
    messages.extend(history)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def generate_action(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    return clean_generation(tokenizer.decode(generated, skip_special_tokens=False))


def challenge_success(task: dict[str, Any], trace: list[dict[str, Any]], parsed_actions: list[dict[str, Any]], final_answer: str) -> bool:
    """Require the declared plan, successful execution, and a final response."""
    expected = task.get("expected_calls")
    if expected is None:
        return False
    actual = [{"tool": step["tool"], "arguments": step["arguments"]} for step in trace]
    if actual != expected:
        return False
    if not parsed_actions or parsed_actions[-1].get("action") != "final" or not final_answer.strip():
        return False
    alternatives = [value.lower() for value in task.get("expected_answer_any", [])]
    return not alternatives or any(value in final_answer.lower() for value in alternatives)


def run_task(model: Any, tokenizer: Any, task: dict[str, Any], manifest: list[dict[str, Any]], max_turns: int, max_new_tokens: int) -> dict[str, Any]:
    episode, executor = build_standard_environment(task["task_id"], task["seed"], task.get("state_override"))
    initial = episode.snapshot()
    trace: list[dict[str, Any]] = []
    history: list[dict[str, str]] = []
    generations: list[str] = []
    parsed_actions: list[dict[str, Any]] = []
    final_answer = ""
    for _ in range(max_turns):
        generated = generate_action(model, tokenizer, make_prompt(tokenizer, task, manifest, history), max_new_tokens)
        generations.append(generated)
        parsed = parse_action(generated)
        parsed_actions.append(parsed.as_dict())
        if parsed.action == "final":
            final_answer = parsed.answer or ""
            break
        if parsed.action != "tool_call":
            break
        result = executor.execute(episode, parsed.tool or "", parsed.arguments or {})
        trace.append({"tool": parsed.tool, "arguments": parsed.arguments, "result": result.as_dict()})
        history.append({"role": "assistant", "content": generated})
        history.append({"role": "tool", "content": json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True)})
    reward = calculate_reward(task, initial, episode.snapshot(), trace, final_answer)
    success = challenge_success(task, trace, parsed_actions, final_answer) if str(task.get("benchmark", "")).startswith("challenge_") else reward.success
    valid_format = bool(parsed_actions) and all(item.get("action") in {"tool_call", "final"} for item in parsed_actions)
    allowed = set(task["allowed_tools"])
    tool_success = all(step["tool"] in allowed and step["result"].get("ok") is True for step in trace)
    parameter_success = all(step["result"].get("ok") is True for step in trace)
    return {
        "task_id": task["task_id"],
        "split": task["split"],
        "task": task,
        "trace": trace,
        "generations": generations,
        "parsed_actions": parsed_actions,
        "final_answer": final_answer,
        "final_state": episode.snapshot(),
        "reward": reward.as_dict(),
        "format_legal": int(valid_format),
        "tool_success": int(tool_success),
        "parameter_success": int(parameter_success),
        "success": int(success),
        "false_success": int((not success) and "completed" in final_answer.lower()),
        "call_count": len(trace),
        "terminal_reward": reward.total_terminal_only,
        "shaped_reward": reward.total_shaped,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="", help="LoRA adapter path; omit for the base model")
    parser.add_argument("--input", default="data/processed/test_seen.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = ROOT / "models" / "Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True, local_files_only=True)
    if args.checkpoint:
        model = PeftModel.from_pretrained(model, ROOT / args.checkpoint, is_trainable=False)
    model.eval()
    _, executor = build_standard_environment()
    tasks = [json.loads(line) for line in (ROOT / args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        tasks = tasks[: args.limit]
    rows = [run_task(model, tokenizer, task, executor.registry.manifest(), args.max_turns, args.max_new_tokens) for task in tasks]
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    summary_rows = [{key: row[key] for key in ("task_id", "split", "format_legal", "tool_success", "parameter_success", "success", "false_success", "call_count", "terminal_reward", "shaped_reward")} for row in rows]
    summary = summarize(summary_rows)
    if args.csv_output:
        write_csv(summary_rows, ROOT / args.csv_output)
    print(json.dumps({"status": "PASS", "checkpoint": args.checkpoint or "base_model", "count": len(rows), "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
