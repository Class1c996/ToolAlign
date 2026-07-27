"""Run a real Base/SFT/GRPO checkpoint through the executable tool environment.

This is intentionally a CLI rather than a scripted gold-call showcase.  Every
assistant turn in its JSON output originates from ``model.generate`` and is
recorded alongside the parsed action, tool result, reward breakdown, and state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs import build_standard_environment, parse_action  # noqa: E402
from rewards import calculate_reward  # noqa: E402
from scripts.evaluate_checkpoint import clean_generation, make_prompt  # noqa: E402


PRESETS = {
    "base": "",
    "sft": "checkpoints/sft_multitask_state_v5",
    "grpo": "checkpoints/grpo_interactive_multitask_v2_small",
}


def load_model(checkpoint: str) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = ROOT / "models" / "Qwen3-1.7B"
    if not base.is_dir():
        raise FileNotFoundError(f"base model is not available: {base}")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if checkpoint:
        adapter = ROOT / checkpoint
        if not adapter.is_dir():
            raise FileNotFoundError(f"checkpoint is not available: {adapter}")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return model, tokenizer


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


def run_demo(model: Any, tokenizer: Any, task: dict[str, Any], max_turns: int, max_new_tokens: int) -> dict[str, Any]:
    episode, executor = build_standard_environment(task["task_id"], task["seed"], task.get("state_override"))
    initial_state = episode.snapshot()
    manifest = executor.registry.manifest()
    history: list[dict[str, str]] = []
    trace: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    final_answer = ""
    for turn_index in range(max_turns):
        prompt = make_prompt(tokenizer, task, manifest, history)
        raw_output = generate_action(model, tokenizer, prompt, max_new_tokens)
        parsed = parse_action(raw_output)
        turn: dict[str, Any] = {
            "turn": turn_index + 1,
            "raw_model_output": raw_output,
            "parsed_action": parsed.as_dict(),
        }
        if parsed.action == "final":
            final_answer = parsed.answer or ""
            turns.append(turn)
            break
        if parsed.action != "tool_call":
            turns.append(turn)
            break
        result = executor.execute(episode, parsed.tool or "", parsed.arguments or {})
        result_payload = result.as_dict()
        turn["tool_result"] = result_payload
        trace.append({"tool": parsed.tool, "arguments": parsed.arguments, "result": result_payload})
        history.extend([
            {"role": "assistant", "content": raw_output},
            {"role": "tool", "content": json.dumps(result_payload, ensure_ascii=False, sort_keys=True)},
        ])
        turns.append(turn)
    reward = calculate_reward(task, initial_state, episode.snapshot(), trace, final_answer)
    return {
        "task_id": task["task_id"],
        "user": task["user"],
        "checkpoint": task.get("_checkpoint", "base"),
        "initial_state": initial_state,
        "turns": turns,
        "trace": trace,
        "final_answer": final_answer,
        "reward_breakdown": reward.as_dict(),
        "final_state": episode.snapshot(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(PRESETS), default="sft", help="Local checkpoint preset.")
    parser.add_argument("--checkpoint", default="", help="Override adapter path; omit for the base model.")
    parser.add_argument("--input", default="data/processed/challenge_v1.jsonl")
    parser.add_argument("--task-id", default="", help="Task ID to run; defaults to the first row.")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--output", default="", help="Optional JSON file for the complete demo trace.")
    args = parser.parse_args()

    tasks = [json.loads(line) for line in (ROOT / args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    task = next((item for item in tasks if item["task_id"] == args.task_id), tasks[0])
    checkpoint = args.checkpoint or PRESETS[args.model]
    task = {**task, "_checkpoint": checkpoint or "base"}
    model, tokenizer = load_model(checkpoint)
    payload = run_demo(model, tokenizer, task, args.max_turns, args.max_new_tokens)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output = ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
