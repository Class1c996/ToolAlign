"""Small, explicit interactive GRPO loop for executable multi-turn tool use.

Each sampled completion is rolled out through the local environment.  The
policy is then updated from group-relative trajectory rewards, so the trainer
cannot optimize a single isolated action while ignoring later tool results.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from envs import build_standard_environment, parse_action
from training.data import build_prompt_messages


ROOT = Path(__file__).resolve().parents[1]


def clean_generation(text: str) -> str:
    text = text.replace("<|im_end|>", "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def load_tasks(path: Path, manifest: list[dict[str, Any]], max_gold_calls: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        calls = task.get("gold_calls", [])
        if len(calls) > max_gold_calls:
            continue
        episode, executor = build_standard_environment(task["task_id"], task["seed"], task.get("state_override"))
        executable = True
        for call in calls:
            result = executor.execute(episode, call["tool"], call["arguments"])
            if not result.ok:
                executable = False
                break
        if executable:
            tasks.append(task)
    if not tasks:
        raise RuntimeError("No executable interactive training tasks were found")
    return tasks


def make_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def trajectory_reward(task: dict[str, Any], trace: list[dict[str, Any]], parsed_actions: list[dict[str, Any]], final_answer: str) -> float:
    expected = task.get("gold_calls", [])
    actual = [{"tool": step["tool"], "arguments": step["arguments"]} for step in trace]
    prefix = 0
    for left, right in zip(actual, expected):
        if left != right:
            break
        prefix += 1
    prefix_ratio = prefix / max(1, len(expected))
    answer_any = [value.lower() for value in task.get("expected_answer_any", [])]
    answer_ok = not answer_any or any(value in final_answer.lower() for value in answer_any)
    terminal = float(actual == expected and parsed_actions and parsed_actions[-1].get("action") == "final" and bool(final_answer.strip()) and answer_ok)
    failures = sum(1 for step in trace if step["result"].get("ok") is not True)
    extra = max(0, len(actual) - len(expected))
    return max(-1.0, 0.25 * prefix_ratio + 0.75 * terminal - 0.15 * failures - 0.05 * extra)


def rollout(model: Any, tokenizer: Any, task: dict[str, Any], manifest: list[dict[str, Any]], max_turns: int, max_new_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
    episode, executor = build_standard_environment(task["task_id"], task["seed"], task.get("state_override"))
    messages = build_prompt_messages(task, manifest)
    trace: list[dict[str, Any]] = []
    parsed_actions: list[dict[str, Any]] = []
    final_answer = ""
    segments: list[tuple[list[int], list[int]]] = []
    for _ in range(max_turns):
        prompt = make_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output[0, prompt_len:].detach().cpu().tolist()
        generated = clean_generation(tokenizer.decode(generated_ids, skip_special_tokens=False))
        parsed = parse_action(generated)
        parsed_actions.append(parsed.as_dict())
        segments.append((inputs["input_ids"][0].detach().cpu().tolist(), generated_ids))
        messages.append({"role": "assistant", "content": generated})
        if parsed.action == "final":
            final_answer = parsed.answer or ""
            break
        if parsed.action != "tool_call":
            break
        result = executor.execute(episode, parsed.tool or "", parsed.arguments or {})
        trace.append({"tool": parsed.tool, "arguments": parsed.arguments, "result": result.as_dict()})
        messages.append({"role": "tool", "content": json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True)})
    reward = trajectory_reward(task, trace, parsed_actions, final_answer)
    return {"segments": segments, "reward": reward, "success": int(reward >= 0.75), "trace": trace, "actions": parsed_actions}


def completion_logprob(model: Any, prompt_ids: list[int], completion_ids: list[int]) -> torch.Tensor:
    if not completion_ids:
        return torch.zeros((), device=model.device)
    ids = torch.tensor([prompt_ids + completion_ids], device=model.device, dtype=torch.long)
    attention = torch.ones_like(ids)
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
    prompt_len = len(prompt_ids)
    predicted = logits[:, prompt_len - 1 : -1, :]
    target = ids[:, prompt_len:]
    log_probs = torch.log_softmax(predicted, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return log_probs.sum()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_interactive.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    base_path = ROOT / config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True, local_files_only=True)
    model = PeftModel.from_pretrained(base, ROOT / config["adapter"], is_trainable=True)
    model.train()
    manifest = build_standard_environment()[1].registry.manifest()
    tasks = load_tasks(ROOT / config["train_file"], manifest, int(config.get("max_gold_calls", 2)))
    rng = random.Random(int(config.get("seed", 7)))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(config.get("learning_rate", 1e-5)))
    max_steps = int(config.get("smoke_steps", 2) if args.smoke else config.get("max_steps", 20))
    group_size = int(config.get("num_generations", 2))
    batch_tasks = int(config.get("batch_tasks", 2))
    max_turns = int(config.get("max_turns", 4))
    zero_variance_fallback = bool(config.get("zero_variance_fallback", False))
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    for step in range(1, max_steps + 1):
        batch = [tasks[rng.randrange(len(tasks))] for _ in range(batch_tasks)]
        trajectories: list[dict[str, Any]] = []
        groups: list[list[dict[str, Any]]] = []
        for task in batch:
            group = [rollout(model, tokenizer, task, manifest, max_turns, int(config.get("max_new_tokens", 96)), float(config.get("temperature", 0.8)), float(config.get("top_p", 0.95))) for _ in range(group_size)]
            groups.append(group)
            trajectories.extend(group)
        advantages: list[float] = []
        advantage_modes: list[str] = []
        for group in groups:
            rewards = torch.tensor([item["reward"] for item in group], dtype=torch.float32)
            mean = float(rewards.mean())
            std = float(rewards.std(unbiased=False))
            if std < 1e-6:
                if zero_variance_fallback:
                    advantages.extend(rewards.tolist())
                    advantage_modes.append("zero_baseline_fallback")
                else:
                    advantages.extend([0.0] * len(group))
                    advantage_modes.append("zero_variance_skip")
            else:
                advantages.extend(((rewards - mean) / (std + 1e-6)).tolist())
                advantage_modes.append("group_relative")
        loss_terms: list[torch.Tensor] = []
        for item, advantage in zip(trajectories, advantages):
            if advantage == 0.0:
                continue
            logprob = sum(completion_logprob(model, prompt_ids, completion_ids) for prompt_ids, completion_ids in item["segments"])
            loss_terms.append(-float(advantage) * logprob / max(1, len(item["segments"])))
        if loss_terms:
            loss = torch.stack(loss_terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu())
        else:
            loss_value = 0.0
        rewards = [item["reward"] for item in trajectories]
        metrics = {"step": step, "loss": loss_value, "reward_mean": sum(rewards) / len(rewards), "reward_std": math.sqrt(sum((x - sum(rewards) / len(rewards)) ** 2 for x in rewards) / len(rewards)), "success_rate": sum(item["success"] for item in trajectories) / len(trajectories), "groups": len(groups), "advantage_modes": advantage_modes}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "steps": max_steps, "training_tasks": len(tasks)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
