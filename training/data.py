"""Task-to-text formatting shared by SFT and GRPO."""

from __future__ import annotations

import json
from typing import Any

from envs import build_standard_environment


def build_messages(task: dict[str, Any], tool_manifest: list[dict[str, Any]]) -> list[dict[str, str]]:
    # Training must replay the same initial state declared by the task.  Without
    # this, a row containing a held-out order/flight/event ID silently receives
    # default-environment tool results and teaches an impossible trajectory.
    episode, executor = build_standard_environment(task["task_id"], task["seed"], task.get("state_override"))
    messages = build_prompt_messages(task, tool_manifest)
    for call in task.get("gold_calls", []):
        action = {"action": "tool_call", "tool": call["tool"], "arguments": call["arguments"]}
        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False, sort_keys=True)})
        result = executor.execute(episode, call["tool"], call["arguments"])
        messages.append({"role": "tool", "content": json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True)})
    answer = task.get("final_answer") or ("Please provide the order ID so I can continue." if task["goal"]["kind"] == "clarification" else "The requested return is ready." if task["goal"]["kind"] == "no_tool" else "Task completed.")
    messages.append({"role": "assistant", "content": json.dumps({"action": "final", "answer": answer}, ensure_ascii=False, sort_keys=True)})
    return messages


def build_prompt_messages(task: dict[str, Any], tool_manifest: list[dict[str, Any]]) -> list[dict[str, str]]:
    allowed = set(task.get("allowed_tools", []))
    task_manifest = [tool for tool in tool_manifest if tool.get("name") in allowed] if "allowed_tools" in task else tool_manifest
    return [
        {"role": "system", "content": "You are a tool-use agent. Emit exactly one JSON action per turn. Use exactly this schema: {\"action\":\"tool_call\",\"tool\":\"<name>\",\"arguments\":{}} or {\"action\":\"final\",\"answer\":\"<text>\"}. Tools=" + json.dumps(task_manifest, sort_keys=True)},
        {"role": "user", "content": f"[task_id={task['task_id']}] {task['user']}"},
    ]


def format_task(task: dict[str, Any], tool_manifest: list[dict[str, Any]], tokenizer: Any | None = None) -> str:
    messages = build_messages(task, tool_manifest)
    if tokenizer is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def format_prompt(task: dict[str, Any], tool_manifest: list[dict[str, Any]], tokenizer: Any) -> str:
    return tokenizer.apply_chat_template(build_prompt_messages(task, tool_manifest), tokenize=False, add_generation_prompt=True, enable_thinking=False)


def build_sft_rows(tasks: list[dict[str, Any]], tool_manifest: list[dict[str, Any]], tokenizer: Any | None = None) -> list[dict[str, Any]]:
    if tokenizer is None:
        return [{"text": format_task(task, tool_manifest), "task_id": task["task_id"], "seed": task["seed"]} for task in tasks]
    rows: list[dict[str, Any]] = []
    eos = tokenizer.eos_token or "<|im_end|>"
    for task in tasks:
        messages = build_messages(task, tool_manifest)
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            prompt_messages = messages[:index]
            prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            completion = message["content"] + eos
            rows.append({"prompt": prompt, "completion": completion, "task_id": task["task_id"], "seed": task["seed"], "turn_index": index})
    return rows
