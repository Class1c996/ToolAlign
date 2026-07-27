"""GRPO reward adapter that executes parsed actions in the local environment."""

from __future__ import annotations

import json
from typing import Any

from envs import build_standard_environment, parse_action
from rewards import calculate_reward
from scripts.task_suite import load_tasks


_TASK_CACHE: dict[str, dict[str, Any]] | None = None


def _load_tasks() -> dict[str, dict[str, Any]]:
    global _TASK_CACHE
    if _TASK_CACHE is None:
        tasks, _ = load_tasks(suite="generated_core")
        _TASK_CACHE = {task["task_id"]: task for task in tasks}
    return _TASK_CACHE


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    if isinstance(completion, list):
        for item in reversed(completion):
            if isinstance(item, dict) and "content" in item:
                return str(item["content"])
        return json.dumps(completion, ensure_ascii=False)
    return json.dumps(completion, ensure_ascii=False)


def _executable_rewards(
    completions: list[Any],
    shaped: bool,
    prompts: list[str] | None = None,
    task_id: list[str] | None = None,
    seed: list[int] | None = None,
    task_json: list[str] | None = None,
    **_: Any,
) -> list[float]:
    values = []
    for index, completion in enumerate(completions):
        text = _completion_text(completion)
        identifier = task_id[index] if task_id is not None and index < len(task_id) else f"grpo-{index}"
        episode_seed = seed[index] if seed is not None and index < len(seed) else 7
        if task_json is not None and index < len(task_json):
            task = json.loads(task_json[index])
        else:
            task = _load_tasks().get(identifier, {"task_id": identifier, "seed": episode_seed, "goal": {"kind": "no_tool", "answer_contains": "return"}})
        episode, executor = build_standard_environment(identifier, episode_seed, task.get("state_override"))
        trace = []
        parsed = parse_action(text)
        if parsed.action == "error":
            values.append(-1.0)
            continue
        if parsed.action == "tool_call":
            result = executor.execute(episode, parsed.tool or "", parsed.arguments or {})
            trace.append({"tool": parsed.tool, "arguments": parsed.arguments, "result": result.as_dict()})
        answer = parsed.answer or ""
        reward = calculate_reward(task, episode.initial_state, episode.snapshot(), trace, answer)
        values.append(reward.total_shaped if shaped else (1.0 if reward.success else -1.0))
    return values


def terminal_reward(completions: list[Any], prompts: list[str] | None = None, task_id: list[str] | None = None, seed: list[int] | None = None, task_json: list[str] | None = None, **kwargs: Any) -> list[float]:
    return _executable_rewards(completions, False, prompts, task_id, seed, task_json, **kwargs)


def shaped_reward(completions: list[Any], prompts: list[str] | None = None, task_id: list[str] | None = None, seed: list[int] | None = None, task_json: list[str] | None = None, **kwargs: Any) -> list[float]:
    return _executable_rewards(completions, True, prompts, task_id, seed, task_json, **kwargs)


# Backward-compatible name used by the interface validation script.
executable_reward = shaped_reward
