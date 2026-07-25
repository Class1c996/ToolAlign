"""Terminal and shaped rewards with explicit component logging."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RewardBreakdown:
    terminal: float
    format_valid: float
    tool_valid: float
    parameter_valid: float
    efficiency: float
    safety_penalty: float
    total_terminal_only: float
    total_shaped: float
    success: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_reward(task: dict[str, Any], initial_state: dict[str, Any], final_state: dict[str, Any], trace: list[dict[str, Any]], final_answer: str = "") -> RewardBreakdown:
    success = _goal_satisfied(task.get("goal", {}), initial_state, final_state, trace, final_answer)
    valid_format = all(isinstance(step.get("tool"), str) and isinstance(step.get("arguments"), dict) for step in trace)
    valid_tool = all(step.get("result", {}).get("error_code") != "UNKNOWN_TOOL" for step in trace)
    valid_params = all(step.get("result", {}).get("ok") is True for step in trace) if trace else True
    duplicate_count = len(trace) - len({(step.get("tool"), repr(step.get("arguments"))) for step in trace})
    efficiency = 0.02 if duplicate_count == 0 else -0.04 * duplicate_count
    safety_penalty = -0.2 * sum(1 for step in trace if step.get("result", {}).get("error_code") in {"UNKNOWN_TOOL", "UNKNOWN_ARGUMENT"})
    terminal = 1.0 if success else 0.0
    format_bonus = 0.05 if valid_format else -0.05
    tool_bonus = 0.05 if valid_tool else -0.1
    parameter_bonus = 0.1 if valid_params else -0.1
    terminal_only = terminal
    # A premature final answer must be worse than an unfinished neutral
    # trajectory; otherwise shaped GRPO can learn to stop after one easy call.
    completion_penalty = 0.0 if success else -0.5
    shaped = terminal + format_bonus + tool_bonus + parameter_bonus + efficiency + safety_penalty + completion_penalty
    return RewardBreakdown(terminal, format_bonus, tool_bonus, parameter_bonus, efficiency, safety_penalty, terminal_only, shaped, success)


def _goal_satisfied(goal: dict[str, Any], initial: dict[str, Any], final: dict[str, Any], trace: list[dict[str, Any]], answer: str) -> bool:
    kind = goal.get("kind")
    if kind == "clarification":
        return not trace and all(field not in answer.lower() for field in ("cancelled", "done", "success"))
    if kind == "no_tool":
        return not trace and goal.get("answer_contains", "").lower() in answer.lower()
    if kind == "order_status":
        return any(x.get("order_id") == goal.get("order_id") and x.get("status") == goal.get("status") for x in final.get("orders", []))
    if kind == "return_created":
        return any(x.get("order_id") == goal.get("order_id") for x in final.get("returns", []))
    if kind == "event_created":
        return any(x.get("title") == goal.get("title") and x.get("date") == goal.get("date") for x in final.get("events", []))
    if kind == "booking_created":
        return any(x.get("passenger_name") == goal.get("passenger_name") and x.get("status") == "confirmed" for x in final.get("bookings", []))
    return False
