from __future__ import annotations

from envs import build_smoke_environment, build_standard_environment, parse_action, replay_digest
from rewards import calculate_reward
from scripts.evaluate_checkpoint import challenge_success
from scripts.run_rollout import run_task
from scripts.task_suite import load_tasks
from training.rollout_reward import shaped_reward


def test_parser_accepts_strict_tool_and_final_actions() -> None:
    tool = parse_action('{"action":"tool_call","tool":"get_order","arguments":{"order_id":"O-1001"}}')
    final = parse_action('```json\n{"action":"final","answer":"done"}\n```')
    assert tool.action == "tool_call" and tool.tool == "get_order"
    assert final.action == "final" and final.answer == "done"
    assert parse_action('{"action":"tool_call","tool":"get_order"}').error_code == "INVALID_TOOL_CALL"


def test_executor_rejects_unknown_and_missing_arguments() -> None:
    episode, executor = build_standard_environment()
    assert executor.execute(episode, "not_a_tool", {}).error_code == "UNKNOWN_TOOL"
    assert executor.execute(episode, "get_order", {}).error_code == "MISSING_REQUIRED"
    assert executor.execute(episode, "get_order", {"order_id": "O-1001", "extra": True}).error_code == "UNKNOWN_ARGUMENT"


def test_state_override_replays_task_specific_environment() -> None:
    task = {
        "task_id": "override-test", "seed": 11, "split": "test", "user": "Inspect order.",
        "allowed_tools": ["get_order"],
        "gold_calls": [{"tool": "get_order", "arguments": {"order_id": "O-9102"}}],
        "state_override": {"orders": [{"order_id": "O-9102", "status": "shipped", "total": 79.0, "items": ["headset"]}]},
        "goal": {"kind": "order_status", "order_id": "O-9102", "status": "shipped"},
    }
    result = run_task(task, "gold")
    assert result["trace"][0]["result"]["data"]["order_id"] == "O-9102"
    assert result["reward"]["success"] is True


def test_reward_adapter_uses_explicit_task_state_override() -> None:
    import json

    task = {
        "task_id": "reward-override", "seed": 13,
        "state_override": {"orders": [{"order_id": "O-9911", "status": "shipped", "total": 42.0, "items": ["case"]}]},
        "goal": {"kind": "order_status", "order_id": "O-9911", "status": "shipped"},
    }
    values = shaped_reward(
        ['{"action":"tool_call","tool":"get_order","arguments":{"order_id":"O-9911"}}'],
        task_id=[task["task_id"]], seed=[task["seed"]], task_json=[json.dumps(task)],
    )
    assert values[0] > 1.0


def test_reward_requires_state_change_for_return() -> None:
    episode, executor = build_standard_environment("reward", 7)
    initial = episode.snapshot()
    result = executor.execute(episode, "create_return", {"order_id": "O-1003", "reason": "damaged"})
    reward = calculate_reward(
        {"goal": {"kind": "return_created", "order_id": "O-1003"}},
        initial,
        episode.snapshot(),
        [{"tool": "create_return", "arguments": {"order_id": "O-1003", "reason": "damaged"}, "result": result.as_dict()}],
        "Return requested.",
    )
    assert result.ok and reward.success and reward.total_terminal_only == 1.0


def test_replay_digest_is_deterministic() -> None:
    episode, executor = build_smoke_environment(seed=19)
    calls = [("search_orders", {"status": "paid"}), ("cancel_order", {"order_id": "O-1001"})]
    assert replay_digest(episode, executor, calls) == replay_digest(episode, executor, calls)


def test_challenge_success_requires_exact_trace_and_final_answer() -> None:
    task = {"expected_calls": [{"tool": "get_order", "arguments": {"order_id": "O-1"}}], "expected_answer_any": ["paid"]}
    trace = [{"tool": "get_order", "arguments": {"order_id": "O-1"}, "result": {"ok": True}}]
    assert challenge_success(task, trace, [{"action": "tool_call"}, {"action": "final"}], "Order is paid.")
    assert not challenge_success(task, [], [{"action": "final"}], "Order is paid.")


def test_public_example_suite_is_explicit_and_small() -> None:
    tasks, paths = load_tasks(suite="public_examples")
    assert len(paths) == 1 and len(tasks) == 8
