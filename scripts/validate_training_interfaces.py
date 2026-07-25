"""Validate parser, SFT formatting, and reward adapter without model packages."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs import build_standard_environment, parse_action  # noqa: E402
from training.data import format_task  # noqa: E402
from training.rollout_reward import executable_reward  # noqa: E402


def main() -> int:
    episode, executor = build_standard_environment()
    tool = parse_action('{"action":"tool_call","tool":"get_order","arguments":{"order_id":"O-1001"}}')
    final = parse_action('<tool_call>{"action":"final","answer":"done"}</tool_call>')
    bad = parse_action('{"action":"tool_call","tool":"get_order"}')
    assert tool.action == "tool_call" and tool.tool == "get_order"
    assert final.action == "final" and bad.error_code == "INVALID_TOOL_CALL"
    task = {"task_id": "iface", "seed": 7, "user": "check order", "gold_calls": [{"tool": "get_order", "arguments": {"order_id": "O-1001"}}], "goal": {"kind": "order_status", "order_id": "O-1001", "status": "paid"}}
    formatted = format_task(task, executor.registry.manifest())
    assert "tool_call" in formatted and "get_order" in formatted
    assert len(executable_reward(['{"action":"final","answer":"return policy"}'])) == 1
    print(json.dumps({"status": "PASS", "tools": len(executor.registry.manifest()), "formatted_chars": len(formatted)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
