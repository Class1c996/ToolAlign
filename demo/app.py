"""Dependency-free CLI demo for one successful multi-step task and clarification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs import build_standard_environment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["order", "clarification", "travel"], default="order")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    episode, executor = build_standard_environment("demo", 7)
    calls = {
        "order": [("get_order", {"order_id": "O-1001"}), ("cancel_order", {"order_id": "O-1001"})],
        "travel": [("search_flights", {"origin": "SHA", "destination": "HND", "date": "2026-08-01"}), ("create_booking", {"flight_id": "F-3001", "passenger_name": "Lin"})],
        "clarification": [],
    }[args.scenario]
    print(json.dumps({"scenario": args.scenario, "checkpoint": args.checkpoint, "initial_state": episode.snapshot()}, ensure_ascii=False, indent=2))
    if args.scenario == "clarification":
        print(json.dumps({"action": "final", "answer": "请提供订单号后我才能继续取消。"}, ensure_ascii=False))
        return 0
    for tool, arguments in calls:
        result = executor.execute(episode, tool, arguments)
        print(json.dumps({"action": "tool_call", "tool": tool, "arguments": arguments, "result": result.as_dict()}, ensure_ascii=False))
    print(json.dumps({"action": "final", "answer": "任务已完成。", "final_state": episode.snapshot()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
