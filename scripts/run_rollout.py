"""Run deterministic gold/no-op rollouts and persist complete trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs import build_standard_environment  # noqa: E402
from rewards import calculate_reward  # noqa: E402
from scripts.task_suite import load_tasks  # noqa: E402


def run_task(task: dict, policy: str) -> dict:
    episode, executor = build_standard_environment(task["task_id"], task["seed"], task.get("state_override"))
    initial = episode.snapshot()
    calls = task["gold_calls"] if policy == "gold" else []
    trace = []
    for call in calls:
        result = executor.execute(episode, call["tool"], call["arguments"])
        trace.append({"tool": call["tool"], "arguments": call["arguments"], "result": result.as_dict()})
    answer = "Please provide the order ID so I can continue." if task["goal"]["kind"] == "clarification" else "The requested return is ready." if task["goal"]["kind"] == "no_tool" else "Task completed."
    reward = calculate_reward(task, initial, episode.snapshot(), trace, answer)
    return {"task_id": task["task_id"], "split": task["split"], "task": task, "trace": trace, "final_answer": answer, "final_state": episode.snapshot(), "reward": reward.as_dict()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="public_smoke", help="Named suite in data/suites.json.")
    parser.add_argument("--input", default="", help="Explicit JSONL path; mutually exclusive with --suite.")
    parser.add_argument("--output", default="reports/rollouts.jsonl")
    parser.add_argument("--policy", choices=["gold", "noop"], default="gold")
    args = parser.parse_args()
    tasks, paths = load_tasks(suite=None if args.input else args.suite, input_path=args.input or None)
    rows = [run_task(task, args.policy) for task in tasks]
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "suite": args.suite if not args.input else None, "inputs": [str(path) for path in paths], "count": len(rows), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
