"""Run reproducible local oracle/no-op checks and write the four-group table.

Actual Base/SFT/GRPO model rows remain marked pending until model dependencies,
weights, and adapters are available; no synthetic model metrics are emitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import summarize  # noqa: E402
from scripts.run_rollout import run_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/test_seen.jsonl")
    parser.add_argument("--output", default="reports/four_group_table.json")
    args = parser.parse_args()
    tasks = [json.loads(line) for line in (ROOT / args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for policy in ("noop", "gold"):
        rollouts = [run_task(task, policy) for task in tasks]
        metric_rows = [{"split": item["split"], "format_legal": int(item["reward"]["format_valid"] >= 0), "tool_success": int(all(x["result"].get("ok") for x in item["trace"])), "parameter_success": int(all(x["result"].get("ok") for x in item["trace"])), "success": int(item["reward"]["success"]), "false_success": 0, "call_count": len(item["trace"])} for item in rollouts]
        rows.append({"name": "deterministic_noop" if policy == "noop" else "environment_oracle", "status": "measured", "metrics": summarize(metric_rows)})
    rows.extend({"name": name, "status": "pending", "reason": "model weights and training dependencies not available"} for name in ["Base", "SFT", "SFT + GRPO terminal-only", "SFT + GRPO shaped reward"])
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output), "measured_rows": 2, "pending_model_rows": 4}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
