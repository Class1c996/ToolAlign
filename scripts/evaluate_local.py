"""Convert rollout logs into the standard ToolAlign metric table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import summarize, write_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--input", default="reports/rollouts.jsonl")
    parser.add_argument("--json-output", default="reports/metrics.json")
    parser.add_argument("--csv-output", default="reports/metrics.csv")
    args = parser.parse_args()
    rows = []
    with (ROOT / args.input).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            trace = item["trace"]
            allowed = set(item["task"]["allowed_tools"])
            reward = item["reward"]
            rows.append({
                "task_id": item["task_id"], "split": item["split"],
                "format_legal": int(reward["format_valid"] >= 0),
                "tool_success": int(all(step["tool"] in allowed and step["result"].get("ok") for step in trace)),
                "parameter_success": int(all(step["result"].get("ok") for step in trace)),
                "success": int(reward["success"]), "false_success": int((not reward["success"]) and item["final_answer"].lower().find("completed") >= 0),
                "call_count": len(trace), "terminal_reward": reward["total_terminal_only"], "shaped_reward": reward["total_shaped"],
            })
    summary = summarize(rows)
    json_path, csv_path = ROOT / args.json_output, ROOT / args.csv_output
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    print(json.dumps({"status": "PASS", "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
