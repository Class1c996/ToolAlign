"""Execute every generated task against the local environment and assert gold success."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_rollout import run_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed")
    args = parser.parse_args()
    rows = []
    for path in sorted((ROOT / args.input).glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    failures = [task["task_id"] for task in rows if not run_task(task, "gold")["reward"]["success"]]
    result = {"status": "PASS" if not failures else "FAIL", "count": len(rows), "successes": len(rows) - len(failures), "failures": failures[:20]}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
