"""Generate deterministic self-built JSONL tasks and split manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.tasks import generate_tasks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = generate_tasks(args.count, args.seed)
    counts = Counter(task["split"] for task in tasks)
    for split in sorted(counts):
        path = output_dir / f"{split}.jsonl"
        rows = [task for task in tasks if task["split"] == split]
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps({"seed": args.seed, "count": len(tasks), "splits": counts}, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps({"status": "PASS", "count": len(tasks), "splits": counts}, ensure_ascii=False, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
