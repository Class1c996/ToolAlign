"""Optional xLAM ingestion and conservative normalization boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Salesforce/xlam-function-calling-60k")
    parser.add_argument("--limit", type=int, default=30000)
    parser.add_argument("--output", default="data/processed/xlam_normalized.jsonl")
    args = parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError:
        print(json.dumps({"status": "BLOCKED", "reason": "datasets_not_installed", "dataset": args.dataset}))
        return 2
    dataset = load_dataset(args.dataset, split="train")
    rows, seen = [], set()
    for index, row in enumerate(dataset):
        if index >= args.limit:
            break
        normalized = {"source": "xlam", "source_id": str(row.get("id", index)), "record": row}
        key = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        rows.append(normalized)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "count": len(rows), "dataset": args.dataset, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
