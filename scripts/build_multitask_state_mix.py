"""Mix the original broad task distribution with state-counterfactual data."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "multitask_state_mix_v5.jsonl"


def main() -> None:
    rows: list[dict] = []
    for relative in ("data/processed/train.jsonl", "data/processed/state_counterfactual_v2.jsonl"):
        for line in (ROOT / relative).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                item["task_id"] = f"multitask-state-{len(rows):05d}"
                item["seed"] = 53000 + len(rows)
                rows.append(item)
    OUT.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT), "count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
