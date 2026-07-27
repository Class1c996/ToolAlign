"""Export a tiny deterministic task sample suitable for a public repository."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "test_seen.jsonl"
OUT = ROOT / "data" / "examples" / "public_smoke_8.jsonl"


def main() -> int:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()][:8]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "count": len(rows), "output": str(OUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
