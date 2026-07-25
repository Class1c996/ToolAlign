"""Optional BFCL adapter boundary; local evaluation remains dependency-free."""

from __future__ import annotations

from pathlib import Path


def evaluate_bfcl(path: str | Path) -> dict[str, object]:
    target = Path(path)
    if not target.exists():
        return {"status": "SKIPPED", "reason": "BFCL data not present", "path": str(target)}
    return {"status": "READY", "reason": "adapter boundary created", "path": str(target)}
