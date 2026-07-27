"""Fail if Git-tracked release files include local artifacts or large binaries."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
BANNED_PREFIXES = (".venv/", ".downloads/", "models/Qwen3-1.7B/", "checkpoints/", "logs/")
BANNED_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".pem", ".key", ".p12", ".pfx")


def main() -> int:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    files = [item.decode("utf-8") for item in output.split(b"\0") if item]
    failures: list[str] = []
    largest = {"path": "", "bytes": 0}
    for relative in files:
        normalized = relative.replace("\\", "/")
        path = ROOT / relative
        if normalized.startswith(BANNED_PREFIXES) or normalized.lower().endswith(BANNED_SUFFIXES):
            failures.append(f"banned tracked artifact: {normalized}")
        if path.is_file():
            size = path.stat().st_size
            if size > largest["bytes"]:
                largest = {"path": normalized, "bytes": size}
            if size > MAX_BYTES:
                failures.append(f"tracked file exceeds {MAX_BYTES} bytes: {normalized} ({size})")
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "tracked_files": len(files),
        "max_file_bytes": MAX_BYTES,
        "largest": largest,
        "failures": failures,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
