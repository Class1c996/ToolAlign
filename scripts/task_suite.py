"""Load explicit task suites without scanning arbitrary JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "suites.json"


def suite_paths(name: str, manifest: Path = DEFAULT_MANIFEST) -> list[Path]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    suites = payload.get("suites", {})
    if name not in suites:
        available = ", ".join(sorted(suites))
        raise ValueError(f"unknown suite {name!r}; available: {available}")
    paths = [ROOT / relative for relative in suites[name].get("files", [])]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("suite files are missing: " + ", ".join(missing))
    return paths


def load_tasks(*, suite: str | None = None, input_path: str | None = None) -> tuple[list[dict[str, Any]], list[Path]]:
    if bool(suite) == bool(input_path):
        raise ValueError("provide exactly one of suite or input_path")
    paths = suite_paths(suite) if suite else [ROOT / str(input_path)]
    if not paths[0].is_file():
        raise FileNotFoundError(paths[0])
    tasks: list[dict[str, Any]] = []
    for path in paths:
        tasks.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return tasks, paths
