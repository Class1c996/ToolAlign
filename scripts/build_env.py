"""Build a deterministic environment manifest and initial snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs import build_standard_environment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env.yaml")
    parser.add_argument("--output", default="data/env_manifest.json")
    args = parser.parse_args()
    episode, executor = build_standard_environment()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"task_id": episode.task_id, "seed": episode.seed, "snapshot": episode.snapshot(), "tools": executor.registry.manifest()}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "tools": len(executor.registry.manifest()), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
