"""One-command local reproduction for the dependency-free ToolAlign loop."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--suite", default="public_smoke", help="Named deterministic suite to validate and replay.")
    args = parser.parse_args()
    commands = [
        ["scripts/build_env.py"],
        ["scripts/prepare_data.py", "--count", str(args.count)],
        ["scripts/validate_storage.py"],
        ["scripts/validate_training_interfaces.py"],
        ["scripts/validate_tasks.py", "--suite", args.suite],
        ["scripts/run_rollout.py", "--suite", args.suite, "--policy", "gold"],
        ["scripts/evaluate.py"],
        ["scripts/run_experiments.py"],
    ]
    for command in commands:
        print("$", PYTHON, *command, flush=True)
        completed = subprocess.run([PYTHON, *command], cwd=ROOT)
        if completed.returncode:
            return completed.returncode
    print("reproduce PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
