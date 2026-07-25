"""CPU-only W0 acceptance test for the deterministic tool boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import build_smoke_environment, replay_digest  # noqa: E402


def main() -> int:
    episode, executor = build_smoke_environment()

    found = executor.execute(episode, "search_orders", {"status": "paid"})
    assert found.ok and found.data["orders"][0]["order_id"] == "O-1001"

    cancelled = executor.execute(episode, "cancel_order", {"order_id": "O-1001"})
    assert cancelled.ok and cancelled.data["status"] == "cancelled"

    missing = executor.execute(episode, "cancel_order", {})
    assert not missing.ok and missing.error_code == "MISSING_REQUIRED"

    unknown_tool = executor.execute(episode, "send_email", {})
    assert not unknown_tool.ok and unknown_tool.error_code == "UNKNOWN_TOOL"

    calls = [("search_orders", {"status": "paid"}), ("cancel_order", {"order_id": "O-1001"})]
    digest_a = replay_digest(episode, executor, calls)
    digest_b = replay_digest(episode, executor, calls)
    assert digest_a == digest_b, "same task/seed/calls must replay identically"

    print(json.dumps({"status": "PASS", "replay_digest": digest_a, "tool_count": 2}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
