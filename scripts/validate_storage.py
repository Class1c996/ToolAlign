from __future__ import annotations

import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs import build_standard_environment  # noqa: E402
from envs.store import JsonSnapshotStore, SQLiteSnapshotStore  # noqa: E402


def main() -> int:
    episode, _ = build_standard_environment("storage", 9)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        json_store = JsonSnapshotStore()
        json_store.save(root / "state.json", episode.task_id, episode.seed, episode.snapshot())
        assert json_store.load(root / "state.json")["state"] == episode.snapshot()
        sqlite_store = SQLiteSnapshotStore(root / "state.sqlite3")
        sqlite_store.save(episode.task_id, episode.seed, episode.snapshot())
        assert sqlite_store.load(episode.task_id)["state"] == episode.snapshot()
    print("storage PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
