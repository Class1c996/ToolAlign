"""JSON and SQLite snapshot stores used for deterministic episode replay."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class JsonSnapshotStore:
    def save(self, path: str | Path, task_id: str, seed: int, state: dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"task_id": task_id, "seed": seed, "state": state}, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))


class SQLiteSnapshotStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS episodes (task_id TEXT PRIMARY KEY, seed INTEGER NOT NULL, state_json TEXT NOT NULL)")
            connection.commit()
        finally:
            connection.close()

    def save(self, task_id: str, seed: int, state: dict[str, Any]) -> None:
        payload = json.dumps(state, ensure_ascii=False, sort_keys=True)
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("INSERT OR REPLACE INTO episodes(task_id, seed, state_json) VALUES (?, ?, ?)", (task_id, seed, payload))
            connection.commit()
        finally:
            connection.close()

    def load(self, task_id: str) -> dict[str, Any]:
        connection = sqlite3.connect(self.path)
        try:
            row = connection.execute("SELECT task_id, seed, state_json FROM episodes WHERE task_id = ?", (task_id,)).fetchone()
        finally:
            connection.close()
        if row is None:
            raise KeyError(task_id)
        return {"task_id": row[0], "seed": row[1], "state": json.loads(row[2])}
