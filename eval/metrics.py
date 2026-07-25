"""Metrics for executable trajectories."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize(rows: list[dict[str, Any]], include_groups: bool = True) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    def mean(key: str) -> float:
        return round(sum(float(row.get(key, 0)) for row in rows) / len(rows), 4)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get("split", "unknown")].append(row)
    summary = {
        "count": len(rows),
        "format_legal_rate": mean("format_legal"),
        "tool_success_rate": mean("tool_success"),
        "parameter_success_rate": mean("parameter_success"),
        "end_to_end_success_rate": mean("success"),
        "hallucinated_tool_rate": 1 - mean("tool_success"),
        "false_success_rate": mean("false_success"),
        "average_calls": mean("call_count"),
    }
    if include_groups:
        summary["by_split"] = {name: summarize(values, include_groups=False) for name, values in sorted(groups.items())}
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task_id", "split", "format_legal", "tool_success", "parameter_success", "success", "false_success", "call_count", "terminal_reward", "shaped_reward"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
