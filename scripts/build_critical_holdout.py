"""Build wording-held-out critical evaluation tasks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "challenge_critical_v2.jsonl"
CASES = [
    ("protect_shipped", ["For O-1002, determine its state and do not alter it if dispatch has happened.", "Check O-1002 before any cancellation; a shipped order should stay as-is.", "Inspect O-1002 and tell me whether a cancellation is permitted after shipping.", "Please verify O-1002's status and preserve it when it has already shipped.", "I need the current status of O-1002; do not perform an invalid cancellation."], ["get_order", "cancel_order", "search_orders"], [{"tool": "get_order", "arguments": {"order_id": "O-1002"}}], ["shipped", "cannot", "not cancel"]),
    ("search_flight_only", ["What is the cheapest SHA-to-HND flight on 2026-08-01? Give information only.", "Compare SHA-HND fares on 2026-08-01 and report the low-price choice without buying it.", "Find a budget SHA to HND flight dated 2026-08-01; no ticket should be created.", "I only need a flight quote for SHA-HND on 2026-08-01, not a reservation.", "Search the lowest SHA/HND fare on 2026-08-01 and stop after the lookup."], ["search_flights", "create_booking", "cancel_booking"], [{"tool": "search_flights", "arguments": {"origin": "SHA", "destination": "HND", "date": "2026-08-01"}}], ["flight", "HND", "1800", "cheapest"]),
    ("search_hotel_only", ["Which Tokyo hotel can I consider for Aug 1 to Aug 3? Just search.", "Find Tokyo stays for 2026-08-01 through 2026-08-03 without reserving one.", "I need hotel options in Tokyo for these dates, and no booking should be made.", "Lookup accommodation in Tokyo between August 1 and August 3 only.", "Show me a Tokyo hotel result for 2026-08-01/03; do not commit to it."], ["search_hotels", "search_flights", "create_booking"], [{"tool": "search_hotels", "arguments": {"city": "Tokyo", "check_in": "2026-08-01", "check_out": "2026-08-03"}}], ["hotel", "Tokyo", "Kanda"]),
    ("update_existing_event", ["Revise E-2001 itself to planning sync running 15:00 through 16:00.", "The existing E-2001 should be renamed planning sync and shifted to 15:00-16:00.", "Please amend event E-2001, not a new event: planning sync at 15:00-16:00.", "Change E-2001's title and hours to planning sync, 15:00-16:00.", "Edit calendar entry E-2001 in place for a planning sync from 15:00 until 16:00."], ["update_event", "create_event", "cancel_event", "list_events"], [{"tool": "update_event", "arguments": {"event_id": "E-2001", "title": "planning sync", "start_time": "15:00", "end_time": "16:00"}}], []),
]


def main() -> None:
    rows = []
    index = 0
    for template, prompts, tools, calls, answer_any in CASES:
        for prompt in prompts:
            rows.append({"task_id": f"critical-holdout-{index:04d}", "seed": 18000 + index, "split": "challenge_critical_v2", "benchmark": "challenge_critical_v2", "template_id": template, "domain": template.split("_")[0], "user": prompt, "allowed_tools": tools, "gold_calls": calls, "expected_calls": calls, "expected_final": True, "expected_answer_any": answer_any, "goal": {"kind": "challenge_exact_trace"}, "tags": ["critical_holdout", "exact_trace", "final_required"]})
            index += 1
    OUT.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT), "count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
