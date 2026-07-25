"""Create counterfactual corrective SFT rows, separate from held-out challenge wording."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "critical_sft_v1.jsonl"
LEADS = [
    "Use the tools carefully.", "Complete the requested workflow.", "Please rely on the returned state.",
    "Handle the request without extra operations.", "Follow the instruction precisely.",
]
CASES = [
    ("protect_shipped", [
        "Retrieve O-1002 and explain why it must remain unchanged.",
        "Verify the status of O-1002; it is not eligible for cancellation once shipped.",
        "Check whether O-1002 can be cancelled and report the restriction.",
        "Look up O-1002, then give its shipping status without modifying it.",
        "Inspect O-1002 only; do not cancel a dispatched order.",
    ], ["get_order", "cancel_order", "search_orders"], [{"tool": "get_order", "arguments": {"order_id": "O-1002"}}], "Order O-1002 is shipped and cannot be cancelled."),
    ("search_flight_only", [
        "Find the lowest priced SHA-HND option for 2026-08-01, then describe it without purchasing.",
        "Search for the best fare from SHA to HND on 2026-08-01; no reservation is requested.",
        "Report the least expensive SHA to HND flight dated 2026-08-01, but do not book it.",
        "Look up fares for SHA-HND on 2026-08-01 and return the cheapest result only.",
        "Price a SHA to HND trip for 2026-08-01; this is an information request, not a booking.",
    ], ["search_flights", "create_booking", "cancel_booking"], [{"tool": "search_flights", "arguments": {"origin": "SHA", "destination": "HND", "date": "2026-08-01"}}], "The cheapest SHA to HND flight is F-3001 for 1800."),
    ("search_hotel_only", [
        "Search Tokyo accommodation for 2026-08-01 to 2026-08-03 and list an option; do not reserve it.",
        "Show lodging choices in Tokyo between 2026-08-01 and 2026-08-03 without creating a booking.",
        "Find a Tokyo hotel for the given dates, then report it without making a reservation.",
        "Look for hotels in Tokyo from August 1 through August 3; this is lookup-only.",
        "Provide a Tokyo hotel result for 2026-08-01/2026-08-03 and leave all bookings unchanged.",
    ], ["search_hotels", "search_flights", "create_booking"], [{"tool": "search_hotels", "arguments": {"city": "Tokyo", "check_in": "2026-08-01", "check_out": "2026-08-03"}}], "Kanda Inn is available in Tokyo; no reservation was made."),
    ("update_existing_event", [
        "Edit existing event E-2001: set its title to planning sync and its time to 15:00-16:00.",
        "Change E-2001 rather than creating another event; rename it planning sync for 15:00 to 16:00.",
        "Update the stored event with ID E-2001 to planning sync, 15:00-16:00.",
        "Modify event E-2001 in place: planning sync from 15:00 until 16:00.",
        "Do not create a new event. Revise E-2001 to planning sync at 15:00-16:00.",
    ], ["update_event", "create_event", "cancel_event", "list_events"], [{"tool": "update_event", "arguments": {"event_id": "E-2001", "title": "planning sync", "start_time": "15:00", "end_time": "16:00"}}], "Event E-2001 has been updated to planning sync from 15:00 to 16:00."),
]


def main() -> None:
    rows = []
    index = 0
    for template, prompts, tools, calls, answer in CASES:
        for lead in LEADS:
            for prompt in prompts:
                rows.append({"task_id": f"critical-train-{index:04d}", "seed": 15000 + index, "split": "train", "template_id": template, "domain": template.split("_")[0], "user": f"{lead} {prompt}", "allowed_tools": tools, "gold_calls": calls, "final_answer": answer, "goal": {"kind": "interactive_exact_trace"}, "tags": ["critical_sft", "counterfactual", "heldout_wording"]})
                index += 1
    OUT.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT), "count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
