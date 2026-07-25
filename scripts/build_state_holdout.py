"""Build a value- and state-held-out executable tool-use benchmark."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "challenge_state_v3.jsonl"


def task(index: int, template_id: str, user: str, tools: list[str], calls: list[dict], answer_any: list[str], state_override: dict) -> dict:
    return {"task_id": f"state-holdout-{index:04d}", "seed": 21000 + index, "split": "challenge_state_v3", "benchmark": "challenge_state_v3", "template_id": template_id, "domain": template_id.split("_")[0], "user": user, "allowed_tools": tools, "gold_calls": calls, "expected_calls": calls, "expected_final": True, "expected_answer_any": answer_any, "state_override": state_override, "goal": {"kind": "challenge_exact_trace"}, "tags": ["state_holdout", "value_holdout", "exact_trace"]}


def main() -> None:
    order_state = {"orders": [{"order_id": "O-9101", "status": "paid", "total": 89.0, "items": ["charger"]}, {"order_id": "O-9102", "status": "shipped", "total": 79.0, "items": ["headset"]}, {"order_id": "O-9103", "status": "delivered", "total": 299.0, "items": ["tablet"]}]}
    flight_state = {"flights": [{"flight_id": "F-9101", "origin": "SHA", "destination": "HND", "date": "2026-09-01", "price": 1600}, {"flight_id": "F-9102", "origin": "SHA", "destination": "HND", "date": "2026-09-01", "price": 2100}]}
    hotel_state = {"hotels": [{"hotel_id": "H-9201", "city": "Osaka", "name": "Namba Stay", "nightly_price": 700}, {"hotel_id": "H-9202", "city": "Osaka", "name": "Umeda Lodge", "nightly_price": 950}]}
    event_state = {"events": [{"event_id": "E-9301", "title": "status meeting", "date": "2026-09-02", "start_time": "09:00", "end_time": "10:00"}], "next_ids": {"return": 1, "event": 2, "booking": 2}}
    prompts = [
        ("protect_shipped", ["Inspect O-9102 and explain whether a cancellation is allowed.", "Check O-9102 first; it has already been dispatched, so avoid an invalid change.", "Report the current state of O-9102 without cancelling an ineligible order.", "Determine if O-9102 may be cancelled after shipment and tell me the result.", "Look up O-9102, preserving its state if shipping prevents cancellation."], ["get_order", "cancel_order", "search_orders"], [{"tool": "get_order", "arguments": {"order_id": "O-9102"}}], ["shipped", "cannot", "not cancel"], order_state),
        ("search_flight_only", ["Find the cheapest SHA-HND flight on 2026-09-01, information only.", "Compare the September 1 SHA to HND fares but do not purchase a ticket.", "Give me the low-price SHA/HND result for 2026-09-01 without booking it.", "I need a flight quote from SHA to HND on 2026-09-01, not a reservation.", "Search the best fare for SHA-HND on 2026-09-01 and stop after the lookup."], ["search_flights", "create_booking", "cancel_booking"], [{"tool": "search_flights", "arguments": {"origin": "SHA", "destination": "HND", "date": "2026-09-01"}}], ["F-9101", "1600", "HND", "cheapest"], flight_state),
        ("search_hotel_only", ["Which Osaka hotel can I consider from Sep 1 to Sep 3? Search only.", "Find accommodation in Osaka for 2026-09-01 through 2026-09-03 without reserving it.", "I need Osaka hotel options for these dates; leave bookings untouched.", "Look up Osaka lodging for September 1-3 only.", "Show a hotel in Osaka for 2026-09-01/03 and do not commit to it."], ["search_hotels", "search_flights", "create_booking"], [{"tool": "search_hotels", "arguments": {"city": "Osaka", "check_in": "2026-09-01", "check_out": "2026-09-03"}}], ["hotel", "Osaka", "Namba"], hotel_state),
        ("update_existing_event", ["Edit E-9301 itself into a roadmap review from 16:00 to 17:00.", "The existing E-9301 should be renamed roadmap review and moved to 16:00-17:00.", "Amend calendar event E-9301 in place: roadmap review at 16:00-17:00.", "Change E-9301's title and hours to roadmap review, 16:00-17:00.", "Do not create another event; revise E-9301 for a 16:00 to 17:00 roadmap review."], ["update_event", "create_event", "cancel_event", "list_events"], [{"tool": "update_event", "arguments": {"event_id": "E-9301", "title": "roadmap review", "start_time": "16:00", "end_time": "17:00"}}], [], event_state),
    ]
    rows = []
    index = 0
    for template, variants, tools, calls, answer_any, state in prompts:
        for user in variants:
            rows.append(task(index, template, user, tools, calls, answer_any, state))
            index += 1
    OUT.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT), "count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
