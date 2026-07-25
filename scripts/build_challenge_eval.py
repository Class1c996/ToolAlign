"""Build a template-disjoint, execution-sensitive challenge benchmark."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "challenge_v1.jsonl"


def row(index: int, template_id: str, user: str, allowed_tools: list[str], expected_calls: list[dict], answer_any: list[str] | None = None) -> dict:
    return {
        "task_id": f"challenge-{index:04d}",
        "seed": 9000 + index,
        "split": "challenge_v1",
        "benchmark": "challenge_v1",
        "template_id": template_id,
        "domain": template_id.split("_")[0],
        "user": user,
        "allowed_tools": allowed_tools,
        "gold_calls": expected_calls,
        "expected_calls": expected_calls,
        "expected_final": True,
        "expected_answer_any": answer_any or [],
        "goal": {"kind": "challenge_exact_trace"},
        "tags": ["held_out_wording", "exact_trace", "final_required"],
    }


def build() -> list[dict]:
    tasks: list[dict] = []
    variants = [
        "Could you handle this for me?",
        "I need you to take care of the following.",
        "Please process this request carefully.",
        "Help me complete this operation.",
        "Can you work through this request?",
        "Please resolve this item.",
        "I would like this dealt with now.",
        "Take the necessary action on this request.",
        "Please do exactly what is needed here.",
        "Can you finish this task?",
    ]
    index = 0

    def add(template_id: str, body: str, tools: list[str], calls: list[dict], answer_any: list[str] | None = None) -> None:
        nonlocal index
        for prefix in variants:
            tasks.append(row(index, template_id, f"{prefix} {body}", tools, calls, answer_any))
            index += 1

    add(
        "orders_cancel_paid",
        "Cancel paid order O-1001, but inspect its current status before changing it.",
        ["search_orders", "get_order", "cancel_order", "check_refund"],
        [{"tool": "get_order", "arguments": {"order_id": "O-1001"}}, {"tool": "cancel_order", "arguments": {"order_id": "O-1001"}}],
    )
    add(
        "orders_do_not_cancel_shipped",
        "Check order O-1002. It must not be cancelled if it is already shipped; report what you found.",
        ["search_orders", "get_order", "cancel_order", "check_refund"],
        [{"tool": "get_order", "arguments": {"order_id": "O-1002"}}],
        ["shipped", "cannot", "not cancel", "not be cancelled"],
    )
    add(
        "orders_return_damaged",
        "Inspect delivered order O-1003 and start a damaged-item return.",
        ["get_order", "create_return", "check_refund", "cancel_order"],
        [{"tool": "get_order", "arguments": {"order_id": "O-1003"}}, {"tool": "create_return", "arguments": {"order_id": "O-1003", "reason": "damaged"}}],
    )
    add(
        "orders_missing_identifier",
        "Cancel my order for me, but ask for the order number because none was provided.",
        ["get_order", "cancel_order", "search_orders"],
        [],
        ["order number", "order id", "which order", "provide"],
    )
    add(
        "policy_no_tool",
        "Explain the general policy for returning an item; do not access an order.",
        [],
        [],
        ["return"],
    )
    add(
        "calendar_create_after_search",
        "Find a free one-hour slot on 2026-07-21 and schedule a design review from 10:00 to 11:00.",
        ["list_events", "find_free_slots", "create_event", "update_event", "cancel_event"],
        [{"tool": "find_free_slots", "arguments": {"date": "2026-07-21", "duration_minutes": 60}}, {"tool": "create_event", "arguments": {"title": "design review", "date": "2026-07-21", "start_time": "10:00", "end_time": "11:00"}}],
    )
    add(
        "calendar_reschedule",
        "Move team sync event E-2001 to 15:00-16:00 and rename it to planning sync.",
        ["list_events", "update_event", "cancel_event", "create_event"],
        [{"tool": "update_event", "arguments": {"event_id": "E-2001", "title": "planning sync", "start_time": "15:00", "end_time": "16:00"}}],
    )
    add(
        "calendar_cancel",
        "Remove the focus block event E-2002 from the calendar.",
        ["list_events", "cancel_event", "update_event", "create_event"],
        [{"tool": "cancel_event", "arguments": {"event_id": "E-2002"}}],
    )
    add(
        "travel_search_cheapest",
        "Look up the cheapest flight from SHA to HND on 2026-08-01 and report the result without booking.",
        ["search_flights", "create_booking", "cancel_booking", "search_hotels"],
        [{"tool": "search_flights", "arguments": {"origin": "SHA", "destination": "HND", "date": "2026-08-01"}}],
        ["flight", "HND", "1800", "cheapest"],
    )
    add(
        "travel_search_then_book",
        "Find the cheapest SHA-HND flight on 2026-08-01, then book the cheapest option for Lin.",
        ["search_flights", "create_booking", "cancel_booking", "search_hotels"],
        [{"tool": "search_flights", "arguments": {"origin": "SHA", "destination": "HND", "date": "2026-08-01"}}, {"tool": "create_booking", "arguments": {"flight_id": "F-3001", "passenger_name": "Lin"}}],
    )
    add(
        "travel_hotel_search",
        "Find Tokyo hotels for check-in 2026-08-01 and check-out 2026-08-03; do not reserve one.",
        ["search_hotels", "search_flights", "create_booking"],
        [{"tool": "search_hotels", "arguments": {"city": "Tokyo", "check_in": "2026-08-01", "check_out": "2026-08-03"}}],
        ["hotel", "Tokyo", "Kanda"],
    )
    add(
        "travel_weather",
        "Tell me the weather in Tokyo on 2026-08-01.",
        ["get_weather", "search_hotels", "search_flights"],
        [{"tool": "get_weather", "arguments": {"city": "Tokyo", "date": "2026-08-01"}}],
        ["sunny", "28", "Tokyo"],
    )
    return tasks


def main() -> None:
    tasks = build()
    OUT.write_text("\n".join(json.dumps(task, ensure_ascii=False, sort_keys=True) for task in tasks) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT), "count": len(tasks), "templates": len({x["template_id"] for x in tasks})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
