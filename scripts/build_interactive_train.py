"""Build hard interactive-training tasks with wording disjoint from challenge_v1."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "interactive_train_v1.jsonl"
PREFIXES = [
    "Please handle this request with the available tools.",
    "Work through this carefully and report when it is done.",
    "I need an accurate result here; inspect state before mutating it.",
    "Use the tool results to decide the next action.",
    "Please avoid unrelated tool calls.",
    "Resolve this request without guessing missing identifiers.",
    "Follow the required sequence and then summarize the outcome.",
    "Check the current state before taking any irreversible action.",
    "Complete only the operation described below.",
    "Please distinguish a lookup from a state-changing request.",
    "Take care with the preconditions in this request.",
    "Use only the necessary tools and stop after completion.",
    "Do not invent parameters; derive them from the request or tool result.",
    "The order of operations matters for this task.",
    "Please perform this as an executable tool workflow.",
    "Act conservatively and confirm the resulting state in your answer.",
    "Handle the request using the smallest valid tool sequence.",
    "Read the returned state before deciding whether to continue.",
    "Avoid changing anything that the user asked only to inspect.",
    "Please finish this workflow with a concise final response.",
]


def make(index: int, template_id: str, body: str, tools: list[str], calls: list[dict], answer_any: list[str] | None = None, final_answer: str = "Task completed.") -> dict:
    return {
        "task_id": f"itrain-{index:04d}",
        "seed": 12000 + index,
        "split": "train",
        "template_id": template_id,
        "domain": template_id.split("_")[0],
        "user": f"{PREFIXES[index % len(PREFIXES)]} {body}",
        "allowed_tools": tools,
        "gold_calls": calls,
        "expected_answer_any": answer_any or [],
        "final_answer": final_answer,
        "goal": {"kind": "interactive_exact_trace"},
        "tags": ["interactive_train", "distractor_tools", "multi_turn" if len(calls) > 1 else "single_turn"],
    }


def build() -> list[dict]:
    specs = [
        ("orders_cancel_paid", "Cancel paid order O-1001 after checking its status.", ["search_orders", "get_order", "cancel_order", "check_refund"], [{"tool": "get_order", "arguments": {"order_id": "O-1001"}}, {"tool": "cancel_order", "arguments": {"order_id": "O-1001"}}], [], "Order O-1001 has been cancelled."),
        ("orders_protect_shipped", "Inspect O-1002 and leave it unchanged because shipped orders cannot be cancelled.", ["search_orders", "get_order", "cancel_order", "check_refund"], [{"tool": "get_order", "arguments": {"order_id": "O-1002"}}], ["shipped", "cannot", "not cancel"], "Order O-1002 is shipped and cannot be cancelled."),
        ("orders_return", "Check delivered O-1003 and request a return with reason damaged.", ["get_order", "create_return", "check_refund", "cancel_order"], [{"tool": "get_order", "arguments": {"order_id": "O-1003"}}, {"tool": "create_return", "arguments": {"order_id": "O-1003", "reason": "damaged"}}], [], "A return for order O-1003 has been requested."),
        ("calendar_create", "Find a one-hour opening on 2026-07-21, then create design review from 10:00 to 11:00.", ["list_events", "find_free_slots", "create_event", "update_event", "cancel_event"], [{"tool": "find_free_slots", "arguments": {"date": "2026-07-21", "duration_minutes": 60}}, {"tool": "create_event", "arguments": {"title": "design review", "date": "2026-07-21", "start_time": "10:00", "end_time": "11:00"}}], [], "The design review has been scheduled."),
        ("calendar_reschedule", "Update E-2001: rename it planning sync and move it to 15:00-16:00.", ["list_events", "update_event", "cancel_event", "create_event"], [{"tool": "update_event", "arguments": {"event_id": "E-2001", "title": "planning sync", "start_time": "15:00", "end_time": "16:00"}}], [], "Event E-2001 has been updated to planning sync from 15:00 to 16:00."),
        ("travel_search", "Search flights SHA to HND on 2026-08-01 and report the cheapest option; do not book.", ["search_flights", "create_booking", "cancel_booking", "search_hotels"], [{"tool": "search_flights", "arguments": {"origin": "SHA", "destination": "HND", "date": "2026-08-01"}}], ["flight", "HND", "1800", "cheapest"], "The cheapest SHA to HND flight is F-3001 for 1800."),
        ("travel_book", "Search SHA to HND on 2026-08-01 and book the cheapest result for Lin.", ["search_flights", "create_booking", "cancel_booking", "search_hotels"], [{"tool": "search_flights", "arguments": {"origin": "SHA", "destination": "HND", "date": "2026-08-01"}}, {"tool": "create_booking", "arguments": {"flight_id": "F-3001", "passenger_name": "Lin"}}], [], "Flight F-3001 has been booked for Lin."),
        ("travel_hotel", "Look up Tokyo lodging from 2026-08-01 through 2026-08-03 without reserving anything.", ["search_hotels", "search_flights", "create_booking"], [{"tool": "search_hotels", "arguments": {"city": "Tokyo", "check_in": "2026-08-01", "check_out": "2026-08-03"}}], ["hotel", "Tokyo", "Kanda"], "Kanda Inn is available in Tokyo; no reservation was made."),
        ("orders_missing_identifier", "Ask for the missing order ID before attempting any cancellation.", ["get_order", "cancel_order", "search_orders"], [], ["order id", "order number", "which order", "provide"], "Please provide the order ID so I can continue."),
        ("policy_no_tool", "Explain the general return policy without looking up an order.", [], [], ["return"], "The general return policy is available; provide an order ID for a specific return."),
    ]
    tasks = []
    index = 0
    for template_id, body, tools, calls, answer_any, final_answer in specs:
        for _ in PREFIXES:
            tasks.append(make(index, template_id, body, tools, calls, answer_any, final_answer))
            index += 1
    return tasks


def main() -> None:
    tasks = build()
    OUT.write_text("\n".join(json.dumps(task, ensure_ascii=False, sort_keys=True) for task in tasks) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT), "count": len(tasks), "templates": len({x["template_id"] for x in tasks})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
