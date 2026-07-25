"""Deterministic self-built task templates and split generation."""

from __future__ import annotations

import random
from typing import Any


def generate_tasks(count: int = 1200, seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    for index in range(count):
        template = _template(index, rng)
        if template["domain"] == "travel" and index % 10 < 7:
            template = _order_template(index, rng)
        template.update({"task_id": f"task-{index:05d}", "seed": seed + index, "split": _split(index)})
        tasks.append(template)
    return tasks


def _split(index: int) -> str:
    bucket = index % 10
    return "train" if bucket < 7 else "dev" if bucket == 7 else "test_seen" if bucket == 8 else "test_unseen"


def _template(index: int, rng: random.Random) -> dict[str, Any]:
    return [_order_template, _return_template, _calendar_template, _clarification_template, _irrelevance_template, _travel_template][index % 6](index, rng)


def _base(template_id: str, domain: str, user: str, tools: list[str], calls: list[dict[str, Any]], goal: dict[str, Any], tags: list[str]) -> dict[str, Any]:
    return {"template_id": template_id, "domain": domain, "user": user, "allowed_tools": tools, "gold_calls": calls, "goal": goal, "tags": tags}


def _order_template(index: int, rng: random.Random) -> dict[str, Any]:
    order_id = "O-1001" if index % 2 == 0 else "O-1002"
    return _base("order_lookup_or_cancel", "orders", f"Please check order {order_id} and cancel it if possible.", ["get_order", "cancel_order"], [{"tool": "get_order", "arguments": {"order_id": order_id}}, {"tool": "cancel_order", "arguments": {"order_id": order_id}}], {"kind": "order_status", "order_id": order_id, "status": "cancelled" if order_id == "O-1001" else "shipped"}, ["multi_step", "stop_after_complete"])


def _return_template(index: int, rng: random.Random) -> dict[str, Any]:
    return _base("create_return", "orders", "The monitor arrived; please start a return for order O-1003 because it is damaged.", ["create_return"], [{"tool": "create_return", "arguments": {"order_id": "O-1003", "reason": "damaged"}}], {"kind": "return_created", "order_id": "O-1003"}, ["single_step", "state_change"])


def _calendar_template(index: int, rng: random.Random) -> dict[str, Any]:
    date, title = "2026-07-21", f"project review {index:04d}"
    return _base("find_slot_create_event", "calendar", f"Find time on {date} for a 60 minute {title} and schedule it.", ["find_free_slots", "create_event"], [{"tool": "find_free_slots", "arguments": {"date": date, "duration_minutes": 60}}, {"tool": "create_event", "arguments": {"title": title, "date": date, "start_time": "10:00", "end_time": "11:00"}}], {"kind": "event_created", "title": title, "date": date}, ["multi_step", "state_change"])


def _clarification_template(index: int, rng: random.Random) -> dict[str, Any]:
    return _base("missing_required_clarification", "orders", "Please cancel my order.", ["cancel_order"], [], {"kind": "clarification", "required": ["order_id"]}, ["clarification", "no_guessing"])


def _irrelevance_template(index: int, rng: random.Random) -> dict[str, Any]:
    return _base("irrelevant_request", "none", "What is the general return policy?", [], [], {"kind": "no_tool", "answer_contains": "return"}, ["irrelevance", "no_tool"])


def _travel_template(index: int, rng: random.Random) -> dict[str, Any]:
    return _base("flight_search_book", "travel", "Find the cheapest SHA to HND flight on 2026-08-01 and book it for Lin.", ["search_flights", "create_booking"], [{"tool": "search_flights", "arguments": {"origin": "SHA", "destination": "HND", "date": "2026-08-01"}}, {"tool": "create_booking", "arguments": {"flight_id": "F-3001", "passenger_name": "Lin"}}], {"kind": "booking_created", "passenger_name": "Lin"}, ["unseen_tools", "multi_step", "state_change"])
