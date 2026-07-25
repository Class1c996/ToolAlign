"""The W1 deterministic order, calendar, and travel tool catalog."""

from __future__ import annotations

import copy
from typing import Any, MutableMapping

from .core import Episode, ToolExecutor, ToolRegistry, ToolSpec


def build_standard_environment(task_id: str = "standard-episode", seed: int = 7, state_override: dict[str, Any] | None = None) -> tuple[Episode, ToolExecutor]:
    state = {
        "orders": [
            {"order_id": "O-1001", "status": "paid", "total": 129.0, "items": ["keyboard"]},
            {"order_id": "O-1002", "status": "shipped", "total": 59.0, "items": ["mouse"]},
            {"order_id": "O-1003", "status": "delivered", "total": 249.0, "items": ["monitor"]},
        ],
        "returns": [], "refunds": [],
        "events": [
            {"event_id": "E-2001", "title": "team sync", "date": "2026-07-21", "start_time": "09:00", "end_time": "10:00"},
            {"event_id": "E-2002", "title": "focus block", "date": "2026-07-21", "start_time": "13:00", "end_time": "14:00"},
        ],
        "flights": [
            {"flight_id": "F-3001", "origin": "SHA", "destination": "HND", "date": "2026-08-01", "price": 1800},
            {"flight_id": "F-3002", "origin": "SHA", "destination": "HND", "date": "2026-08-01", "price": 2200},
            {"flight_id": "F-3003", "origin": "PEK", "destination": "LAX", "date": "2026-08-02", "price": 4900},
        ],
        "hotels": [
            {"hotel_id": "H-4001", "city": "Tokyo", "name": "Kanda Inn", "nightly_price": 800},
            {"hotel_id": "H-4002", "city": "Tokyo", "name": "Bay Hotel", "nightly_price": 1200},
        ],
        "bookings": [{"booking_id": "B-5001", "flight_id": "F-3001", "passenger_name": "Lin", "status": "confirmed"}],
        "weather": {"Tokyo": {"2026-08-01": {"condition": "sunny", "temperature_c": 28}}},
        "next_ids": {"return": 1, "event": 3, "booking": 2},
    }
    if state_override:
        for key, value in state_override.items():
            state[key] = copy.deepcopy(value)
    episode = Episode(task_id=task_id, seed=seed, state=state)
    registry = ToolRegistry()
    for spec in _specs():
        registry.register(spec)
    return episode, ToolExecutor(registry)


def _object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


def _str(enum: list[str] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"type": "string"}
    if enum:
        value["enum"] = enum
    return value


def _specs() -> list[ToolSpec]:
    read, mutate = "read_only", "mutates_episode_state"
    return [
        ToolSpec("search_orders", "1.0.0", _object({"status": _str(["paid", "shipped", "delivered", "cancelled"])}), read, _search_orders),
        ToolSpec("get_order", "1.0.0", _object({"order_id": _str()}, ["order_id"]), read, _get_order),
        ToolSpec("cancel_order", "1.0.0", _object({"order_id": _str()}, ["order_id"]), mutate, _cancel_order),
        ToolSpec("create_return", "1.0.0", _object({"order_id": _str(), "reason": _str(["damaged", "unwanted", "wrong_item"])}, ["order_id", "reason"]), mutate, _create_return),
        ToolSpec("check_refund", "1.0.0", _object({"order_id": _str()}, ["order_id"]), read, _check_refund),
        ToolSpec("list_events", "1.0.0", _object({"date": _str()}), read, _list_events),
        ToolSpec("find_free_slots", "1.0.0", _object({"date": _str(), "duration_minutes": {"type": "number"}}, ["date", "duration_minutes"]), read, _find_free_slots),
        ToolSpec("create_event", "1.0.0", _object({"title": _str(), "date": _str(), "start_time": _str(), "end_time": _str()}, ["title", "date", "start_time", "end_time"]), mutate, _create_event),
        ToolSpec("update_event", "1.0.0", _object({"event_id": _str(), "title": _str(), "start_time": _str(), "end_time": _str()}, ["event_id"]), mutate, _update_event),
        ToolSpec("cancel_event", "1.0.0", _object({"event_id": _str()}, ["event_id"]), mutate, _cancel_event),
        ToolSpec("search_flights", "1.0.0", _object({"origin": _str(), "destination": _str(), "date": _str()}, ["origin", "destination", "date"]), read, _search_flights),
        ToolSpec("search_hotels", "1.0.0", _object({"city": _str(), "check_in": _str(), "check_out": _str()}, ["city", "check_in", "check_out"]), read, _search_hotels),
        ToolSpec("get_booking", "1.0.0", _object({"booking_id": _str()}, ["booking_id"]), read, _get_booking),
        ToolSpec("create_booking", "1.0.0", _object({"flight_id": _str(), "passenger_name": _str()}, ["flight_id", "passenger_name"]), mutate, _create_booking),
        ToolSpec("cancel_booking", "1.0.0", _object({"booking_id": _str()}, ["booking_id"]), mutate, _cancel_booking),
        ToolSpec("get_weather", "1.0.0", _object({"city": _str(), "date": _str()}, ["city", "date"]), read, _get_weather),
    ]


def _find(items: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any]:
    for item in items:
        if item.get(key) == value:
            return item
    raise KeyError(value)


def _search_orders(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    values = state["orders"]
    if args.get("status"):
        values = [x for x in values if x["status"] == args["status"]]
    return {"orders": sorted(copy.deepcopy(values), key=lambda x: x["order_id"])}


def _get_order(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(_find(state["orders"], "order_id", args["order_id"]))


def _cancel_order(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    order = _find(state["orders"], "order_id", args["order_id"])
    if order["status"] != "paid":
        raise ValueError("only paid orders can be cancelled")
    order["status"] = "cancelled"
    return {"order_id": order["order_id"], "status": order["status"]}


def _create_return(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    order = _find(state["orders"], "order_id", args["order_id"])
    if order["status"] not in {"delivered", "shipped"}:
        raise ValueError("order is not returnable")
    number = state["next_ids"]["return"]
    state["next_ids"]["return"] += 1
    result = {"return_id": f"R-{number:04d}", "order_id": order["order_id"], "status": "requested", "reason": args["reason"]}
    state["returns"].append(result)
    return copy.deepcopy(result)


def _check_refund(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    for refund in state["refunds"]:
        if refund["order_id"] == args["order_id"]:
            return copy.deepcopy(refund)
    return {"order_id": args["order_id"], "status": "not_started"}


def _list_events(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    events = state["events"]
    if args.get("date"):
        events = [x for x in events if x["date"] == args["date"]]
    return {"events": sorted(copy.deepcopy(events), key=lambda x: (x["date"], x["start_time"]))}


def _find_free_slots(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    events = sorted([x for x in state["events"] if x["date"] == args["date"]], key=lambda x: x["start_time"])
    occupied = {(x["start_time"], x["end_time"]) for x in events}
    slots = [{"start_time": "10:00", "end_time": "12:00"}, {"start_time": "14:00", "end_time": "17:00"}]
    if occupied:
        slots = [x for x in slots if (x["start_time"], x["end_time"]) not in occupied]
    return {"date": args["date"], "duration_minutes": args["duration_minutes"], "slots": slots}


def _create_event(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    number = state["next_ids"]["event"]
    state["next_ids"]["event"] += 1
    result = {"event_id": f"E-{2000 + number:04d}", **args}
    state["events"].append(result)
    return copy.deepcopy(result)


def _update_event(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    event = _find(state["events"], "event_id", args["event_id"])
    for key in ("title", "start_time", "end_time"):
        if key in args:
            event[key] = args[key]
    return copy.deepcopy(event)


def _cancel_event(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    event = _find(state["events"], "event_id", args["event_id"])
    state["events"].remove(event)
    return {"event_id": args["event_id"], "status": "cancelled"}


def _search_flights(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    values = [x for x in state["flights"] if all(x[k] == args[k] for k in ("origin", "destination", "date"))]
    return {"flights": sorted(copy.deepcopy(values), key=lambda x: x["price"])}


def _search_hotels(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    values = [x for x in state["hotels"] if x["city"].lower() == args["city"].lower()]
    return {"hotels": sorted(copy.deepcopy(values), key=lambda x: x["nightly_price"]), "check_in": args["check_in"], "check_out": args["check_out"]}


def _get_booking(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(_find(state["bookings"], "booking_id", args["booking_id"]))


def _create_booking(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    _find(state["flights"], "flight_id", args["flight_id"])
    number = state["next_ids"]["booking"]
    state["next_ids"]["booking"] += 1
    result = {"booking_id": f"B-{5000 + number:04d}", **args, "status": "confirmed"}
    state["bookings"].append(result)
    return copy.deepcopy(result)


def _cancel_booking(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    booking = _find(state["bookings"], "booking_id", args["booking_id"])
    booking["status"] = "cancelled"
    return copy.deepcopy(booking)


def _get_weather(state: MutableMapping[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    city = state["weather"].get(args["city"], {})
    return {"city": args["city"], "date": args["date"], **copy.deepcopy(city.get(args["date"], {"condition": "unknown"}))}
