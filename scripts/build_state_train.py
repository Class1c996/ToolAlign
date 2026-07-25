"""Build state/value-randomized corrective SFT data.

The generated identifiers, cities, dates, prices, and event contents are
deliberately disjoint from ``challenge_state_v3``.  Each row contains its own
environment state, so supervised tool-result turns are executable rather than
fixed demonstrations with swapped IDs.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "state_train_v1.jsonl"


def row(index: int, template_id: str, user: str, tools: list[str], calls: list[dict], answer: str, state_override: dict) -> dict:
    return {
        "task_id": f"state-train-{index:04d}",
        "seed": 31000 + index,
        "split": "train",
        "template_id": template_id,
        "domain": template_id.split("_")[0],
        "user": user,
        "allowed_tools": tools,
        "gold_calls": calls,
        "expected_calls": calls,
        "expected_answer_any": [answer.split()[0]],
        "final_answer": answer,
        "state_override": state_override,
        "goal": {"kind": "interactive_exact_trace"},
        "tags": ["state_randomized", "value_randomized", "counterfactual"],
    }


def main() -> None:
    rows: list[dict] = []
    cities = [("PEK", "ICN"), ("CAN", "BKK"), ("SZX", "SIN"), ("CTU", "KUL")]
    hotel_cities = ["Seoul", "Bangkok", "Busan", "Taipei", "Kuala Lumpur"]
    order_wording = [
        "Inspect {order_id} and report its status; do not attempt an ineligible cancellation.",
        "Check {order_id} first. It has left the warehouse, so preserve its current state.",
        "Tell me whether {order_id} can be cancelled after dispatch, without changing it.",
        "Look up {order_id} only and explain the cancellation restriction.",
    ]
    flight_wording = [
        "Find the cheapest {origin}-{destination} flight on {date}; this is information only.",
        "Compare {origin} to {destination} fares for {date}, but do not book anything.",
        "I need a {origin}/{destination} quote on {date}; stop after the search.",
        "Search the best fare from {origin} to {destination} dated {date}, with no reservation.",
    ]
    hotel_wording = [
        "Search {city} hotels from {check_in} to {check_out}; do not reserve one.",
        "Show accommodation options in {city} for {check_in} through {check_out}, lookup only.",
        "Find a hotel in {city} on those dates and leave bookings untouched: {check_in} to {check_out}.",
        "I need {city} lodging for {check_in}/{check_out}; return options but make no booking.",
    ]
    event_wording = [
        "Update existing event {event_id} itself to {title} from {start_time} to {end_time}.",
        "Do not create another event. Revise {event_id} to {title}, {start_time}-{end_time}.",
        "Edit {event_id} in place: set the title to {title} and hours to {start_time} through {end_time}.",
        "Amend calendar event {event_id} rather than adding one; make it {title} at {start_time}-{end_time}.",
    ]

    for i in range(60):
        order_id = f"O-{7001 + i:04d}"
        other_id = f"O-{7601 + i:04d}"
        state = {"orders": [
            {"order_id": other_id, "status": "paid", "total": 40.0 + i, "items": ["cable"]},
            {"order_id": order_id, "status": "shipped", "total": 80.0 + i, "items": ["accessory"]},
        ]}
        user = order_wording[i % len(order_wording)].format(order_id=order_id)
        rows.append(row(len(rows), "protect_shipped", user, ["get_order", "cancel_order", "search_orders"], [{"tool": "get_order", "arguments": {"order_id": order_id}}], f"{order_id} is shipped and cannot be cancelled.", state))

    for i in range(60):
        origin, destination = cities[i % len(cities)]
        date = f"2026-{10 + (i % 2):02d}-{1 + (i % 27):02d}"
        cheap_id, high_id = f"F-{7001 + i * 2:04d}", f"F-{7002 + i * 2:04d}"
        cheap_price = 1100 + 17 * i
        state = {"flights": [
            {"flight_id": high_id, "origin": origin, "destination": destination, "date": date, "price": cheap_price + 450},
            {"flight_id": cheap_id, "origin": origin, "destination": destination, "date": date, "price": cheap_price},
        ]}
        user = flight_wording[i % len(flight_wording)].format(origin=origin, destination=destination, date=date)
        calls = [{"tool": "search_flights", "arguments": {"origin": origin, "destination": destination, "date": date}}]
        rows.append(row(len(rows), "search_flight_only", user, ["search_flights", "create_booking", "cancel_booking"], calls, f"{cheap_id} is the cheapest {origin} to {destination} flight at {cheap_price}.", state))

    for i in range(60):
        city = hotel_cities[i % len(hotel_cities)]
        check_in = f"2026-{10 + (i % 2):02d}-{2 + (i % 25):02d}"
        check_out = f"2026-{10 + (i % 2):02d}-{4 + (i % 25):02d}"
        low_id, high_id = f"H-{7001 + i * 2:04d}", f"H-{7002 + i * 2:04d}"
        low_name, high_name = f"{city} Garden {i + 1}", f"{city} Plaza {i + 1}"
        low_price = 420 + 13 * i
        state = {"hotels": [
            {"hotel_id": high_id, "city": city, "name": high_name, "nightly_price": low_price + 230},
            {"hotel_id": low_id, "city": city, "name": low_name, "nightly_price": low_price},
        ]}
        user = hotel_wording[i % len(hotel_wording)].format(city=city, check_in=check_in, check_out=check_out)
        calls = [{"tool": "search_hotels", "arguments": {"city": city, "check_in": check_in, "check_out": check_out}}]
        rows.append(row(len(rows), "search_hotel_only", user, ["search_hotels", "search_flights", "create_booking"], calls, f"{low_name} is the lowest-priced {city} option at {low_price} per night; no reservation was made.", state))

    for i in range(60):
        event_id = f"E-{7001 + i:04d}"
        date = f"2026-{10 + (i % 2):02d}-{3 + (i % 24):02d}"
        start_hour = 10 + (i % 6)
        start_time, end_time = f"{start_hour:02d}:00", f"{start_hour + 1:02d}:00"
        title = f"planning review {i + 1}"
        state = {"events": [{"event_id": event_id, "title": f"old meeting {i + 1}", "date": date, "start_time": "09:00", "end_time": "10:00"}], "next_ids": {"return": 1, "event": 2, "booking": 2}}
        user = event_wording[i % len(event_wording)].format(event_id=event_id, title=title, start_time=start_time, end_time=end_time)
        calls = [{"tool": "update_event", "arguments": {"event_id": event_id, "title": title, "start_time": start_time, "end_time": end_time}}]
        rows.append(row(len(rows), "update_existing_event", user, ["update_event", "create_event", "cancel_event", "list_events"], calls, f"{event_id} was updated to {title} from {start_time} to {end_time}.", state))

    OUT.write_text("\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT), "count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
