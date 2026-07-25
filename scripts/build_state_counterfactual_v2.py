"""Build a stronger, leakage-free state-generalization SFT corpus.

Every lookup prompt withholds the decisive state from the user message.  The
only way to produce a grounded answer is to make the declared tool call.  This
counteracts the direct-final-answer failure observed in the first state SFT.
No ``challenge_state_v3`` ID, city, date, price, or prompt is copied here.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "state_counterfactual_v2.jsonl"


def make(index: int, template: str, user: str, tools: list[str], calls: list[dict], final: str, state: dict) -> dict:
    return {
        "task_id": f"state-counterfactual-{index:04d}", "seed": 42000 + index,
        "split": "train", "template_id": template, "domain": template.split("_")[0],
        "user": user, "allowed_tools": tools, "gold_calls": calls, "final_answer": final,
        "state_override": state, "goal": {"kind": "interactive_exact_trace"},
        "tags": ["state_counterfactual_v2", "tool_first", "no_holdout_leakage"],
    }


def main() -> None:
    rows: list[dict] = []
    order_prompts = [
        "Retrieve the current record for {order_id}, then tell me whether cancellation is permitted. Do not change it.",
        "I need a grounded status check for {order_id}; inspect it before responding and leave the order untouched.",
        "Look up {order_id} and report its actual fulfilment state only. No mutation is requested.",
        "Before answering about cancellation eligibility for {order_id}, check the stored order record. Do not cancel it.",
    ]
    for i in range(100):
        order_id = f"O-{8101 + i:04d}"
        status = ["shipped", "delivered", "paid", "shipped", "shipped"][i % 5]
        state = {"orders": [
            {"order_id": order_id, "status": status, "total": 99.0 + i, "items": ["item"]},
            {"order_id": f"O-{8501 + i:04d}", "status": "paid", "total": 39.0 + i, "items": ["spare"]},
        ]}
        eligibility = "can be cancelled" if status == "paid" else "cannot be cancelled"
        rows.append(make(len(rows), "inspect_order_state", order_prompts[i % 4].format(order_id=order_id), ["get_order", "cancel_order", "search_orders"], [{"tool": "get_order", "arguments": {"order_id": order_id}}], f"{order_id} is {status} and {eligibility}.", state))

    routes = [("PVG", "TPE"), ("WUH", "NRT"), ("XIY", "HAN"), ("XMN", "MNL")]
    flight_prompts = [
        "Search fares from {origin} to {destination} on {date} and give the lowest result; do not purchase a ticket.",
        "Please look up, rather than book, the least expensive {origin}-{destination} option for {date}.",
        "I need the actual low fare for {origin} to {destination} dated {date}; this is a search-only request.",
        "Compare stored flights for {origin}/{destination} on {date}, then report the cheapest one without a reservation.",
    ]
    for i in range(100):
        origin, destination = routes[i % len(routes)]
        date = f"2026-{11 + (i % 2):02d}-{1 + (i % 26):02d}"
        low, high = f"F-{8101 + 2*i:04d}", f"F-{8102 + 2*i:04d}"
        price = 1250 + 19 * i
        state = {"flights": [
            {"flight_id": low, "origin": origin, "destination": destination, "date": date, "price": price},
            {"flight_id": high, "origin": origin, "destination": destination, "date": date, "price": price + 610},
        ]}
        call = {"tool": "search_flights", "arguments": {"origin": origin, "destination": destination, "date": date}}
        rows.append(make(len(rows), "search_flight_only", flight_prompts[i % 4].format(origin=origin, destination=destination, date=date), ["search_flights", "create_booking", "cancel_booking"], [call], f"{low} is the lowest {origin} to {destination} fare at {price}.", state))

    cities = ["Hanoi", "Jakarta", "Da Nang", "Penang", "Sapporo"]
    hotel_prompts = [
        "Search available hotels in {city} from {check_in} to {check_out}; report the lowest option and make no booking.",
        "Look up {city} accommodation for {check_in} through {check_out}. I only need options, not a reservation.",
        "Find the least expensive hotel actually listed for {city}, {check_in} to {check_out}, and leave bookings unchanged.",
        "Please query lodging in {city} for {check_in}/{check_out}; return a result without committing to it.",
    ]
    for i in range(100):
        city = cities[i % len(cities)]
        check_in = f"2026-{11 + (i % 2):02d}-{2 + (i % 24):02d}"
        check_out = f"2026-{11 + (i % 2):02d}-{4 + (i % 24):02d}"
        low, high = f"H-{8101 + 2*i:04d}", f"H-{8102 + 2*i:04d}"
        name, price = f"{city} Harbor {i + 1}", 510 + 11 * i
        state = {"hotels": [
            {"hotel_id": high, "city": city, "name": f"{city} Grand {i + 1}", "nightly_price": price + 280},
            {"hotel_id": low, "city": city, "name": name, "nightly_price": price},
        ]}
        call = {"tool": "search_hotels", "arguments": {"city": city, "check_in": check_in, "check_out": check_out}}
        rows.append(make(len(rows), "search_hotel_only", hotel_prompts[i % 4].format(city=city, check_in=check_in, check_out=check_out), ["search_hotels", "search_flights", "create_booking"], [call], f"{name} is the lowest priced {city} option at {price} per night; no reservation was made.", state))

    event_prompts = [
        "Modify existing event {event_id} in place: set it to {title} from {start} to {end}. Do not create a new event.",
        "Update the stored calendar item {event_id} itself to {title}, {start}-{end}; adding another event would be incorrect.",
        "Revise {event_id} rather than creating anything: title {title}, time {start} through {end}.",
        "Make an in-place edit to event {event_id}: {title} at {start}-{end}. Keep one event, not two.",
    ]
    for i in range(100):
        event_id = f"E-{8101 + i:04d}"
        start = f"{10 + (i % 7):02d}:00"
        end = f"{11 + (i % 7):02d}:00"
        title = f"delivery review {i + 1}"
        state = {"events": [{"event_id": event_id, "title": f"draft {i + 1}", "date": f"2026-12-{1 + (i % 25):02d}", "start_time": "09:00", "end_time": "10:00"}], "next_ids": {"return": 1, "event": 2, "booking": 2}}
        call = {"tool": "update_event", "arguments": {"event_id": event_id, "title": title, "start_time": start, "end_time": end}}
        rows.append(make(len(rows), "update_existing_event", event_prompts[i % 4].format(event_id=event_id, title=title, start=start, end=end), ["update_event", "create_event", "cancel_event", "list_events"], [call], f"{event_id} was updated to {title} from {start} to {end}.", state))

    # Keep the earlier randomized data and original corrective cases so the
    # adapter learns the same policy under both common and adversarial wording.
    for source in (ROOT / "data" / "processed" / "state_train_v1.jsonl", ROOT / "data" / "processed" / "critical_sft_v1.jsonl"):
        for line in source.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            item["task_id"] = f"state-mix-{len(rows):04d}"
            item["seed"] = 47000 + len(rows)
            rows.append(item)

    OUT.write_text("\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT), "count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
