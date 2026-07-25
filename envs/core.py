"""Small, deterministic tool-execution boundary used by training and eval.

The first version deliberately has no network, shell, or filesystem side effects.
Domain tools can be added behind :class:`ToolSpec` without changing the router
or reward code that consumes the structured result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional


JSON = Dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    schema: JSON
    side_effects: str
    handler: Callable[[MutableMapping[str, Any], JSON], JSON]


@dataclass
class Episode:
    """An isolated state snapshot that can be reset and replayed."""

    task_id: str
    seed: int
    state: MutableMapping[str, Any]
    initial_state: MutableMapping[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.initial_state = copy.deepcopy(self.state)

    def reset(self) -> None:
        self.state = copy.deepcopy(self.initial_state)

    def snapshot(self) -> JSON:
        return copy.deepcopy(dict(self.state))


class ToolRegistry:
    def __init__(self, specs: Optional[list[ToolSpec]] = None) -> None:
        self._specs = {spec.name: spec for spec in (specs or [])}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def manifest(self) -> list[JSON]:
        return [
            {
                "name": spec.name,
                "version": spec.version,
                "schema": copy.deepcopy(spec.schema),
                "side_effects": spec.side_effects,
            }
            for spec in sorted(self._specs.values(), key=lambda item: item.name)
        ]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    tool_name: str
    data: JSON
    error_code: Optional[str] = None
    latency_ms: int = 0

    def as_dict(self) -> JSON:
        result: JSON = {
            "ok": self.ok,
            "tool_name": self.tool_name,
            "data": self.data,
            "latency_ms": self.latency_ms,
        }
        if self.error_code is not None:
            result["error_code"] = self.error_code
        return result


class ToolExecutor:
    """Validate and execute one registered tool inside an episode sandbox."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, episode: Episode, tool_name: str, arguments: Any) -> ToolResult:
        started = time.perf_counter()
        spec = self.registry.get(tool_name)
        if spec is None:
            return self._error(tool_name, "UNKNOWN_TOOL", started)
        if not isinstance(arguments, dict):
            return self._error(tool_name, "INVALID_JSON", started)
        validation_error = _validate_object_schema(arguments, spec.schema)
        if validation_error is not None:
            return self._error(tool_name, validation_error, started)
        try:
            data = spec.handler(episode.state, copy.deepcopy(arguments))
        except KeyError:
            return self._error(tool_name, "NOT_FOUND", started)
        except ValueError:
            return self._error(tool_name, "INVALID_ARGUMENT", started)
        return ToolResult(
            ok=True,
            tool_name=tool_name,
            data=_stable_json(data),
            latency_ms=_elapsed_ms(started),
        )

    @staticmethod
    def _error(tool_name: str, code: str, started: float) -> ToolResult:
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            data={},
            error_code=code,
            latency_ms=_elapsed_ms(started),
        )


def build_smoke_environment(task_id: str = "smoke-order", seed: int = 7) -> tuple[Episode, ToolExecutor]:
    """Build the minimal W0 order environment used by the smoke test."""
    rng = random.Random(seed)
    state: JSON = {
        "orders": [
            {"order_id": "O-1001", "status": "paid", "total": 129.0},
            {"order_id": "O-1002", "status": "shipped", "total": 59.0},
        ],
        "seed_marker": rng.randrange(1_000_000),
    }
    episode = Episode(task_id=task_id, seed=seed, state=state)
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="search_orders",
            version="0.1.0",
            schema={
                "type": "object",
                "properties": {"status": {"type": "string", "enum": ["paid", "shipped"]}},
                "required": [],
                "additionalProperties": False,
            },
            side_effects="read_only",
            handler=_search_orders,
        )
    )
    registry.register(
        ToolSpec(
            name="cancel_order",
            version="0.1.0",
            schema={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
            side_effects="mutates_episode_state",
            handler=_cancel_order,
        )
    )
    return episode, ToolExecutor(registry)


def replay_digest(episode: Episode, executor: ToolExecutor, calls: list[tuple[str, JSON]]) -> str:
    """Reset and replay calls, returning a stable digest of results and state."""
    episode.reset()
    results = [executor.execute(episode, name, args).as_dict() for name, args in calls]
    payload = {"results": results, "state": episode.snapshot()}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _search_orders(state: MutableMapping[str, Any], args: JSON) -> JSON:
    status = args.get("status")
    orders = state["orders"]
    if status is not None:
        orders = [order for order in orders if order["status"] == status]
    return {"orders": sorted(copy.deepcopy(orders), key=lambda order: order["order_id"])}


def _cancel_order(state: MutableMapping[str, Any], args: JSON) -> JSON:
    order_id = args["order_id"]
    for order in state["orders"]:
        if order["order_id"] == order_id:
            if order["status"] != "paid":
                raise ValueError("only paid orders can be cancelled")
            order["status"] = "cancelled"
            return {"order_id": order_id, "status": "cancelled"}
    raise KeyError(order_id)


def _validate_object_schema(value: Any, schema: Mapping[str, Any]) -> Optional[str]:
    if schema.get("type") == "object" and not isinstance(value, dict):
        return "INVALID_ARGUMENT"
    properties = schema.get("properties", {})
    for required in schema.get("required", []):
        if required not in value:
            return "MISSING_REQUIRED"
    if schema.get("additionalProperties") is False:
        unknown = set(value) - set(properties)
        if unknown:
            return "UNKNOWN_ARGUMENT"
    for key, item in value.items():
        expected = properties.get(key, {}).get("type")
        if expected == "string" and not isinstance(item, str):
            return "INVALID_TYPE"
        if expected == "number" and (not isinstance(item, (int, float)) or isinstance(item, bool)):
            return "INVALID_TYPE"
        if "enum" in properties.get(key, {}) and item not in properties[key]["enum"]:
            return "INVALID_ENUM"
    return None


def _stable_json(value: JSON) -> JSON:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))
