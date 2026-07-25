"""Strict action parser for model-produced tool calls and final answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedAction:
    action: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    answer: str | None = None
    error_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "tool": self.tool, "arguments": self.arguments, "answer": self.answer, "error_code": self.error_code}


def parse_action(text: str) -> ParsedAction:
    candidate = text.strip()
    tagged = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", candidate, flags=re.S)
    if tagged:
        candidate = tagged.group(1).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.S | re.I)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return ParsedAction("error", error_code="PARSE_ERROR")
    if not isinstance(value, dict):
        return ParsedAction("error", error_code="INVALID_ACTION")
    action = value.get("action")
    if action == "tool_call":
        if not isinstance(value.get("tool"), str) or not isinstance(value.get("arguments"), dict):
            return ParsedAction("error", error_code="INVALID_TOOL_CALL")
        return ParsedAction("tool_call", tool=value["tool"], arguments=value["arguments"])
    if action == "final" and isinstance(value.get("answer"), str):
        return ParsedAction("final", answer=value["answer"])
    return ParsedAction("error", error_code="INVALID_ACTION")
