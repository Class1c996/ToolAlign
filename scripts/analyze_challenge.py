"""Summarize challenge benchmark results and failure modes."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "base": ROOT / "reports" / "challenge_base_v1.jsonl",
    "sft": ROOT / "reports" / "challenge_sft_v1.jsonl",
    "terminal_grpo": ROOT / "reports" / "challenge_terminal_v1.jsonl",
    "shaped_grpo": ROOT / "reports" / "challenge_shaped_v1.jsonl",
    "interactive_grpo": ROOT / "reports" / "challenge_interactive_v1_small.jsonl",
    "interactive_grpo_v2": ROOT / "reports" / "challenge_interactive_v2_small.jsonl",
    "targeted_sft_v1": ROOT / "reports" / "challenge_sft_targeted_v1.jsonl",
    "critical_sft_v2": ROOT / "reports" / "challenge_sft_critical_v2.jsonl",
}
OUT = ROOT / "reports" / "challenge_v1_summary.json"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def failure_kind(item: dict) -> str:
    task = item["task"]
    expected = task.get("expected_calls", [])
    actual = [{"tool": step["tool"], "arguments": step["arguments"]} for step in item.get("trace", [])]
    actions = item.get("parsed_actions", [])
    final = item.get("final_answer", "")
    if actual != expected:
        if any(step.get("result", {}).get("ok") is not True for step in item.get("trace", [])):
            return "tool_execution_error"
        return "plan_mismatch_extra_or_missing_call"
    if not actions or actions[-1].get("action") != "final":
        return "missing_final"
    expected_answer = [x.lower() for x in task.get("expected_answer_any", [])]
    if expected_answer and not any(x in final.lower() for x in expected_answer):
        return "wrong_final_answer"
    return "other"


def main() -> None:
    report = {"benchmark": "challenge_v1", "models": {}, "winner_by_success": None}
    for model, path in INPUTS.items():
        rows = load(path)
        by_template: dict[str, list[dict]] = defaultdict(list)
        failures = Counter()
        for item in rows:
            by_template[item["task"]["template_id"]].append(item)
            if not item["success"]:
                failures[failure_kind(item)] += 1
        report["models"][model] = {
            "count": len(rows),
            "format_legal_rate": round(sum(x["format_legal"] for x in rows) / len(rows), 4),
            "tool_success_rate": round(sum(x["tool_success"] for x in rows) / len(rows), 4),
            "end_to_end_success_rate": round(sum(x["success"] for x in rows) / len(rows), 4),
            "false_success_rate": round(sum(x["false_success"] for x in rows) / len(rows), 4),
            "average_calls": round(sum(x["call_count"] for x in rows) / len(rows), 4),
            "failure_modes": dict(sorted(failures.items())),
            "by_template": {
                name: {
                    "count": len(values),
                    "success_rate": round(sum(x["success"] for x in values) / len(values), 4),
                    "false_success_rate": round(sum(x["false_success"] for x in values) / len(values), 4),
                }
                for name, values in sorted(by_template.items())
            },
        }
    report["winner_by_success"] = max(report["models"], key=lambda name: report["models"][name]["end_to_end_success_rate"])
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT), "winner_by_success": report["winner_by_success"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
