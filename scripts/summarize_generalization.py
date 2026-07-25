"""Write a compact, reproducible summary across in-distribution and holdout tests."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "challenge_v1": {
        "base": "reports/challenge_base_v1.jsonl",
        "sft": "reports/challenge_sft_v1.jsonl",
        "interactive_grpo_v2": "reports/challenge_interactive_v2_small.jsonl",
        "critical_sft_v2": "reports/challenge_sft_critical_v2.jsonl",
        "multitask_state_sft_v5": "reports/challenge_v1_sft_multitask_state_v5.jsonl",
        "interactive_grpo_multitask_v2": "reports/challenge_v1_grpo_interactive_multitask_v2_small.jsonl",
    },
    "critical_wording_holdout": {
        "targeted_sft_v1": "reports/critical_holdout_sft_targeted_v1.jsonl",
        "critical_sft_v2": "reports/critical_holdout_sft_critical_v2.jsonl",
        "multitask_state_sft_v5": "reports/critical_wording_sft_multitask_state_v5.jsonl",
    },
    "state_value_holdout": {
        "base": "reports/state_holdout_base.jsonl",
        "critical_sft_v2": "reports/state_holdout_sft_critical_v2.jsonl",
        "multitask_state_sft_v5": "reports/state_holdout_sft_multitask_state_v5.jsonl",
        "interactive_grpo_multitask_v2": "reports/state_holdout_grpo_interactive_multitask_v2_small.jsonl",
    },
}


def metrics(path: Path) -> dict[str, float | int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "count": len(rows),
        "format_legal_rate": round(sum(row["format_legal"] for row in rows) / len(rows), 4),
        "tool_success_rate": round(sum(row["tool_success"] for row in rows) / len(rows), 4),
        "end_to_end_success_rate": round(sum(row["success"] for row in rows) / len(rows), 4),
        "false_success_rate": round(sum(row["false_success"] for row in rows) / len(rows), 4),
    }


def main() -> None:
    report = {suite: {name: metrics(ROOT / path) for name, path in models.items()} for suite, models in SPECS.items()}
    report["conclusion"] = "The multitask state SFT V5 is the current best model: it improves over the base model on challenge_v1 and state/value holdout while retaining zero false-success rate. The small interactive GRPO run preserves task success but does not improve it and slightly lowers broad JSON legality, so it is retained as a negative ablation rather than the selected checkpoint."
    out = ROOT / "reports" / "generalization_summary.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
