"""Minimal TRL GRPO entry point with explicit dependency and checkpoint gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dependency_check import missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_terminal.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/sft")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    missing_packages = missing()
    if missing_packages:
        print(json.dumps({"status": "BLOCKED", "reason": "training_dependencies_missing", "missing": missing_packages, "config": args.config}))
        return 2
    import yaml
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from envs import build_standard_environment
    from training.data import format_prompt
    from training.rollout_reward import shaped_reward, terminal_reward

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_dir = Path(config.get("data_dir", "data/processed"))
    dataset = load_dataset("json", data_files={"train": str(data_dir / "train.jsonl")}, split="train")
    max_gold_calls = config.get("max_gold_calls")
    if max_gold_calls is not None:
        dataset = dataset.filter(lambda row: len(row.get("gold_calls", [])) <= int(max_gold_calls))
    tokenizer = AutoTokenizer.from_pretrained(config.get("base_model", "models/Qwen3-1.7B"), local_files_only=True)
    _, manifest_executor = build_standard_environment()
    manifest = manifest_executor.registry.manifest()
    dataset = dataset.map(
        lambda row: {
            "prompt": format_prompt(row, manifest, tokenizer),
            "task_id": row["task_id"],
            "seed": row["seed"],
            "task_json": json.dumps(row, ensure_ascii=False, sort_keys=True),
        },
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    # A saved PEFT adapter is marked inference-only by default.  Load it with
    # is_trainable=True so GRPO can update the SFT adapter instead of producing
    # a loss tensor with no autograd graph.
    base_model = config.get("base_model", "models/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=True)
    train_args = GRPOConfig(output_dir=config["output_dir"], per_device_train_batch_size=config.get("per_device_train_batch_size", config["num_generations"]), gradient_accumulation_steps=1, learning_rate=1e-5, max_steps=1 if args.smoke else config.get("max_steps", 1), num_generations=config["num_generations"], max_completion_length=config["max_completion_length"], beta=config["beta"], use_vllm=config["use_vllm"], report_to=[])
    reward_func = terminal_reward if config.get("reward") == "terminal_only" else shaped_reward
    trainer = GRPOTrainer(model=model, reward_funcs=[reward_func], args=train_args, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(config["output_dir"])
    print(json.dumps({"status": "PASS", "mode": "smoke" if args.smoke else "full", "config": args.config, "checkpoint": args.checkpoint, "output_dir": config["output_dir"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
