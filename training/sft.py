"""QLoRA SFT entry point; dependency-gated so environment tests stay CPU-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dependency_check import missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_qwen3_1p7b.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()
    missing_packages = missing()
    if missing_packages:
        print(json.dumps({"status": "BLOCKED", "reason": "training_dependencies_missing", "missing": missing_packages, "config": args.config}))
        return 2
    import yaml
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTConfig, SFTTrainer

    from envs import build_standard_environment
    from training.data import build_sft_rows

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    _, executor = build_standard_environment()
    task_path = Path(config.get("train_file", "data/processed/train.jsonl"))
    tasks = [json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.smoke:
        tasks = tasks[:2]
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    dataset = Dataset.from_list(build_sft_rows(tasks, executor.registry.manifest(), tokenizer))
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], quantization_config=quant, device_map="auto")
    model.config.use_cache = False
    adapter = config.get("adapter")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        if config["gradient_checkpointing"]:
            model.enable_input_require_grads()
        lora = None
    else:
        lora = LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"], target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]), task_type="CAUSAL_LM")
    training_args = SFTConfig(output_dir=config["output_dir"], per_device_train_batch_size=config["micro_batch_size"], gradient_accumulation_steps=config["gradient_accumulation_steps"], learning_rate=config["learning_rate"], max_steps=2 if args.smoke else config["max_steps"], gradient_checkpointing=config["gradient_checkpointing"], bf16=True, logging_steps=1, save_steps=50, report_to=[], max_length=config["max_seq_length"], dataset_text_field="text")
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset, processing_class=tokenizer, peft_config=lora)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(config["output_dir"])
    print(json.dumps({"status": "PASS", "mode": "smoke" if args.smoke else "full", "config": args.config, "rows": len(tasks), "output_dir": config["output_dir"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
