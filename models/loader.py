"""Optional QLoRA model loader kept separate from the deterministic environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    base_model: str = "Qwen/Qwen3-1.7B"
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"


def load_model_and_tokenizer(config: ModelConfig) -> tuple[Any, Any]:
    """Load a model only when the optional training stack is installed."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("install torch and transformers before loading a model") from exc
    dtype = getattr(torch, config.compute_dtype)
    quantization_config = BitsAndBytesConfig(load_in_4bit=config.load_in_4bit, bnb_4bit_quant_type=config.quant_type, bnb_4bit_use_double_quant=config.double_quant, bnb_4bit_compute_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForCausalLM.from_pretrained(config.base_model, quantization_config=quantization_config, device_map="auto")
    return model, tokenizer
