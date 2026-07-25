from __future__ import annotations

import importlib.util


REQUIRED = ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate")


def missing() -> list[str]:
    return [name for name in REQUIRED if importlib.util.find_spec(name) is None]
