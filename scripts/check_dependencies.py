from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.dependency_check import missing  # noqa: E402

print(json.dumps({"missing_training_packages": missing(), "status": "READY" if not missing() else "BLOCKED"}))
