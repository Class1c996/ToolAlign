"""README-compatible alias for the local evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_local import main  # noqa: E402

raise SystemExit(main())
