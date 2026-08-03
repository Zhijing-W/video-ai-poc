from __future__ import annotations

import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for import_root in (REPO_ROOT, EXPERIMENT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from body_reid_experiment.runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
