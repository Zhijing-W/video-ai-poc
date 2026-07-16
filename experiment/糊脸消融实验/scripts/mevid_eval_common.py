"""兼容旧导入路径：共享实现已迁到 experiment/糊脸消融实验/common。"""
from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from common.mevid_eval_common import *  # noqa: F401,F403,E402
