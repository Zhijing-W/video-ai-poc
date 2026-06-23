"""兼容旧导入路径：转发到 app.core.config。"""
from .core.config import (
    ALLOWED_VIDEO_SUFFIXES,
    BASE_DIR,
    DATA_DIR,
    JOBS_DIR,
    MONITOR_DIR,
    Settings,
    TEMPLATES_DIR,
    settings,
)

__all__ = [
    "ALLOWED_VIDEO_SUFFIXES",
    "BASE_DIR",
    "DATA_DIR",
    "JOBS_DIR",
    "MONITOR_DIR",
    "Settings",
    "TEMPLATES_DIR",
    "settings",
]
