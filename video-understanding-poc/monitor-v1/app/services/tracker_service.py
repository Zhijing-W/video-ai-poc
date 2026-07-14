"""ByteTrack 跟踪服务（Phase 3 · Step 11）：复用 tracker 核心并对齐 service 层导出。

与 yolo_service 包装 detector 同理——把 app.tracker 的有状态跟踪能力暴露给 routers，
颜色着色直接复用 yolo_service.enrich_detection_colors（track 结果与 yolo 结果同形）。
"""
from __future__ import annotations

from ..tracker import active_sessions, reset_all_trackers, reset_tracker, track_objects
from .yolo_service import enrich_detection_colors

__all__ = [
    "active_sessions",
    "enrich_detection_colors",
    "reset_all_trackers",
    "reset_tracker",
    "track_objects",
]
