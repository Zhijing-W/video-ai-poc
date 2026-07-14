"""多线索融合服务层（Phase 3 · Step 15 / 3.5）：把 track_fusion 暴露给 routers。

与 tracker_service / gallery_service 同理——只做转发，不放业务逻辑。三时钟编排（Step 12）
将来在进程内直接调 `app.track_fusion`，HTTP 端点主要供独立验证/排查。
"""
from __future__ import annotations

from .. import track_fusion as _f

add_observation = _f.add_observation
resolve_track = _f.resolve_track
resolve_session = _f.resolve_session
reset_fusion = _f.reset_fusion
reset_all_fusion = _f.reset_all_fusion
active_fusion_sessions = _f.active_fusion_sessions

__all__ = [
    "active_fusion_sessions",
    "add_observation",
    "reset_all_fusion",
    "reset_fusion",
    "resolve_session",
    "resolve_track",
]
