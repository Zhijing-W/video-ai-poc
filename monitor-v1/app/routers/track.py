"""`/track` ByteTrack 多目标跟踪路由（Phase 3 · Step 11）。

与 `/detect`（无状态逐帧）不同：`/track` 是**有状态**的——按 `session_id` 维护跨帧
跟踪状态，返回的每个检测都带跨帧稳定的 `track_id`。前端按视频帧时序、用同一 session_id
连续调用即可获得稳定身份；换视频/重新开始时先调 `/track/reset`（或带 reset=true）。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..core import now_iso
from ..models import TrackRequest, TrackResetRequest
from ..services import enrich_detection_colors, reset_identity, reset_track_gate, reset_tracker, track_objects

router = APIRouter(tags=["track"])


@router.post("/track")
def track(req: TrackRequest) -> dict:
    if not req.image:
        raise HTTPException(status_code=400, detail="缺少 image 字段")
    if req.reset:
        reset_tracker(req.session_id)
        reset_track_gate(req.session_id)
        reset_identity(req.session_id)
    try:
        result = track_objects(req.image, session_id=req.session_id, conf=req.conf)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ByteTrack 跟踪失败：{exc}")
    enrich_detection_colors(req.image, result)
    return {"analyzed_at": now_iso(), **result}


@router.post("/track/reset")
def track_reset(req: TrackResetRequest) -> dict:
    # 同时清空 ByteTrack 跟踪状态、track 门控结论缓存、主体记忆（换视频/重开监控时）。
    existed = reset_tracker(req.session_id)
    reset_track_gate(req.session_id)
    reset_identity(req.session_id)
    return {"reset": existed, "session_id": req.session_id}
