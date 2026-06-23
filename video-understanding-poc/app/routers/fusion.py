"""`/fusion` 多线索融合 + 最佳帧投票路由（Phase 3 · Step 15 / 3.5）。

在 `/track`(Step 11) + `/identify`(Step 14) 之上：把同一 track 跨帧的识别观测攒起来，
做最佳帧选择 + 多帧加权投票 + 多线索融合，给整条轨迹定一个**稳定**身份——别赌单帧。

有状态：按 session_id 隔离融合缓冲；换视频先 reset。本路由只做聚合裁决，不查库、不调 LLM；
"何时来喂观测/取裁决"由三时钟编排（Step 12）决定。
"""
from __future__ import annotations

from fastapi import APIRouter

from ..core import now_iso
from ..models import FusionObserveRequest, FusionResetRequest
from ..services import add_observation, reset_fusion, resolve_session, resolve_track

router = APIRouter(tags=["fusion"])


@router.post("/fusion/observe")
def fusion_observe(req: FusionObserveRequest) -> dict:
    if req.reset:
        reset_fusion(req.session_id)
    touched: set[int] = set()
    for ob in req.observations:
        add_observation(
            req.session_id,
            ob.track_id,
            frame_idx=ob.frame_idx,
            box=ob.box,
            quality=ob.quality,
            reid_subject=ob.reid_subject,
            reid_decision=ob.reid_decision,
            reid_score=ob.reid_score,
            color=ob.color,
        )
        touched.add(ob.track_id)
    results = [resolve_track(req.session_id, tid) for tid in sorted(touched)]
    return {"analyzed_at": now_iso(), "session_id": req.session_id, "results": results}


@router.get("/fusion/resolve")
def fusion_resolve(session_id: str = "default") -> dict:
    return {"analyzed_at": now_iso(), **resolve_session(session_id)}


@router.post("/fusion/reset")
def fusion_reset(req: FusionResetRequest) -> dict:
    existed = reset_fusion(req.session_id)
    return {"reset": existed, "session_id": req.session_id}
