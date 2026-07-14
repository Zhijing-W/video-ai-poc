"""`/identify` 主体记忆 / ReID 向量库路由（Phase 3 · Step 14）。

与 `/track`（Step 11，给目标稳定 track_id）配合：`/track` 回传带 track_id 的检测框后，
把整帧 + 这些框丢给 `/identify`，逐个提 ReID 指纹查库——命中即复用已知主体档案（省下
一次 LLM），未命中则开放集登记建档。**有状态**：按 session_id 隔离记忆库，换视频先 reset。

注意：本路由只做"认这一批 crop"，不决定"何时该来认"（那是三时钟编排 Step 12 的职责）。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..core import now_iso
from ..models import GalleryResetRequest, IdentifyRequest
from ..services import (
    gallery_backend_info,
    gallery_stats,
    identify_detections,
    reset_gallery,
)

router = APIRouter(tags=["identify"])


@router.post("/identify")
def identify(req: IdentifyRequest) -> dict:
    if not req.image:
        raise HTTPException(status_code=400, detail="缺少 image 字段")
    if req.reset:
        reset_gallery(req.session_id)

    # 归一化成 detections 列表：优先 detections，退而求其次用单个 box。
    if req.detections:
        dets = [d.model_dump() for d in req.detections]
    elif req.box:
        dets = [{"box": req.box, "track_id": req.track_id, "label": req.label}]
    else:
        raise HTTPException(status_code=400, detail="需要 detections 或 box 之一")

    try:
        result = identify_detections(
            req.image, dets, session_id=req.session_id, auto_enroll=req.auto_enroll
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"主体识别失败：{exc}")
    return {"analyzed_at": now_iso(), **result}


@router.get("/gallery/stats")
def stats(session_id: str = "default") -> dict:
    return {"backend_info": gallery_backend_info(), **gallery_stats(session_id)}


@router.post("/gallery/reset")
def gallery_reset(req: GalleryResetRequest) -> dict:
    existed = reset_gallery(req.session_id)
    return {"reset": existed, "session_id": req.session_id}
