"""`/detect` YOLO 检测路由。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..core import now_iso
from ..models import DetectRequest
from ..services import detect_objects, enrich_detection_colors

router = APIRouter(tags=["detect"])


@router.post("/detect")
def detect(req: DetectRequest) -> dict:
    if not req.image:
        raise HTTPException(status_code=400, detail="缺少 image 字段")
    try:
        result = detect_objects(req.image, conf=req.conf)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    enrich_detection_colors(req.image, result)
    return {"analyzed_at": now_iso(), **result}
