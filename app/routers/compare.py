"""`/compile-target` 与 `/cruise-frame` 路由。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..core import now_iso
from ..models import CompileTargetRequest, CruiseRequest
from ..services import (
    apply_plan,
    class_names,
    compile_target_rule,
    detect_objects,
    enrich_detection_colors,
)

router = APIRouter(tags=["compare"])


@router.post("/compile-target")
def compile_target(req: CompileTargetRequest) -> dict:
    if not (req.target or "").strip():
        raise HTTPException(status_code=400, detail="缺少 target 字段")
    try:
        plan = compile_target_rule(req.target, class_names(), req.reference_image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"目标编译失败：{exc}")
    return {"compiled_at": now_iso(), "plan": plan}


@router.post("/cruise-frame")
def cruise_frame(req: CruiseRequest) -> dict:
    if not req.image:
        raise HTTPException(status_code=400, detail="缺少 image 字段")
    try:
        yolo = detect_objects(req.image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"YOLO 检测失败：{exc}")
    enrich_detection_colors(req.image, yolo)
    cruise = apply_plan(req.image, yolo, req.plan.model_dump())
    return {"analyzed_at": now_iso(), "yolo": yolo, "cruise": cruise}
