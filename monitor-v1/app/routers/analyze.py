"""`/analyze-frame` 分析路由。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..core import now_iso
from ..models import AnalyzeFrameRequest
from ..services import (
    analyze_frame_content,
    apply_plan,
    decide_gate,
    decide_track_gate,
    detect_objects,
    enrich_detection_colors,
    enrich_with_identity,
    record_llm_conclusion,
    record_reuse,
    synthesize_result_from_yolo,
    track_objects,
    yolo_signature,
)

router = APIRouter(tags=["analyze"])


def _analyze_frame_tracked(req: AnalyzeFrameRequest) -> dict:
    """Track 门控路径（Phase 3 · Step 12）：按活跃轨迹集合决定调 LLM / 复用结论。

    只有"新轨迹出生（新主体进入）"或心跳/比对才触发 gpt-4o；轨迹集合未变则复用上次结论，
    一次 LLM 都不调——这是 Phase 3 "认过一次就记住，整条轨迹共享"的省钱核心。
    """
    session_id = req.session_id or "default"
    try:
        yolo = track_objects(req.image, session_id=session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ByteTrack 跟踪失败：{exc}")
    enrich_detection_colors(req.image, yolo)

    # Phase 3 · "连"：在 track 流程上叠加"认人"——查主体记忆 + 融合，给检测补 subject_id，
    # 并产出身份摘要。失败绝不拖垮主流程（认人是增强维度，门控判定不依赖它）。
    try:
        identity = enrich_with_identity(session_id, req.image, yolo)
    except Exception:
        identity = None

    decision = decide_track_gate(session_id, yolo.get("detections", []), comparing=req.comparing)
    signature = yolo_signature(yolo)
    base = {
        "analyzed_at": now_iso(),
        "signature": signature,
        "yolo": yolo,
        "track_gate": decision,
        "identity": identity,
    }

    if decision["verdict"] == "skip":
        return {
            **base,
            "gated": False,
            "reused": False,
            "gate_reason": decision["reason"],
            "gate_priority": decision["priority"],
            "result": synthesize_result_from_yolo(yolo, decision["reason"]),
        }

    if decision["verdict"] == "reuse":
        record_reuse(session_id)
        cached = dict(decision["conclusion"] or {})
        cached["notification"] = "♻ 轨迹集合未变，复用上次 gpt-4o 结论（省一次调用）"
        return {
            **base,
            "gated": False,
            "reused": True,
            "gate_reason": decision["reason"],
            "gate_priority": decision["priority"],
            "result": cached,
        }

    # verdict == "pass"：新主体 / 心跳 / 比对 → 真正调 gpt-4o，并把结论挂到该会话供后续复用。
    try:
        result = analyze_frame_content(
            req.image,
            target=req.target,
            reference_image=req.reference_image,
            detections=yolo.get("detections"),
            img_w=yolo.get("img_w"),
            img_h=yolo.get("img_h"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    merged = list(result.get("detected_objects") or [])
    for label in yolo.get("counts", {}):
        if label not in merged:
            merged.append(label)
    result["detected_objects"] = merged
    record_llm_conclusion(session_id, result)

    cruise_match = apply_plan(req.image, yolo, req.plan.model_dump() if req.plan else None)
    return {
        **base,
        "gated": True,
        "reused": False,
        "gate_reason": decision["reason"],
        "gate_priority": decision["priority"],
        "result": result,
        "cruise_match": cruise_match,
    }


@router.post("/analyze-frame")
def analyze_frame(req: AnalyzeFrameRequest) -> dict:
    if not req.image:
        raise HTTPException(status_code=400, detail="缺少 image 字段")

    # Phase 3 · Step 12：开启 track 门控时走逐轨迹复用路径（向后兼容：默认关闭即旧逻辑）。
    if req.track_enabled:
        return _analyze_frame_tracked(req)

    try:
        yolo = detect_objects(req.image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"YOLO 检测失败：{exc}")
    enrich_detection_colors(req.image, yolo)

    gate = (
        decide_gate(
            yolo.get("counts", {}),
            prev_counts=req.prev_counts,
            since_last_llm_ms=req.since_last_llm_ms,
            comparing=req.comparing,
        )
        if req.gate_enabled
        else {"pass": True, "reason": "全帧模式（门控关闭）", "priority": "high", "signals": {}}
    )
    signature = yolo_signature(yolo)

    if not gate["pass"]:
        return {
            "analyzed_at": now_iso(),
            "gated": False,
            "gate_reason": gate["reason"],
            "gate_priority": gate["priority"],
            "signature": signature,
            "yolo": yolo,
            "result": synthesize_result_from_yolo(yolo, gate["reason"]),
        }

    reuse_ok = (
        req.last_llm_signature is not None
        and signature == req.last_llm_signature
        and gate["priority"] != "low"
        and not gate["signals"].get("hit_key_classes")
    )
    if reuse_ok:
        return {
            "analyzed_at": now_iso(),
            "gated": False,
            "reused": True,
            "gate_reason": "复用上次 gpt-4o 结论（YOLO 签名未变，省一次调用）",
            "gate_priority": gate["priority"],
            "signature": signature,
            "yolo": yolo,
            "result": None,
        }

    try:
        result = analyze_frame_content(
            req.image,
            target=req.target,
            reference_image=req.reference_image,
            detections=yolo.get("detections"),
            img_w=yolo.get("img_w"),
            img_h=yolo.get("img_h"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    merged = list(result.get("detected_objects") or [])
    for label in yolo.get("counts", {}):
        if label not in merged:
            merged.append(label)
    result["detected_objects"] = merged

    cruise_match = apply_plan(req.image, yolo, req.plan.model_dump() if req.plan else None)
    return {
        "analyzed_at": now_iso(),
        "gated": True,
        "gate_reason": gate["reason"],
        "gate_priority": gate["priority"],
        "signature": signature,
        "yolo": yolo,
        "result": result,
        "cruise_match": cruise_match,
    }
