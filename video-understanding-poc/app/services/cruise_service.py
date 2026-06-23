"""巡航裁决服务：YOLO 检测 + 廉价颜色校验。"""
from __future__ import annotations

from ..core import settings
from ..models import CruiseDecision, TargetPlan
from ..utils import color_matches
from .yolo_service import decode_image


def _plan_to_dict(plan: TargetPlan | dict | None) -> dict | None:
    if plan is None:
        return None
    if isinstance(plan, TargetPlan):
        return plan.model_dump()
    return plan


def apply_plan(image: str, yolo: dict, plan: TargetPlan | dict | None) -> dict | None:
    plan_dict = _plan_to_dict(plan)
    if not plan_dict:
        return None

    yolo_class = plan_dict.get("yolo_class")
    if not yolo_class:
        return None

    boxes = [
        detection
        for detection in yolo.get("detections", [])
        if detection.get("label") == yolo_class
    ]
    if not boxes:
        return CruiseDecision(
            is_match=False,
            reason=f"未检出 {yolo_class}",
            matched_boxes=[],
        ).model_dump()

    attribute = plan_dict.get("attribute") or {}
    if attribute.get("type") != "color":
        return CruiseDecision(
            is_match=True,
            reason=f"检出 {yolo_class}×{len(boxes)}",
            matched_boxes=[box["box"] for box in boxes],
        ).model_dump()

    img = decode_image(image)
    region = attribute.get("region", "whole")
    if yolo_class == "person" and region == "whole":
        region = "torso"
    target_color = attribute.get("value", "")

    # Phase 3 · Step 13：person 的颜色用 Pose 躯干区判定（更准）；整帧只跑一次 Pose。
    use_pose = bool(settings.pose_color) and yolo_class == "person"
    poses: list[dict] = []
    if use_pose:
        try:
            from .. import pose as pose_mod

            poses = pose_mod.estimate_persons(img)
        except Exception:
            use_pose = False

    matched_boxes: list[list[float]] = []
    detected_colors: list[str] = []
    for box in boxes:
        if use_pose:
            from .perception_service import person_color_matches

            matched, detected = person_color_matches(img, box["box"], target_color, poses)
        else:
            matched, detected = color_matches(img, box["box"], target_color, region)
        detected_colors.append(detected or "?")
        if matched:
            matched_boxes.append(box["box"])

    if matched_boxes:
        reason = f"检出 {yolo_class}×{len(boxes)}，其中 {len(matched_boxes)} 个为{target_color}色"
    else:
        reason = (
            f"检出 {yolo_class}×{len(boxes)}，颜色为 {','.join(detected_colors)}，非{target_color}"
        )
    return CruiseDecision(
        is_match=bool(matched_boxes),
        reason=reason,
        matched_boxes=matched_boxes,
    ).model_dump()
