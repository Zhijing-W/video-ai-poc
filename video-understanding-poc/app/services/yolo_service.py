"""YOLO 检测服务：复用 detector 并补充主色信息。"""
from __future__ import annotations

from ..core import settings
from ..detector import _decode_image as decode_image
from ..detector import class_names, detect_objects
from ..utils import COLOR_ZH, dominant_color


def enrich_detection_colors(image: str, yolo: dict) -> None:
    detections = yolo.get("detections") or []
    if not detections:
        return
    try:
        img = decode_image(image)
    except Exception:
        return

    for detection in detections:
        try:
            region = "torso" if detection.get("label") == "person" else "whole"
            color = dominant_color(img, detection.get("box"), region)
        except Exception:
            color = None
        if color:
            detection["color"] = color
            detection["color_zh"] = COLOR_ZH.get(color, color)

    # Phase 3 · Step 13：开启 POSE_COLOR 时，用 Pose 躯干区覆盖 person 的颜色（更准）。
    if settings.pose_color:
        try:
            from .perception_service import enrich_person_colors_with_pose

            enrich_person_colors_with_pose(img, detections)
        except Exception:
            pass
