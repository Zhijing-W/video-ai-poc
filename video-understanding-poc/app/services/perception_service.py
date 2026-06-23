"""细粒度感知服务（Phase 3 · Step 13）：用 Pose 派生躯干区，给人的颜色取得更准。

把 `pose.py`（关键点→躯干区）接到既有颜色链路：对每个 person 检测框，匹配同帧的一个
Pose 人体，取其躯干区（上衣区）作为取色区域，再复用 `color_utils.dominant_color`。
匹配用 IoU（Pose 与检测是两个模型，框不会完全重合，取重叠最大者）。

向后兼容：Pose 不可用 / 未开启 / 该人无可靠躯干区 → 回落到 Phase 2 的写死比例 "torso"，
最差不劣于原行为。整帧只跑一次 Pose，按需（有人 + POSE_COLOR 开）触发。
"""
from __future__ import annotations

from ..config import settings
from ..utils import COLOR_ZH, dominant_color
from .. import pose as pose_mod


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_pose(box: list[float], persons: list[dict]) -> dict | None:
    best, best_iou = None, 0.30  # 低于 0.3 视为没匹配上
    for p in persons:
        iou = _iou(box, p["box"])
        if iou > best_iou:
            best, best_iou = p, iou
    return best


def person_torso_color(img, box: list[float], persons: list[dict]) -> tuple[str | None, str]:
    """对一个 person 框求上衣主色。返回 (color, source)。

    source: "pose_full" / "pose_shoulders"（Pose 躯干区）/ "fallback_torso"（写死比例兜底）。
    """
    matched = _match_pose(box, persons) if persons else None
    if matched is not None:
        region = pose_mod.torso_region(matched["kpts"], matched["box"])
        if region is not None:
            torso_box, source = region
            color = dominant_color(img, torso_box, "whole")
            if color is not None:
                return color, source
    return dominant_color(img, box, "torso"), "fallback_torso"


def person_color_matches(
    img, box: list[float], target_color: str, persons: list[dict]
) -> tuple[bool, str | None]:
    """巡航比对用：对一个 person 框，用 Pose 躯干色判断是否为目标颜色。返回 (matched, detected)。"""
    detected, _ = person_torso_color(img, box, persons)
    if detected is None:
        return False, None
    target = (target_color or "").strip().lower()
    zh_to_en = {zh: en for en, zh in COLOR_ZH.items()}
    target = zh_to_en.get(target, target)
    return detected == target, detected


def enrich_person_colors_with_pose(img, detections: list[dict]) -> None:
    """就地给 detections 里的 person 补「基于 Pose 躯干区」的更准颜色。

    只处理 person；非 person 由 yolo_service 的既有逻辑负责。整帧只跑一次 Pose。
    """
    persons_det = [d for d in detections if d.get("label") == "person"]
    if not persons_det:
        return
    try:
        poses = pose_mod.estimate_persons(img)
    except Exception:
        poses = []  # Pose 不可用 → 全部走 fallback

    for d in persons_det:
        box = d.get("box")
        if not box:
            continue
        try:
            color, source = person_torso_color(img, box, poses)
        except Exception:
            continue
        if color:
            d["color"] = color
            d["color_zh"] = COLOR_ZH.get(color, color)
            d["color_source"] = source
