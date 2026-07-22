from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np

from ...config import settings


FaceEligibility = Literal["direct", "recoverable", "unusable", "none"]


class FaceQuality(TypedDict, total=False):
    category: str
    eligibility: FaceEligibility
    can_enroll: bool
    can_match: bool
    can_superres: bool
    enhanced: bool
    reason: str | None


def _frontalness(kps: np.ndarray) -> float:
    """用 5 点关键点估正脸度（0~1，越大越正）。

    kps 顺序：左眼、右眼、鼻、左嘴角、右嘴角。正脸时鼻子在两眼中线上、左右大致对称；
    侧脸时鼻子明显偏向一侧。用"鼻子到两眼水平中点的偏移 / 两眼间距"度量偏转，越小越正。
    """
    if kps is None or len(kps) < 3:
        return 0.0
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_dist = float(np.hypot(*(right_eye - left_eye))) or 1.0
    offset = abs(nose[0] - eye_mid_x) / eye_dist  # 0=正脸，越大越偏
    return float(max(0.0, 1.0 - offset))

def _blur_var(bgr: np.ndarray, bbox) -> float:
    """脸框内灰度拉普拉斯方差（清晰度代理，越大越清晰）。无 cv2 依赖，用 numpy。"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    crop = bgr[y1:y2, x1:x2].astype(np.float32).mean(axis=2)  # 灰度
    lap = (
        -4 * crop
        + np.roll(crop, 1, 0) + np.roll(crop, -1, 0)
        + np.roll(crop, 1, 1) + np.roll(crop, -1, 1)
    )
    return float(lap[1:-1, 1:-1].var()) if crop.size > 9 else 0.0

def _pose_angles(kps: np.ndarray) -> tuple[float, float]:
    """用 5 点关键点近似估计 (yaw, pitch)，单位度。

    kps 顺序：左眼、右眼、鼻、左嘴角、右嘴角。
    - yaw（左右转）：鼻子相对两眼水平中点的偏移 / 半眼距 → ×90。0=正脸，±90≈全侧脸。
    - pitch（上下，负=低头）：鼻子在「眼线→嘴线」纵向区间里的相对位置。正脸约 0.5；
      俯拍低头时鼻子上移(靠近眼)→比例<0.5→pitch 为负；抬头则相反。
    真值角需 3D/PnP，这里是可用的粗略代理（客户阈值 yaw~80/pitch~45 较宽松，足够门控/分桶）。
    """
    if kps is None or len(kps) < 5:
        return 0.0, 0.0
    le, re, nose, lm, rm = [np.asarray(p, dtype=np.float32) for p in kps[:5]]
    eye_mid = (le + re) / 2.0
    mouth_mid = (lm + rm) / 2.0
    eye_dist = float(np.hypot(*(re - le))) or 1.0
    yaw = float(np.clip((nose[0] - eye_mid[0]) / (eye_dist / 2.0) * 90.0, -90.0, 90.0))
    face_h = float(mouth_mid[1] - eye_mid[1])
    if abs(face_h) < 1e-3:
        return yaw, 0.0
    nose_rel = (nose[1] - eye_mid[1]) / face_h            # ~0.5 正脸
    pitch = float(np.clip((nose_rel - 0.5) * 200.0, -90.0, 90.0))  # <0=低头
    return yaw, pitch

def deep_fiqa_score(aligned_bgr: np.ndarray | None) -> float | None:
    """CR-FIQA预测人脸对身份识别的可用性，分数越大越可靠。"""
    if settings.face_fiqa_backend in {"off", "none", ""}:
        return None
    if aligned_bgr is None:
        return None
    from .fiqa.cr_fiqa import score

    return score(aligned_bgr)

def assess_quality(
    face: dict,
    bgr: np.ndarray | None = None,
    aligned_bgr: np.ndarray | None = None,
) -> dict:
    """评估一张脸质量，产出**分级类别**（对齐客户「人脸过滤」：主看模糊+角度）。

    - 模糊：拉普拉斯方差 `blur_var` + 关键点/检测置信度 `det_score`（+ 可插拔深度 FIQA）。
    - 角度：`yaw` / `pitch`（俯拍低头比抬头更不容忍）。
    返回 category ∈ {clear, marginal, poor}（唯一真源，产品降权 & 实验分桶都用它），
    并保留 quality 标量 / quality_ok（= 非 poor，向后兼容旧消费方）/ 各分量。
    """
    bbox = face["bbox"]
    w = max(0.0, bbox[2] - bbox[0])
    h = max(0.0, bbox[3] - bbox[1])
    area = w * h
    det = float(face.get("det_score", 0.0))
    kps = np.asarray(face["kps"]) if face.get("kps") is not None else None
    front = _frontalness(kps) if kps is not None else 0.0
    yaw, pitch = _pose_angles(kps) if kps is not None else (0.0, 0.0)
    blur = _blur_var(bgr, bbox) if bgr is not None else None
    fiqa = _deep_fiqa_score(aligned_bgr)

    # ---- 角度判级 ----
    yaw_bad = abs(yaw) >= settings.face_yaw_max
    pitch_bad = (pitch <= -settings.face_pitch_down_max) or (pitch >= settings.face_pitch_up_max)
    angle_clear = abs(yaw) <= settings.face_yaw_clear and abs(pitch) <= settings.face_pitch_clear

    # ---- 清晰度 / FIQA 判级 ----
    detection_bad = det < settings.face_min_det_score
    blur_bad = blur is not None and blur < settings.face_min_blur_var
    blur_degraded = blur is not None and blur < settings.face_blur_clear_var
    blur_clear = (blur is None or blur >= settings.face_blur_clear_var) and not detection_bad
    fiqa_bad = fiqa is not None and fiqa < settings.face_fiqa_poor_thresh
    fiqa_clear = fiqa is None or fiqa >= settings.face_fiqa_clear_thresh

    size_bad = min(w, h) < settings.face_min_size

    # ---- 合成类别：规则硬失败或FIQA过低→poor；全部清晰→clear；其余marginal ----
    if yaw_bad or pitch_bad or blur_bad or detection_bad or size_bad or fiqa_bad:
        category = "poor"
    elif angle_clear and blur_clear and fiqa_clear:
        category = "clear"
    else:
        category = "marginal"

    defects: list[str] = []
    if size_bad:
        defects.append("small_face")
    if yaw_bad:
        defects.append("extreme_yaw")
    elif abs(yaw) > settings.face_yaw_clear:
        defects.append("pose_yaw")
    if pitch_bad:
        defects.append("extreme_pitch")
    elif abs(pitch) > settings.face_pitch_clear:
        defects.append("pose_pitch")
    if blur_degraded:
        defects.append("blur")
    if detection_bad:
        defects.append("low_detection")
    if fiqa is not None and not fiqa_clear:
        defects.append("low_fiqa")
    if settings.face_fiqa_backend not in {"off", "none", ""} and fiqa is None:
        defects.append("fiqa_unavailable")

    reason = None
    if size_bad:
        reason = "too_small"
    elif yaw_bad:
        reason = "yaw_extreme"
    elif pitch_bad:
        reason = "pitch_down" if pitch < 0 else "pitch_up"
    elif blur_bad or detection_bad:
        reason = "too_blurry"
    elif fiqa_bad:
        reason = "low_fiqa"

    # 规则式连续分保留作诊断；启用FIQA后，融合/最佳脸优先使用FIQA识别可用性分数。
    size_term = min(1.0, area / float(settings.face_ref_area)) if settings.face_ref_area > 0 else 1.0
    rule_quality = det * (0.5 + 0.5 * front) * (0.5 + 0.5 * size_term)
    if blur is not None and settings.face_min_blur_var > 0:
        rule_quality *= min(1.0, blur / (settings.face_min_blur_var * 4))
    quality = (
        max(0.0, min(1.0, float(fiqa)))
        if fiqa is not None
        else float(rule_quality)
    )

    extreme_pose = yaw_bad or pitch_bad
    short_side = min(w, h)
    recoverable_size = (
        settings.face_recoverable_min_size
        <= short_side
        <= settings.face_superres_max_size
    )
    if extreme_pose or detection_bad or short_side < settings.face_recoverable_min_size:
        eligibility: FaceEligibility = "unusable"
    elif size_bad:
        eligibility = "recoverable"
    elif (blur_bad or fiqa_bad) and recoverable_size:
        eligibility = "recoverable"
    elif blur_degraded and recoverable_size:
        eligibility = "recoverable"
    elif blur_bad or fiqa_bad:
        eligibility = "unusable"
    else:
        eligibility = "direct"

    can_match = eligibility == "direct"
    can_superres = eligibility == "recoverable"
    can_enroll = eligibility == "direct" and category == "clear"
    superres_size_candidate = recoverable_size
    match_weight = max(0.0, min(1.0, quality))

    return {
        "det_score": round(det, 3),
        "area": int(area),
        "frontalness": round(front, 3),
        "yaw": round(yaw, 1),
        "pitch": round(pitch, 1),
        "blur_var": round(blur, 2) if blur is not None else None,
        "fiqa": round(fiqa, 3) if fiqa is not None else None,
        "rule_quality": round(float(rule_quality), 4),
        "quality": round(quality, 4),
        "category": category,
        "eligibility": eligibility,
        "short_side": round(short_side, 1),
        "quality_ok": category != "poor",  # 向后兼容：非 poor 视为可用
        "reason": reason,
        "defects": defects,
        "can_enroll": can_enroll,
        "can_match": can_match,
        "can_superres": can_superres,
        "superres_size_candidate": superres_size_candidate,
        "match_weight": round(match_weight, 4),
    }


def no_face_quality(reason: str = "not_detected") -> FaceQuality:
    return {
        "category": "none",
        "eligibility": "none",
        "can_enroll": False,
        "can_match": False,
        "can_superres": False,
        "reason": reason,
    }


def superres_quality_ok(
    fiqa_after: float | None,
    *,
    poor_threshold: float | None = None,
) -> tuple[bool, str | None]:
    """Product post-GFPGAN acceptance gate shared with offline experiments."""
    threshold = (
        settings.face_fiqa_poor_thresh
        if poor_threshold is None
        else float(poor_threshold)
    )
    if fiqa_after is not None and fiqa_after < threshold:
        return False, "fiqa_below_poor_threshold"
    return True, None


def face_gallery_quality_ok(quality: FaceQuality | None) -> tuple[bool, str | None]:
    if not quality:
        return False, "missing_face_quality"
    if quality.get("enhanced"):
        return False, "restored_face_not_enrollable"
    if quality.get("eligibility") != "direct":
        return False, "face_not_direct"
    if quality.get("category") != "clear" or not quality.get("can_enroll"):
        return False, "face_not_clear"
    return True, None


_deep_fiqa_score = deep_fiqa_score

__all__ = [
    "FaceEligibility",
    "FaceQuality",
    "assess_quality",
    "deep_fiqa_score",
    "face_gallery_quality_ok",
    "no_face_quality",
    "superres_quality_ok",
]
