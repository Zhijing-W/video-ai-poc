from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

_FACE_ELIGIBILITY_RANK = {
    "none": 0,
    "unusable": 1,
    "recoverable": 2,
    "direct": 3,
}
_FACE_CATEGORY_RANK = {"none": 0, "poor": 1, "marginal": 2, "clear": 3}


def body_quality_score(quality: dict) -> float:
    blur_var = float(quality.get("blur_var") or 0.0)
    area = float(quality.get("area") or 0.0)
    return blur_var * min(1.0, area / 20000.0)


def face_candidate_proxy(
    person_crop: Image.Image,
    *,
    person_bbox: list[float],
    image_size: tuple[int, int],
    detection_confidence: float,
) -> float:
    """Cheap face-frame proxy using only the already available person crop."""
    width, height = person_crop.size
    if width < 4 or height < 8:
        return 0.0

    head_height = max(4, min(height, int(round(height * 0.38))))
    head = np.asarray(person_crop.crop((0, 0, width, head_height)).convert("L"), dtype=np.float32)
    lap = (
        -4 * head
        + np.roll(head, 1, 0)
        + np.roll(head, -1, 0)
        + np.roll(head, 1, 1)
        + np.roll(head, -1, 1)
    )
    sharpness = float(lap[1:-1, 1:-1].var()) if head.size > 9 else 0.0

    x1, y1, x2, y2 = [float(value) for value in person_bbox[:4]]
    image_width, image_height = image_size
    clipped = x1 <= 1 or y1 <= 1 or x2 >= image_width - 1 or y2 >= image_height - 1
    clipping_penalty = 0.65 if clipped else 1.0
    scale = min(1.0, max(0.0, width * head_height / 6000.0))
    return sharpness * scale * max(0.0, min(1.0, detection_confidence)) * clipping_penalty


def update_face_candidates(
    candidates: list[dict],
    candidate: dict,
    *,
    top_k: int,
    min_gap_frames: int,
) -> list[dict]:
    """Keep a temporally diverse, score-ranked bounded candidate set."""
    if top_k <= 0:
        return []
    gap = max(1, min_gap_frames)
    ordered = sorted(
        [*(dict(item) for item in candidates), dict(candidate)],
        key=lambda item: (
            float(item.get("proxy_score") or 0.0),
            -int(item["frame_index"]),
        ),
        reverse=True,
    )
    selected = []
    for item in ordered:
        frame_index = int(item["frame_index"])
        if any(
            abs(int(existing["frame_index"]) - frame_index) < gap
            for existing in selected
        ):
            continue
        selected.append(item)
        if len(selected) >= top_k:
            break
    return selected


def ensure_body_fallback(candidates: list[dict], body_best: dict | None, *, top_k: int) -> list[dict]:
    if not body_best or top_k <= 0:
        return list(candidates)
    frame_index = int(body_best["frame_index"])
    if any(int(item["frame_index"]) == frame_index for item in candidates):
        return list(candidates)
    fallback = {
        "track_id": body_best["track_id"],
        "frame_index": frame_index,
        "timestamp": body_best["timestamp"],
        "person_bbox": list(body_best["person_bbox"]),
        "proxy_score": -1.0,
        "det_confidence": body_best.get("det_confidence"),
        "fallback": "body_best",
    }
    if len(candidates) < top_k:
        return [*candidates, fallback]
    return [*candidates[:-1], fallback]


def face_evidence_rank(candidate: dict) -> tuple:
    """Canonical product ordering for already evaluated face evidence."""
    face = candidate.get("_face") or candidate.get("face") or {}
    quality = face.get("quality") or candidate.get("quality") or {}
    return (
        _FACE_ELIGIBILITY_RANK.get(quality.get("eligibility", "unusable"), 0),
        _FACE_CATEGORY_RANK.get(quality.get("category", "poor"), 0),
        float(quality.get("quality") or 0.0),
        float(
            face.get("association_score")
            or candidate.get("association_score")
            or 0.0
        ),
        float(face.get("det_score") or candidate.get("det_score") or 0.0),
        -int(candidate.get("frame_index", 0)),
    )


def public_evidence(evidence: dict | None) -> dict | None:
    if not evidence:
        return None
    private = {"crop", "person_crop", "embedding", "_face", "_aligned"}
    return {
        key: _public_value(value)
        for key, value in evidence.items()
        if key not in private
    }


def _public_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _public_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public_value(item) for item in value]
    return value


__all__ = [
    "body_quality_score",
    "ensure_body_fallback",
    "face_candidate_proxy",
    "face_evidence_rank",
    "public_evidence",
    "update_face_candidates",
]
