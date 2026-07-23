from __future__ import annotations

from collections.abc import Callable

import numpy as np


class FaceRecord(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def geometry_descriptor(landmarks_3d: np.ndarray) -> np.ndarray | None:
    """Build a pose/scale-invariant descriptor from 68 3D landmarks."""
    if landmarks_3d is None or landmarks_3d.shape[0] < 68:
        return None
    points = landmarks_3d.astype(np.float32).copy()
    points -= points.mean(axis=0, keepdims=True)
    scale = float(np.sqrt((points ** 2).sum(axis=1).mean()))
    if scale <= 1e-6:
        return None
    points /= scale
    index_pairs = [
        (36, 45),
        (39, 42),
        (31, 35),
        (27, 33),
        (48, 54),
        (51, 57),
        (0, 16),
        (8, 27),
        (17, 26),
        (21, 22),
        (3, 13),
        (30, 8),
    ]
    features = [
        float(np.linalg.norm(points[first] - points[second]))
        for first, second in index_pairs
    ]
    features.extend(
        [
            float(points[30, 2] - points[27, 2]),
            float(points[8, 2] - points[30, 2]),
            float(points[0, 2] - points[30, 2]),
        ]
    )
    vector = np.asarray(features, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32) if norm > 0 else vector


def attach_optional_geometry(
    app,
    bgr: np.ndarray,
    item: dict,
    *,
    enabled: bool,
    descriptor_fn: Callable[[np.ndarray], np.ndarray | None] = geometry_descriptor,
    face_record_cls=FaceRecord,
) -> None:
    if not enabled:
        return
    try:
        face_obj = face_record_cls(
            bbox=np.asarray(item["bbox"], dtype=np.float32),
            kps=item.get("_kps_array"),
            det_score=float(item["det_score"]),
        )
        for model in app.models.values():
            if getattr(model, "taskname", "") != "landmark_3d_68":
                continue
            model.get(bgr, face_obj)
            landmarks = getattr(face_obj, "landmark_3d_68", None)
            if landmarks is not None:
                descriptor = descriptor_fn(
                    np.asarray(landmarks, dtype=np.float32)
                )
                if descriptor is not None:
                    item["geom3d"] = descriptor
            return
    except Exception as exc:
        item["geom3d_error"] = f"{type(exc).__name__}: {exc}"


__all__ = ["FaceRecord", "attach_optional_geometry", "geometry_descriptor"]
