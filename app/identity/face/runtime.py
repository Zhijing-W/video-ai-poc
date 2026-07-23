from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from ...config import settings

_lock = threading.Lock()
_state: dict = {"backend": None, "model": None}


def resolve_cuda(device: str, ort: bool = False) -> bool:
    d = (device or "auto").strip().lower()
    if d == "cpu":
        return False
    if ort:
        try:
            import onnxruntime as ort_runtime

            return "CUDAExecutionProvider" in ort_runtime.get_available_providers()
        except Exception:
            return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_insightface(
    *,
    resolve_cuda_fn: Callable[[str, bool], bool] = resolve_cuda,
):
    """Load the configured InsightFace detection/recognition runtime."""
    from insightface.app import FaceAnalysis

    modules = ["detection", "recognition"]
    if settings.face_3d_cue:
        modules.insert(1, "landmark_3d_68")
    use_cuda = resolve_cuda_fn(settings.face_device, True)
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_cuda
        else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(
        name=settings.face_model,
        allowed_modules=modules,
        providers=providers,
    )
    app.prepare(
        ctx_id=0 if use_cuda else -1,
        det_thresh=settings.face_min_det_score,
        det_size=(settings.face_det_size, settings.face_det_size),
    )
    return {"app": app}


def ensure_backend(
    state: dict | None = None,
    *,
    load_insightface_fn: Callable[[], dict] = load_insightface,
    lock=None,
) -> None:
    """Thread-safely initialize the single caller-provided runtime state."""
    active_state = _state if state is None else state
    active_lock = _lock if lock is None else lock
    if active_state["backend"] is not None:
        return
    with active_lock:
        if active_state["backend"] is not None:
            return
        want = (settings.face_backend or "insightface").strip().lower()
        if want != "insightface":
            raise ValueError(f"未知 FACE_BACKEND：{want}")
        active_state["model"] = load_insightface_fn()
        active_state["backend"] = "insightface"


def to_bgr(image) -> np.ndarray:
    """Convert supported project image inputs to an InsightFace BGR array."""
    from PIL import Image

    if isinstance(image, np.ndarray):
        return image[:, :, ::-1].copy() if image.ndim == 3 else image
    if isinstance(image, Image.Image):
        pil = image
    else:
        from ...detector import _decode_image

        pil = _decode_image(image)
    rgb = np.asarray(pil.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def align_face(bgr: np.ndarray, kps: np.ndarray | None) -> np.ndarray | None:
    if kps is None:
        return None
    try:
        from insightface.utils import face_align

        return face_align.norm_crop(bgr, np.asarray(kps), image_size=112)
    except Exception:
        return None


def detect_face_candidates(app, bgr: np.ndarray) -> list[dict]:
    """Run SCRFD alone so recognition remains behind the quality gate."""
    bboxes, kpss = app.det_model.detect(bgr, max_num=0, metric="default")
    candidates = []
    for index in range(bboxes.shape[0]):
        kps = (
            np.asarray(kpss[index], dtype=np.float32)
            if kpss is not None
            else None
        )
        candidates.append(
            {
                "bbox": [float(value) for value in bboxes[index, :4]],
                "kps": kps.tolist() if kps is not None else None,
                "det_score": float(bboxes[index, 4]),
                "_kps_array": kps,
            }
        )
    return candidates


__all__ = [
    "_state",
    "align_face",
    "detect_face_candidates",
    "ensure_backend",
    "load_insightface",
    "resolve_cuda",
    "to_bgr",
]
