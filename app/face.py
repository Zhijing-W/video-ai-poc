"""Face detection and identity orchestration compatibility façade.

The cohesive detector/runtime, recognition, geometry, and association
implementations live in ``app.identity.face``. This module retains the product
orchestration and legacy public/private monkeypatch seams.
"""
from __future__ import annotations

import numpy as np

from .config import settings
from .identity.face import association as _association
from .identity.face import geometry as _geometry
from .identity.face import recognition as _recognition
from .identity.face import runtime as _runtime
from .identity.face.adaface import embed as _adaface_embed
from .identity.face.adaface import load_error as _adaface_error
from .identity.face.quality import assess_quality as _assess_quality_impl
from .identity.face.quality import deep_fiqa_score as _deep_fiqa_score
from .identity.face.quality import superres_quality_ok as _superres_quality_ok
from .identity.face.super_resolution import _ensure_superres as _ensure_superres_impl
from .identity.face.super_resolution import available_backends as _available_superres_backends
from .identity.face.super_resolution import enhance as _enhance_impl
from .identity.face.super_resolution import register_backend as _register_superres_backend
from .identity.face.super_resolution import superres_error as _superres_error_impl
from .identity.face.super_resolution import validate_backend as _validate_superres_backend

_lock = _runtime._lock
_state = _runtime._state
_FaceRecord = _geometry.FaceRecord

FACE_DIM = 512


# Runtime compatibility seams -------------------------------------------------
def _resolve_cuda(device: str, ort: bool = False) -> bool:
    return _runtime.resolve_cuda(device, ort)


def _load_insightface():
    return _runtime.load_insightface(resolve_cuda_fn=_resolve_cuda)


def _ensure_backend() -> None:
    _runtime.ensure_backend(
        _state,
        load_insightface_fn=_load_insightface,
        lock=_lock,
    )


def active_backend() -> str:
    _ensure_backend()
    return _state["backend"]


def _to_bgr(image) -> np.ndarray:
    return _runtime.to_bgr(image)


def _align_face(
    bgr: np.ndarray,
    kps: np.ndarray | None,
) -> np.ndarray | None:
    return _runtime.align_face(bgr, kps)


def align_face(image, kps) -> np.ndarray | None:
    """Use the product five-point alignment and return a 112x112 BGR face."""
    points = (
        np.asarray(kps, dtype=np.float32)
        if kps is not None
        else None
    )
    return _align_face(_to_bgr(image), points)


def _detect_face_candidates(app, bgr: np.ndarray) -> list[dict]:
    return _runtime.detect_face_candidates(app, bgr)


# Recognition compatibility seams -------------------------------------------
def adaface_error() -> str | None:
    return _adaface_error()


def _reembed(bgr_face: np.ndarray) -> np.ndarray | None:
    return _recognition.reembed(bgr_face, _state)


def embed_aligned_face(
    bgr_face: np.ndarray,
    backend: str,
) -> np.ndarray | None:
    """Extract an embedding from an already aligned face."""
    _ensure_backend()
    return _recognition.embed_aligned_face(
        bgr_face,
        backend,
        arcface_embed=_reembed,
        adaface_embed=_adaface_embed,
    )


# Geometry compatibility seams ----------------------------------------------
def geometry_descriptor(
    landmarks_3d: np.ndarray,
) -> np.ndarray | None:
    return _geometry.geometry_descriptor(landmarks_3d)


def _attach_optional_geometry(
    app,
    bgr: np.ndarray,
    item: dict,
) -> None:
    _geometry.attach_optional_geometry(
        app,
        bgr,
        item,
        enabled=settings.face_3d_cue,
        descriptor_fn=geometry_descriptor,
        face_record_cls=_FaceRecord,
    )


# Detection and identity orchestration --------------------------------------
def detect(
    image,
    with_quality: bool = True,
    enhance_blurry: bool | None = None,
    with_identity: bool = True,
    with_geometry: bool = True,
    superres_backend: str | None = None,
) -> list[dict]:
    """Detect faces, assess quality, then optionally attach identity evidence."""
    if superres_backend is not None:
        superres_backend = _validate_superres_backend(superres_backend)
    _ensure_backend()
    bgr = _to_bgr(image)
    app = _state["model"]["app"]
    candidates = _detect_face_candidates(app, bgr)
    selected_superres = (
        settings.face_superres
        if superres_backend is None
        else superres_backend
    )
    use_sr = (
        selected_superres not in {"off", "none", ""}
        if enhance_blurry is None
        else enhance_blurry
    )
    if use_sr:
        selected_superres = _validate_superres_backend(selected_superres)
        use_sr = selected_superres != "off"

    output: list[dict] = []
    for candidate in candidates:
        item = dict(candidate)
        aligned_bgr = _align_face(bgr, item.get("_kps_array"))
        quality = assess_quality(item, bgr, aligned_bgr=aligned_bgr)
        if with_geometry:
            _attach_optional_geometry(app, bgr, item)
        _attach_identity(
            item,
            aligned_bgr,
            quality,
            use_sr=use_sr,
            superres_backend=selected_superres,
            with_identity=with_identity,
        )
        item.pop("_kps_array", None)
        if with_quality:
            item["quality"] = quality
        output.append(item)
    return output


def _attach_identity(
    item: dict,
    aligned_bgr: np.ndarray | None,
    quality: dict,
    *,
    use_sr: bool,
    superres_backend: str,
    with_identity: bool,
) -> None:
    """Finalize one frozen face candidate without rerunning detection."""
    _recognition.attach_identity(
        item,
        aligned_bgr,
        quality,
        use_sr=use_sr,
        superres_backend=superres_backend,
        with_identity=with_identity,
        rec_backend=settings.face_rec_backend,
        enhance_fn=enhance,
        superres_error_fn=superres_error,
        fiqa_fn=_deep_fiqa_score,
        superres_quality_fn=_superres_quality_ok,
        embed_fn=embed_aligned_face,
    )


def finalize_identity(
    image,
    face: dict,
    enhance_blurry: bool | None = None,
    superres_backend: str | None = None,
) -> dict:
    """Embed a previously detected face without rerunning SCRFD."""
    if superres_backend is not None:
        superres_backend = _validate_superres_backend(superres_backend)
    _ensure_backend()
    bgr = _to_bgr(image)
    item = dict(face)
    keypoints = item.get("kps")
    aligned_bgr = _align_face(
        bgr,
        (
            np.asarray(keypoints, dtype=np.float32)
            if keypoints is not None
            else None
        ),
    )
    quality = dict(
        item.get("quality")
        or assess_quality(item, bgr, aligned_bgr=aligned_bgr)
    )
    selected_superres = (
        settings.face_superres
        if superres_backend is None
        else superres_backend
    )
    use_sr = (
        selected_superres not in {"off", "none", ""}
        if enhance_blurry is None
        else enhance_blurry
    )
    if use_sr:
        selected_superres = _validate_superres_backend(selected_superres)
        use_sr = selected_superres != "off"
    _attach_identity(
        item,
        aligned_bgr,
        quality,
        use_sr=use_sr,
        superres_backend=selected_superres,
        with_identity=True,
    )
    item["quality"] = quality
    item.pop("_kps_array", None)
    return item


# Recognition selection/fusion façade ---------------------------------------
def best_face(faces: list[dict]) -> dict | None:
    return _recognition.best_face(faces)


def fuse_embeddings(faces: list[dict]) -> np.ndarray | None:
    return _recognition.fuse_embeddings(faces)


# Association façade ---------------------------------------------------------
def _iou_contain(face_box, person_box) -> float:
    return _association.containment(face_box, person_box)


def associate_to_persons(
    faces: list[dict],
    person_dets: list[dict],
) -> dict[int, dict]:
    return _association.associate_to_persons(
        faces,
        person_dets,
        min_contain=settings.face_assoc_min_contain,
        ambiguity_margin=settings.face_assoc_ambiguity_margin,
        max_head_y_ratio=settings.face_assoc_max_head_y_ratio,
        containment_fn=_iou_contain,
    )


# Super-resolution and quality compatibility façade -------------------------
def available_superres_backends() -> tuple[str, ...]:
    return _available_superres_backends()


def validate_superres_backend(backend: str | None = None) -> str:
    return _validate_superres_backend(backend)


def register_superres_backend(
    name: str,
    loader,
    enhancer,
    *,
    replace: bool = False,
) -> None:
    _register_superres_backend(
        name,
        loader,
        enhancer,
        replace=replace,
    )


def superres_error(backend: str | None = None) -> str | None:
    if backend is None:
        return _superres_error_impl()
    return _superres_error_impl(backend)


def _ensure_superres(backend: str | None = None):
    """Retain the legacy experiment integration seam."""
    if backend is None:
        return _ensure_superres_impl()
    return _ensure_superres_impl(backend)


def enhance(
    image,
    *,
    aligned: bool = False,
    backend: str | None = None,
):
    return _enhance_impl(
        image,
        aligned=aligned,
        backend=backend,
    )


def assess_quality(
    face: dict,
    bgr: np.ndarray | None = None,
    aligned_bgr: np.ndarray | None = None,
) -> dict:
    return _assess_quality_impl(
        face,
        bgr=bgr,
        aligned_bgr=aligned_bgr,
    )


__all__ = [
    "FACE_DIM",
    "active_backend",
    "align_face",
    "adaface_error",
    "assess_quality",
    "associate_to_persons",
    "available_superres_backends",
    "best_face",
    "detect",
    "finalize_identity",
    "embed_aligned_face",
    "enhance",
    "fuse_embeddings",
    "geometry_descriptor",
    "register_superres_backend",
    "superres_error",
    "validate_superres_backend",
]
