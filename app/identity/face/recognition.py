from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image


def reembed(bgr_face: np.ndarray, state: dict) -> np.ndarray | None:
    """Extract an ArcFace embedding with the loaded InsightFace recognizer."""
    app = state["model"]["app"]
    recognizer = next(
        (
            model
            for model in app.models.values()
            if getattr(model, "taskname", "") == "recognition"
        ),
        None,
    )
    if recognizer is None:
        return None
    try:
        feature = recognizer.get_feat(bgr_face).reshape(-1).astype(np.float32)
    except Exception:
        return None
    norm = float(np.linalg.norm(feature))
    return feature / norm if norm > 0 else feature


def embed_aligned_face(
    bgr_face: np.ndarray,
    backend: str,
    *,
    arcface_embed: Callable[[np.ndarray], np.ndarray | None],
    adaface_embed: Callable[[np.ndarray], np.ndarray | None],
) -> np.ndarray | None:
    name = backend.strip().lower()
    if name == "arcface":
        return arcface_embed(bgr_face)
    if name == "adaface":
        return adaface_embed(bgr_face)
    raise ValueError(f"未知人脸识别后端：{backend}")


def attach_identity(
    item: dict,
    aligned_bgr: np.ndarray | None,
    quality: dict,
    *,
    use_sr: bool,
    superres_backend: str,
    with_identity: bool,
    rec_backend: str,
    enhance_fn: Callable,
    superres_error_fn: Callable[[str], str | None],
    fiqa_fn: Callable[[np.ndarray], float | None],
    superres_quality_fn: Callable[
        [float | None],
        tuple[bool, str | None],
    ],
    embed_fn: Callable[[np.ndarray, str], np.ndarray | None],
) -> None:
    """Attach direct or restored identity evidence to one frozen face."""
    eligibility = quality.get(
        "eligibility",
        "direct" if quality.get("can_match") else "unusable",
    )
    identity_input = aligned_bgr if eligibility == "direct" else None
    match_source = "original" if identity_input is not None else "none"
    quality["enhanced"] = False

    if (
        use_sr
        and eligibility == "recoverable"
        and aligned_bgr is not None
    ):
        item["superres_attempted"] = True
        item["superres_backend"] = superres_backend
        aligned_rgb = Image.fromarray(aligned_bgr[:, :, ::-1])
        enhanced = enhance_fn(
            aligned_rgb,
            aligned=True,
            backend=superres_backend,
        )
        backend_error = superres_error_fn(superres_backend)
        if backend_error:
            item["superres_error"] = backend_error
        if enhanced is not None and enhanced is not aligned_rgb:
            restored_bgr = (
                np.asarray(enhanced.convert("RGB"))[:, :, ::-1].copy()
            )
            fiqa_after = fiqa_fn(restored_bgr)
            if fiqa_after is not None and np.isfinite(float(fiqa_after)):
                fiqa_after = float(fiqa_after)
                quality["fiqa_after_superres"] = round(fiqa_after, 3)
                before = quality.get("fiqa")
                if before is not None:
                    quality["fiqa_delta_superres"] = round(
                        fiqa_after - float(before),
                        3,
                    )
            fiqa_passed, fiqa_reason = superres_quality_fn(fiqa_after)
            item["superres_fiqa_passed"] = bool(fiqa_passed)
            if fiqa_reason:
                item["superres_fiqa_diagnostic"] = fiqa_reason
            identity_input = restored_bgr
            match_source = "superres"
            item["enhanced"] = True
            quality["enhanced"] = True

    item["match_source"] = match_source
    item["match_ready"] = False
    if not with_identity or identity_input is None:
        return
    try:
        identity_embedding = embed_fn(identity_input, rec_backend)
        if identity_embedding is not None:
            item["embedding"] = identity_embedding
            item["rec_backend"] = rec_backend
            item["match_ready"] = True
    except Exception as exc:
        item["identity_error"] = f"{type(exc).__name__}: {exc}"


def best_face(faces: list[dict]) -> dict | None:
    candidates = [face for face in faces if face.get("embedding") is not None]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda face: (face.get("quality", {}) or {}).get("quality", 0.0),
    )


def fuse_embeddings(faces: list[dict]) -> np.ndarray | None:
    embeddings, weights = [], []
    for face in faces:
        embedding = face.get("embedding")
        if embedding is None:
            continue
        embeddings.append(np.asarray(embedding, dtype=np.float32))
        quality = face.get("quality", {}) or {}
        weights.append(
            max(
                0.0,
                min(
                    1.0,
                    float(
                        quality.get(
                            "match_weight",
                            quality.get("quality", 0.0),
                        )
                    ),
                ),
            )
        )
    if not embeddings:
        return None
    weight_array = np.asarray(weights, dtype=np.float32)
    if weight_array.sum() <= 0:
        weight_array = np.ones(len(embeddings), dtype=np.float32)
    fused = (
        np.stack(embeddings) * weight_array[:, None]
    ).sum(axis=0)
    norm = float(np.linalg.norm(fused))
    return (
        (fused / norm).astype(np.float32)
        if norm > 0
        else fused.astype(np.float32)
    )


__all__ = [
    "attach_identity",
    "best_face",
    "embed_aligned_face",
    "fuse_embeddings",
    "reembed",
]
