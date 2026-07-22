from __future__ import annotations

import numpy as np
from PIL import Image

from .. import body_reid as reid_mod
from .. import face as face_mod
from ..core.config import settings
from . import embedding_gallery as gallery_mod
from .evidence_selection import face_evidence_rank, public_evidence
from .face.quality import face_gallery_quality_ok, no_face_quality


def _empty_face_record(reason: str, *, error: str | None = None) -> dict:
    quality = no_face_quality(reason)
    record = {
        "observed": False,
        "quality": "none",
        "eligibility": "none",
        "quality_score": None,
        "fiqa_score": None,
        "defects": [],
        "can_enroll": False,
        "can_match": False,
        "can_superres": False,
        "match_ready": False,
        "match_source": "none",
        "matched": False,
        "face_subject_id": None,
        "match_score": None,
        "quality_detail": quality,
    }
    if error:
        record["face_error"] = error
    return record


def _legacy_candidates(tid: int, track: dict, frames: list) -> list[dict]:
    if track.get("face_candidates"):
        return [dict(item) for item in track["face_candidates"]]
    frame_index = int(track.get("best_idx", 0))
    return [
        {
            "track_id": tid,
            "frame_index": frame_index,
            "timestamp": getattr(frames[frame_index], "timestamp", None),
            "person_bbox": list(track.get("best_box") or []),
            "proxy_score": -1.0,
            "fallback": "legacy_body_best",
        }
    ]


def _person_crop(image: Image.Image, box: list[float]) -> Image.Image | None:
    x1, y1, x2, y2 = [int(value) for value in box[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    return image.crop((x1, y1, x2, y2)) if x2 - x1 >= 4 and y2 - y1 >= 4 else None


def _track_consistency(
    selected: dict,
    track: dict,
    body_embedding: np.ndarray | None,
    image: Image.Image,
) -> tuple[bool, float | None, str]:
    body_best = track.get("body_best") or {}
    if int(selected["frame_index"]) == int(body_best.get("frame_index", track.get("best_idx", -1))):
        return True, 1.0, "same_frame"
    if body_embedding is None:
        return False, None, "body_embedding_unavailable"
    crop = _person_crop(image, selected["person_bbox"])
    if crop is None:
        return False, None, "person_crop_unavailable"
    try:
        candidate_embedding = np.asarray(reid_mod.embed(crop), dtype=np.float32).reshape(-1)
        reference = np.asarray(body_embedding, dtype=np.float32).reshape(-1)
        denom = float(np.linalg.norm(candidate_embedding) * np.linalg.norm(reference))
        score = float(np.dot(candidate_embedding, reference) / denom) if denom > 0 else -1.0
    except Exception as exc:
        return False, None, f"body_consistency_error:{type(exc).__name__}:{exc}"
    return (
        score >= settings.face_track_consistency_thresh,
        round(score, 4),
        "passed" if score >= settings.face_track_consistency_thresh else "failed",
    )


def attach_faces(
    frames: list,
    tracks: dict[int, dict],
    identities: dict[int, dict],
    session_id: str,
    body_embeddings: dict[int, np.ndarray] | None = None,
) -> None:
    """Select and finalize face evidence independently from body-best."""
    face_sess = f"{session_id}-face"
    gallery_mod.reset_gallery(face_sess)
    body_embeddings = body_embeddings or {}

    candidates_by_tid: dict[int, list[dict]] = {}
    for tid, track in tracks.items():
        if identities.get(tid, {}).get("skipped"):
            identities[tid]["face"] = _empty_face_record("track_skipped")
            continue
        candidates_by_tid[tid] = _legacy_candidates(tid, track, frames)

    evaluated: dict[int, list[dict]] = {tid: [] for tid in candidates_by_tid}
    frame_cache: dict[int, tuple[dict[int, dict], str | None]] = {}
    max_rounds = max((len(items) for items in candidates_by_tid.values()), default=0)

    for rank in range(max_rounds):
        by_frame: dict[int, list[int]] = {}
        for tid, candidates in candidates_by_tid.items():
            if rank >= len(candidates):
                continue
            by_frame.setdefault(int(candidates[rank]["frame_index"]), []).append(tid)

        for frame_index, target_tids in sorted(by_frame.items()):
            if frame_index not in frame_cache:
                try:
                    image = Image.open(frames[frame_index].local_path).convert("RGB")
                    faces = face_mod.detect(
                        image,
                        with_quality=True,
                        enhance_blurry=False,
                        with_identity=False,
                        with_geometry=False,
                    )
                    person_dets = [
                        {
                            "box": track["boxes"][frame_index],
                            "track_id": tid,
                            "label": "person",
                        }
                        for tid, track in tracks.items()
                        if frame_index in track.get("boxes", {})
                    ]
                    frame_cache[frame_index] = (
                        face_mod.associate_to_persons(faces, person_dets),
                        None,
                    )
                except Exception as exc:
                    frame_cache[frame_index] = (
                        {},
                        f"{type(exc).__name__}: {exc}",
                    )
            associations, error = frame_cache[frame_index]
            for tid in target_tids:
                source = candidates_by_tid[tid][rank]
                face = associations.get(tid)
                if face is None:
                    if error:
                        identities[tid].setdefault("face_candidate_errors", []).append(
                            {"frame_index": frame_index, "error": error}
                        )
                    continue
                evidence = {
                    **source,
                    "face_bbox": list(face.get("bbox") or []),
                    "association_score": face.get("association_score"),
                    "quality": dict(face.get("quality") or {}),
                    "_face": face,
                }
                evaluated[tid].append(evidence)

    for tid, track in tracks.items():
        if identities.get(tid, {}).get("skipped"):
            continue
        options = evaluated.get(tid) or []
        if not options:
            error_rows = identities[tid].pop("face_candidate_errors", [])
            error = error_rows[-1]["error"] if error_rows else None
            identities[tid]["face"] = _empty_face_record(
                "detector_error" if error else "not_detected",
                error=error,
            )
            track["face_best"] = None
            continue

        selected = max(options, key=face_evidence_rank)
        frame_index = int(selected["frame_index"])
        image = Image.open(frames[frame_index].local_path).convert("RGB")
        consistent, consistency_score, consistency_status = _track_consistency(
            selected,
            track,
            body_embeddings.get(tid),
            image,
        )
        selected["track_consistency_score"] = consistency_score
        selected["track_consistency_status"] = consistency_status
        track["face_best"] = selected

        frozen_face = selected["_face"]
        quality = dict(frozen_face.get("quality") or {})
        if not consistent:
            rec = {
                **_empty_face_record("track_consistency_failed"),
                "observed": True,
                "quality": quality.get("category", "poor"),
                "eligibility": quality.get("eligibility", "unusable"),
                "quality_score": quality.get("quality"),
                "fiqa_score": quality.get("fiqa"),
                "defects": quality.get("defects") or [],
                "evidence": public_evidence(selected),
                "track_consistency_status": consistency_status,
                "track_consistency_score": consistency_score,
                "quality_detail": quality,
            }
            identities[tid]["face"] = rec
            continue

        finalized = face_mod.finalize_identity(image, frozen_face)
        quality = dict(finalized.get("quality") or quality)
        embedding = finalized.get("embedding")
        match_ready = bool(finalized.get("match_ready") and embedding is not None)
        rec = {
            "observed": True,
            "score": quality.get("det_score"),
            "quality": quality.get("category", "poor"),
            "eligibility": quality.get("eligibility", "unusable"),
            "quality_score": quality.get("quality"),
            "fiqa_score": quality.get("fiqa"),
            "defects": quality.get("defects") or [],
            "can_enroll": bool(quality.get("can_enroll")),
            "can_match": bool(quality.get("can_match")),
            "can_superres": bool(quality.get("can_superres")),
            "match_ready": match_ready,
            "match_source": finalized.get("match_source", "none"),
            "matched": False,
            "face_subject_id": None,
            "match_score": None,
            "evidence": public_evidence(selected),
            "track_consistency_status": consistency_status,
            "track_consistency_score": consistency_score,
            "quality_detail": quality,
        }
        if finalized.get("identity_error"):
            rec["face_error"] = finalized["identity_error"]
        if finalized.get("superres_rejected"):
            rec["superres_rejected"] = finalized["superres_rejected"]

        if match_ready:
            try:
                face_vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
                can_enroll = bool(quality.get("can_enroll")) and finalized.get("match_source") == "original"
                result = gallery_mod.with_gallery_locked(
                    face_sess,
                    face_mod.FACE_DIM,
                    lambda gallery: gallery.identify_or_enroll(
                        face_vector,
                        quality,
                        auto_enroll=can_enroll,
                        hit_thresh=settings.face_hit_thresh,
                        new_thresh=settings.face_new_thresh,
                        quality_gate=face_gallery_quality_ok,
                        low_quality_hit_thresh=settings.face_hit_thresh,
                    ),
                )
                rec["face_subject_id"] = result.get("subject_id")
                rec["match_score"] = result.get("score")
                rec["matched"] = result.get("decision") == "hit"
                rec["enrolled"] = result.get("enrolled")
                rec["gallery_quality_ok"] = result.get("quality_ok")
                if result.get("subject_id") is not None:
                    rec["route_subject"] = {
                        "route": "face",
                        "local_subject_id": result["subject_id"],
                    }
            except Exception as exc:
                rec["face_error"] = str(exc)
        identities[tid]["face"] = rec


__all__ = ["attach_faces"]
