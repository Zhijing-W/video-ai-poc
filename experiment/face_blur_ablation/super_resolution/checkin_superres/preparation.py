"""Input selection and immutable schema-v3 manifest preparation."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from common import mevid_eval_common as common
from app.identity.evidence_selection import (
    body_quality_score,
    ensure_body_fallback,
    face_candidate_proxy,
    face_evidence_rank,
    public_evidence,
    update_face_candidates,
)
from app.identity.face.quality import face_gallery_quality_ok, no_face_quality

from .common import (
    SCHEMA_VERSION,
    annotation_pid_set,
    audit_prefix_coverage,
    file_sha256,
    load_checkin_front_images,
    manifest_identity,
    sample_evenly_indexed,
    stable_hash,
    _relative,
    _save_bgr,
)


def _model_provenance(settings) -> dict:
    def local_weight(value: str) -> dict:
        path = Path(value) if value else None
        return {
            "configured": value or None,
            "sha256": file_sha256(path) if path and path.is_file() else None,
        }

    def artifact_tree(root: Path) -> list[dict]:
        if not root.is_dir():
            return []
        return [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]

    configured_gfpgan = (settings.face_gfpgan_weights or "").strip()
    if configured_gfpgan.startswith(("http://", "https://")):
        filename = Path(urlparse(configured_gfpgan).path).name
        gfpgan_local = Path.home() / ".cache" / "gfpgan" / filename
    elif configured_gfpgan:
        gfpgan_local = Path(configured_gfpgan)
    else:
        gfpgan_local = Path.home() / ".cache" / "gfpgan" / "GFPGANv1.3.pth"
    insightface_root = Path.home() / ".insightface" / "models" / settings.face_model
    return {
        "face_backend": settings.face_backend,
        "face_model": settings.face_model,
        "face_rec_backend": "arcface",
        "insightface_artifacts": artifact_tree(insightface_root),
        "gfpgan": {
            "implementation": "GFPGANv1.3",
            "deterministic_noise": True,
            "weights": {
                "configured": configured_gfpgan or "GFPGANv1.3 default URL",
                "resolved_path": str(gfpgan_local.resolve()),
                "sha256": (
                    file_sha256(gfpgan_local) if gfpgan_local.is_file() else None
                ),
            },
        },
        "fiqa": {
            "backend": settings.face_fiqa_backend,
            "arch": settings.face_fiqa_arch,
            "weights": local_weight(settings.face_fiqa_weights),
        },
    }


def _provenance_compatible(expected: dict, actual: dict) -> bool:
    if (
        expected.get("face_backend") != actual.get("face_backend")
        or expected.get("face_model") != actual.get("face_model")
        or expected.get("face_rec_backend") != actual.get("face_rec_backend")
    ):
        return False
    expected_insightface = expected.get("insightface_artifacts") or []
    if expected_insightface and expected_insightface != actual.get(
        "insightface_artifacts"
    ):
        return False
    for component in ("gfpgan", "fiqa"):
        expected_component = expected.get(component) or {}
        actual_component = actual.get(component) or {}
        for key, value in expected_component.items():
            if key == "weights":
                expected_sha = (value or {}).get("sha256")
                if expected_sha and expected_sha != (
                    actual_component.get("weights") or {}
                ).get("sha256"):
                    return False
            elif value != actual_component.get(key):
                return False
    return True
def _product_config(settings, args: argparse.Namespace) -> dict:
    return {
        "protocol": "checkin_superres_abc_v3",
        "frames_per_track": int(args.frames_per_track),
        "face_candidate_top_k": int(args.top_k),
        "face_candidate_min_gap_frames": int(args.min_gap_frames),
        "gallery_shots": int(args.gallery_shots),
        "frame_sampling": "deterministic_even_including_endpoints",
        "face_best_ranking": "app.identity.evidence_selection.face_evidence_rank",
        "body_best_fallback": True,
        "face_det_size": settings.face_det_size,
        "face_min_det_score": settings.face_min_det_score,
        "face_min_size": settings.face_min_size,
        "face_recoverable_min_size": settings.face_recoverable_min_size,
        "face_superres_max_size": settings.face_superres_max_size,
        "face_ref_area": settings.face_ref_area,
        "face_min_blur_var": settings.face_min_blur_var,
        "face_blur_clear_var": settings.face_blur_clear_var,
        "face_yaw_clear": settings.face_yaw_clear,
        "face_yaw_max": settings.face_yaw_max,
        "face_pitch_clear": settings.face_pitch_clear,
        "face_pitch_down_max": settings.face_pitch_down_max,
        "face_pitch_up_max": settings.face_pitch_up_max,
        "face_fiqa_backend": settings.face_fiqa_backend,
        "face_fiqa_arch": settings.face_fiqa_arch,
        "face_fiqa_poor_thresh": settings.face_fiqa_poor_thresh,
        "face_fiqa_clear_thresh": settings.face_fiqa_clear_thresh,
        "face_hit_thresh": settings.face_hit_thresh,
    }


def _best_face_in_image(image: Image.Image, face_mod, frame_index: int = 0) -> dict | None:
    faces = face_mod.detect(
        image,
        with_quality=True,
        enhance_blurry=False,
        with_identity=False,
        with_geometry=False,
    )
    options = []
    for face in faces:
        aligned = face_mod.align_face(image, face.get("kps"))
        options.append(
            {
                "frame_index": frame_index,
                "face_bbox": list(face.get("bbox") or []),
                "quality": dict(face.get("quality") or {}),
                "_face": face,
                "_aligned": aligned,
            }
        )
    return max(options, key=face_evidence_rank) if options else None


def _gallery_candidate(
    path: Path,
    pid: str,
    index: int,
    aligned_dir: Path,
    manifest_base: Path,
    face_mod,
) -> tuple[dict | None, str]:
    image = Image.open(path).convert("RGB")
    selected = _best_face_in_image(image, face_mod)
    if selected is None:
        return None, "not_detected"
    quality = selected["quality"]
    gallery_ok, rejection_reason = face_gallery_quality_ok(quality)
    if not gallery_ok:
        return None, f"ineligible:{rejection_reason}"
    aligned = selected["_aligned"]
    if aligned is None:
        return None, "unaligned"
    sample_id = f"checkin_{pid}_{index:04d}_{stable_hash(path.name)[:10]}"
    aligned_path = aligned_dir / f"{sample_id}.png"
    aligned_hash = _save_bgr(aligned_path, aligned)
    row = {
        "sample_id": sample_id,
        "pid": pid,
        "source_path": str(path.resolve()),
        "source_sha256": file_sha256(path),
        "source_name": path.name,
        "view": "F",
        "bbox": selected["face_bbox"],
        "quality": quality,
        "eligibility": quality.get("eligibility"),
        "aligned_path": _relative(aligned_path, manifest_base),
        "aligned_sha256": aligned_hash,
        "_rank": face_evidence_rank(selected),
    }
    return row, "eligible"


def _query_candidates(
    tracklet: common.Tracklet,
    *,
    frames_per_track: int,
    top_k: int,
    min_gap_frames: int,
    body_mod,
) -> tuple[list[dict], dict, list[dict]]:
    sampled = sample_evenly_indexed(tracklet.frames, frames_per_track)
    if not sampled:
        raise RuntimeError(f"Query tracklet没有帧：{tracklet.track}")
    candidates: list[dict] = []
    body_best = None
    provenance = []
    for frame_index, path in sampled:
        image = Image.open(path).convert("RGB")
        bbox = [0, 0, image.width, image.height]
        body_quality = dict(body_mod.assess_quality(image) or {})
        body_score = body_quality_score(body_quality)
        proxy = face_candidate_proxy(
            image,
            person_bbox=bbox,
            image_size=image.size,
            detection_confidence=1.0,
        )
        candidate = {
            "track_id": tracklet.track,
            "frame_index": frame_index,
            "timestamp": frame_index,
            "person_bbox": bbox,
            "source_path": str(path.resolve()),
            "proxy_score": round(float(proxy), 6),
            "body_quality_score": round(float(body_score), 6),
        }
        provenance.append(
            {
                "frame_index": frame_index,
                "source_path": str(path.resolve()),
                "proxy_score": candidate["proxy_score"],
                "body_quality_score": candidate["body_quality_score"],
            }
        )
        if body_best is None or body_score > body_best["body_quality_score"]:
            body_best = dict(candidate)
        candidates = update_face_candidates(
            candidates,
            candidate,
            top_k=top_k,
            min_gap_frames=min_gap_frames,
        )
    assert body_best is not None
    bounded = ensure_body_fallback(candidates, body_best, top_k=top_k)
    return bounded, body_best, provenance


def _freeze_query(
    tracklet: common.Tracklet,
    query_index: int,
    aligned_dir: Path,
    manifest_base: Path,
    args: argparse.Namespace,
    body_mod,
    face_mod,
) -> dict:
    sample_id = (
        f"query_{tracklet.pid}_c{tracklet.cam}_o{tracklet.outfit}_"
        f"t{tracklet.track:04d}"
    )
    row = {
        "sample_id": sample_id,
        "pid": tracklet.pid,
        "cam": tracklet.cam,
        "outfit": tracklet.outfit,
        "track": tracklet.track,
        "official_query_index": query_index,
        "face_status": "none",
        "face_best_frame_index": None,
        "source_path": None,
        "source_sha256": None,
        "bbox": None,
        "quality": no_face_quality(),
        "eligibility": "none",
        "aligned_path": None,
        "aligned_sha256": None,
    }
    bounded, body_best, sampled = _query_candidates(
        tracklet,
        frames_per_track=args.frames_per_track,
        top_k=args.top_k,
        min_gap_frames=args.min_gap_frames,
        body_mod=body_mod,
    )
    row["candidate_provenance"] = {
        "sampled_count": len(sampled),
        "sampled": sampled,
        "bounded_count": len(bounded),
        "bounded": [public_evidence(item) for item in bounded],
        "body_best_frame_index": body_best["frame_index"],
        "body_best_source_path": body_best["source_path"],
        "body_best_fallback_included": any(
            item.get("fallback") == "body_best" for item in bounded
        ),
    }
    evaluated = []
    errors = []
    for candidate in bounded:
        try:
            path = Path(candidate["source_path"])
            image = Image.open(path).convert("RGB")
            selected = _best_face_in_image(
                image,
                face_mod,
                frame_index=int(candidate["frame_index"]),
            )
            if selected is None:
                continue
            selected.update(candidate)
            evaluated.append(selected)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "frame_index": candidate["frame_index"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    row["candidate_provenance"]["detected_candidate_count"] = len(evaluated)
    row["candidate_provenance"]["errors"] = errors
    if not evaluated:
        row["quality"] = no_face_quality("detector_error" if errors else "not_detected")
        row["face_status"] = "detector_error" if errors else "none"
        return row

    selected = max(evaluated, key=face_evidence_rank)
    quality = dict(selected.get("quality") or {})
    source = Path(selected["source_path"])
    row.update(
        {
            "face_status": "detected",
            "face_best_frame_index": int(selected["frame_index"]),
            "source_path": str(source.resolve()),
            "source_sha256": file_sha256(source),
            "bbox": list(selected.get("face_bbox") or []),
            "quality": quality,
            "eligibility": quality.get("eligibility", "unusable"),
            "selected_candidate": public_evidence(selected),
        }
    )
    aligned = selected.get("_aligned")
    if aligned is None:
        row["face_status"] = "unaligned"
        return row
    aligned_path = aligned_dir / f"{sample_id}.png"
    row["aligned_sha256"] = _save_bgr(aligned_path, aligned)
    row["aligned_path"] = _relative(aligned_path, manifest_base)
    return row


def prepare(args: argparse.Namespace) -> int:
    from app import body_reid as body_mod
    from app import face as face_mod
    from app.core.config import settings

    data_root = Path(args.data).resolve()
    checkin_root = (
        Path(args.checkin).resolve()
        if args.checkin
        else data_root / "actor_checkin"
    )
    manifest_path = Path(args.manifest).resolve()
    cache_root = Path(args.cache).resolve()
    gallery_dir = cache_root / "gallery_original"
    query_dir = cache_root / "query_original"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    settings.face_rec_backend = "arcface"
    settings.face_superres = "off"
    test_tracklets = common.load_mevid(data_root)
    queries = [tracklet for tracklet in test_tracklets if tracklet.is_query]
    checkin = load_checkin_front_images(checkin_root)
    coverage = audit_prefix_coverage(
        set(checkin),
        annotation_pid_set(data_root, "track_train_info.txt"),
        {tracklet.pid for tracklet in test_tracklets},
        {tracklet.pid for tracklet in queries},
    )

    gallery_candidates = []
    gallery_rejections = defaultdict(int)
    started = time.perf_counter()
    for pid, paths in sorted(checkin.items()):
        for index, path in enumerate(paths):
            try:
                row, status = _gallery_candidate(
                    path,
                    pid,
                    index,
                    gallery_dir,
                    manifest_path.parent,
                    face_mod,
                )
                gallery_rejections[status] += 1
                if row is not None:
                    gallery_candidates.append(row)
            except Exception as exc:  # noqa: BLE001
                gallery_rejections[f"error:{type(exc).__name__}"] += 1

    grouped_gallery: dict[str, list[dict]] = defaultdict(list)
    for row in gallery_candidates:
        grouped_gallery[row["pid"]].append(row)
    gallery = []
    for pid, rows in sorted(grouped_gallery.items()):
        for shot, row in enumerate(
            sorted(rows, key=lambda item: item["_rank"], reverse=True)[
                : args.gallery_shots
            ],
            start=1,
        ):
            public_row = {key: value for key, value in row.items() if key != "_rank"}
            public_row["selected_shot"] = shot
            gallery.append(public_row)

    frozen_queries = []
    for index, tracklet in enumerate(queries):
        frozen_queries.append(
            _freeze_query(
                tracklet,
                index,
                query_dir,
                manifest_path.parent,
                args,
                body_mod,
                face_mod,
            )
        )
        if (index + 1) % 20 == 0 or index + 1 == len(queries):
            print(f"    prepare queries {index + 1}/{len(queries)}", flush=True)

    config = _product_config(settings, args)
    provenance = _model_provenance(settings)
    gfpgan_sha = (
        (provenance.get("gfpgan") or {}).get("weights") or {}
    ).get("sha256")
    if not gfpgan_sha:
        raise RuntimeError(
            "正式prepare必须预先下载并固定GFPGAN权重；"
            "FACE_GFPGAN_WEIGHTS或~/.cache/gfpgan中未找到可hash文件"
        )
    config_hash = stable_hash(config)
    provenance_hash = stable_hash(provenance)
    fixed_identity = manifest_identity(
        coverage,
        config_hash,
        provenance_hash,
        gallery,
        frozen_queries,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "checkin_superres_abc_manifest",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_id": stable_hash(fixed_identity),
        "data_root_hint": data_root.name,
        "checkin_root_hint": checkin_root.name,
        "protocol": {
            "gallery_source": "actor_checkin_front_only",
            "official_person_reid_gallery_used_as_face_gallery": False,
            "query_universe": "every official MEVID Query tracklet",
            "selection_uses_recognition_outcome": False,
            "formal_defaults_are_cost_bounded": True,
            "test_set_threshold_tuning": False,
        },
        "coverage": coverage,
        "prepare_config": config,
        "prepare_config_hash": config_hash,
        "model_provenance": provenance,
        "model_provenance_hash": provenance_hash,
        "gallery": gallery,
        "queries": frozen_queries,
        "summary": {
            "front_checkin_images": sum(len(paths) for paths in checkin.values()),
            "eligible_gallery_candidates": len(gallery_candidates),
            "selected_gallery_images": len(gallery),
            "selected_gallery_pids": len({row["pid"] for row in gallery}),
            "gallery_processing": dict(sorted(gallery_rejections.items())),
            "query_tracklets": len(frozen_queries),
            "query_aligned": sum(bool(row["aligned_path"]) for row in frozen_queries),
            "query_by_eligibility": {
                value: sum(row["eligibility"] == value for row in frozen_queries)
                for value in ("direct", "recoverable", "unusable", "none")
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[saved] {manifest_path}")
    return 0
