"""Fixed actor-check-in Gallery experiment for MEVID Query tracklets.

prepare freezes schema-v3 inputs without recognition or super-resolution.
evaluate computes A/B once, derives C from the cache, and emits auditable reports.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
SUPERRES_DIR = SCRIPT_DIR.parent
EXPERIMENT_DIR = SUPERRES_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for import_root in (REPO_ROOT, EXPERIMENT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import mevid_eval_common as common  # noqa: E402
from app.identity.evidence_selection import (  # noqa: E402
    body_quality_score,
    ensure_body_fallback,
    face_candidate_proxy,
    face_evidence_rank,
    public_evidence,
    update_face_candidates,
)
from app.identity.face.quality import (  # noqa: E402
    deep_fiqa_score,
    face_gallery_quality_ok,
    no_face_quality,
    superres_quality_ok,
)

SCHEMA_VERSION = 3
ARMS = ("A_original", "B_all_superres", "C_gated_superres")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHECKIN_RE = re.compile(r"^(?P<pid>\d+)-.+-(?P<view>[fb])$", re.IGNORECASE)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def manifest_identity(
    coverage: dict,
    config_hash: str,
    model_provenance_hash: str,
    gallery: list[dict],
    queries: list[dict],
) -> dict:
    """Complete immutable protocol input used to detect stale/tampered manifests."""
    return {
        "coverage": coverage,
        "config_hash": config_hash,
        "model_provenance_hash": model_provenance_hash,
        "gallery": gallery,
        "queries": queries,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checkin_image(path: Path) -> tuple[str, str] | None:
    match = CHECKIN_RE.match(path.stem)
    if not match:
        return None
    return f"{int(match.group('pid')):04d}", match.group("view").upper()


def load_checkin_front_images(root: Path) -> dict[str, list[Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"actor check-in目录不存在：{root}")
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(
        item for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    ):
        parsed = parse_checkin_image(path)
        if parsed and parsed[1] == "F":
            grouped[parsed[0]].append(path)
    if not grouped:
        raise RuntimeError(f"actor check-in目录没有可解析的-F正脸照片：{root}")
    return dict(grouped)


def audit_prefix_coverage(
    checkin_pids: set[str],
    train_pids: set[str],
    test_pids: set[str],
    query_pids: set[str],
) -> dict:
    coverage = {
        "checkin_prefixes": sorted(checkin_pids),
        "train_pids": sorted(train_pids),
        "test_pids": sorted(test_pids),
        "query_pids": sorted(query_pids),
        "missing_train_pids": sorted(train_pids - checkin_pids),
        "missing_test_pids": sorted(test_pids - checkin_pids),
        "missing_query_pids": sorted(query_pids - checkin_pids),
        "extra_checkin_prefixes": sorted(checkin_pids - train_pids - test_pids),
    }
    coverage["counts"] = {
        key: len(value)
        for key, value in coverage.items()
        if isinstance(value, list)
    }
    if coverage["missing_query_pids"]:
        raise RuntimeError(
            "actor check-in前缀缺少官方Query PID映射："
            + ",".join(coverage["missing_query_pids"])
        )
    return coverage


def annotation_pid_set(data_root: Path, filename: str) -> set[str]:
    path = (
        data_root
        / "annotation"
        / "mevid-v1-annotation-data"
        / filename
    )
    if not path.is_file():
        raise FileNotFoundError(f"MEVID annotation不存在：{path}")
    return {
        f"{int(float(parts[2])):04d}"
        for line in path.read_text(encoding="utf-8").splitlines()
        if len(parts := line.split()) >= 3
    }


def sample_evenly_indexed(items: list[Path], count: int) -> list[tuple[int, Path]]:
    if count <= 0 or len(items) <= count:
        return list(enumerate(items))
    if count == 1:
        indices = [0]
    else:
        indices = [
            int(round(index * (len(items) - 1) / (count - 1)))
            for index in range(count)
        ]
    return [(index, items[index]) for index in dict.fromkeys(indices)]


def select_arm_embeddings(
    original: np.ndarray | None,
    superres: np.ndarray | None,
    *,
    eligibility: str,
    superres_succeeded: bool,
    post_superres_accepted: bool,
) -> dict[str, np.ndarray | None]:
    """Derive A/B/C without fallback and without invoking GFPGAN."""
    b_vector = superres if superres_succeeded and superres is not None else None
    if eligibility == "direct":
        c_vector = original
    elif eligibility == "recoverable":
        c_vector = b_vector if post_superres_accepted else None
    else:
        c_vector = None
    return {
        "A_original": original,
        "B_all_superres": b_vector,
        "C_gated_superres": c_vector,
    }


def build_image_manifest_records(
    queries: list[dict],
    artifacts_by_sample: dict[str, dict],
) -> list[dict]:
    rows = []
    for query in queries:
        sample_id = query["sample_id"]
        artifact = artifacts_by_sample.get(sample_id) or {}
        aligned = bool(query.get("aligned_path"))
        rows.append(
            {
                "sample_id": sample_id,
                "pid": query["pid"],
                "track": query["track"],
                "eligibility": query.get("eligibility", "none"),
                "source_frame_index": query.get("face_best_frame_index"),
                "status": artifact.get("status")
                or ("missing_artifact" if aligned else "non_processed"),
                "reason": artifact.get("reason")
                or (None if aligned else query.get("face_status", "not_detected")),
                "comparison_path": artifact.get("comparison_path"),
                "original_aligned_path": artifact.get("original_aligned_path"),
                "superres_aligned_path": artifact.get("superres_aligned_path"),
            }
        )
    return rows


def _relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve(value: str | None, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _save_bgr(path: Path, bgr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(bgr)[:, :, ::-1]).save(path)
    return file_sha256(path)


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


def _verify_manifest(payload: dict, manifest_path: Path) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("evaluate仅接受schema-v3 manifest")
    if payload.get("kind") != "checkin_superres_abc_manifest":
        raise ValueError("不是checkin_superres_abc_manifest")
    if stable_hash(payload["prepare_config"]) != payload.get("prepare_config_hash"):
        raise RuntimeError("prepare配置快照hash不匹配")
    if stable_hash(payload["model_provenance"]) != payload.get(
        "model_provenance_hash"
    ):
        raise RuntimeError("模型provenance hash不匹配")
    fixed_identity = manifest_identity(
        payload["coverage"],
        payload["prepare_config_hash"],
        payload["model_provenance_hash"],
        payload["gallery"],
        payload["queries"],
    )
    if stable_hash(fixed_identity) != payload.get("manifest_id"):
        raise RuntimeError("manifest固定输入identity hash不匹配")
    for row in [*payload["gallery"], *payload["queries"]]:
        for path_key, hash_key in (
            ("source_path", "source_sha256"),
            ("aligned_path", "aligned_sha256"),
        ):
            path = _resolve(row.get(path_key), manifest_path.parent)
            expected = row.get(hash_key)
            if path is None:
                continue
            if not path.is_file():
                raise FileNotFoundError(f"固定输入不存在：{path}")
            if expected and file_sha256(path) != expected:
                raise RuntimeError(f"固定输入hash不匹配：{path}")


def _normalise(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 0 else None


def _pack_vectors(vectors: list[np.ndarray | None], dim: int = 512) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(vectors), dim), dtype=np.float32)
    valid = np.zeros(len(vectors), dtype=np.bool_)
    for index, vector in enumerate(vectors):
        value = _normalise(vector)
        if value is not None:
            if value.size != dim:
                raise ValueError(f"embedding维度错误：{value.size} != {dim}")
            matrix[index] = value
            valid[index] = True
    return matrix, valid


def _unpack_vectors(matrix: np.ndarray, valid: np.ndarray) -> list[np.ndarray | None]:
    return [
        np.asarray(matrix[index], dtype=np.float32) if bool(valid[index]) else None
        for index in range(len(valid))
    ]


def _compute_embedding_cache(
    payload: dict,
    manifest_path: Path,
    artifact_dir: Path,
    face_mod,
    settings,
) -> tuple[list[dict], list[dict], dict]:
    original_dir = artifact_dir / "aligned_original"
    superres_dir = artifact_dir / "aligned_superres"
    original_dir.mkdir(parents=True, exist_ok=True)
    superres_dir.mkdir(parents=True, exist_ok=True)

    gallery_records = []
    gallery_vectors = []
    for row in payload["gallery"]:
        path = _resolve(row["aligned_path"], manifest_path.parent)
        assert path is not None
        vector = _normalise(
            face_mod.embed_aligned_face(
                np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy(),
                "arcface",
            )
        )
        gallery_records.append(dict(row))
        gallery_vectors.append(vector)

    settings.face_superres = "gfpgan"
    face_mod._ensure_superres()
    startup_error = face_mod.superres_error()
    frozen_poor_threshold = float(
        payload["prepare_config"]["face_fiqa_poor_thresh"]
    )
    query_records = []
    query_a = []
    query_b = []
    query_c = []
    gfpgan_seconds = 0.0
    superres_requests = 0
    gfpgan_invocations = 0
    gfpgan_outputs = 0
    enhanced_embedding_success = 0
    startup_blocked = 0
    gfpgan_no_output = 0
    enhanced_embedding_failure = 0
    for index, frozen in enumerate(payload["queries"], start=1):
        record = dict(frozen)
        original = enhanced = None
        sr_success = False
        sr_reason = None
        fiqa_after = None
        post_accepted = False
        restoration_output = False
        aligned_path = _resolve(frozen.get("aligned_path"), manifest_path.parent)
        if aligned_path is not None:
            original_path = original_dir / f"{frozen['sample_id']}.png"
            shutil.copyfile(aligned_path, original_path)
            original_hash = file_sha256(original_path)
            original_bgr = np.asarray(
                Image.open(original_path).convert("RGB")
            )[:, :, ::-1].copy()
            original = _normalise(
                face_mod.embed_aligned_face(original_bgr, "arcface")
            )
            record["original_aligned_path"] = _relative(original_path, artifact_dir)
            record["original_aligned_sha256"] = original_hash
            superres_requests += 1
            if startup_error:
                startup_blocked += 1
                sr_reason = f"startup_error:{startup_error}"
            else:
                gfpgan_invocations += 1
                original_rgb = Image.fromarray(original_bgr[:, :, ::-1])
                started = time.perf_counter()
                restored = face_mod.enhance(original_rgb, aligned=True)
                elapsed = time.perf_counter() - started
                gfpgan_seconds += elapsed
                record["gfpgan_seconds"] = round(elapsed, 6)
                if restored is original_rgb:
                    gfpgan_no_output += 1
                    sr_reason = "gfpgan_no_output"
                else:
                    restoration_output = True
                    gfpgan_outputs += 1
                    enhanced_bgr = np.asarray(
                        restored.convert("RGB")
                    )[:, :, ::-1].copy()
                    sr_path = superres_dir / f"{frozen['sample_id']}.png"
                    sr_hash = _save_bgr(sr_path, enhanced_bgr)
                    enhanced = _normalise(
                        face_mod.embed_aligned_face(enhanced_bgr, "arcface")
                    )
                    if enhanced is None:
                        enhanced_embedding_failure += 1
                        sr_reason = "superres_embedding_failed"
                    else:
                        sr_success = True
                        enhanced_embedding_success += 1
                    fiqa_after = deep_fiqa_score(enhanced_bgr)
                    post_accepted, post_reason = superres_quality_ok(
                        fiqa_after,
                        poor_threshold=frozen_poor_threshold,
                    )
                    record["superres_aligned_path"] = _relative(
                        sr_path, artifact_dir
                    )
                    record["superres_aligned_sha256"] = sr_hash
                    record["post_superres_reason"] = post_reason
        else:
            sr_reason = frozen.get("face_status", "not_detected")

        arms = select_arm_embeddings(
            original,
            enhanced,
            eligibility=frozen.get("eligibility", "none"),
            superres_succeeded=sr_success,
            post_superres_accepted=post_accepted,
        )
        query_a.append(arms["A_original"])
        query_b.append(arms["B_all_superres"])
        query_c.append(arms["C_gated_superres"])
        record.update(
            {
                "superres_attempted": aligned_path is not None,
                "restoration_output": restoration_output,
                "superres_succeeded": sr_success,
                "superres_failure_reason": sr_reason,
                "fiqa_before": (frozen.get("quality") or {}).get("fiqa"),
                "fiqa_after": fiqa_after,
                "post_superres_accepted": post_accepted,
                "embedding_cosine_original_superres": (
                    round(float(original @ enhanced), 6)
                    if original is not None and enhanced is not None
                    else None
                ),
            }
        )
        query_records.append(record)
        if index % 20 == 0 or index == len(payload["queries"]):
            print(f"    evaluate A/B {index}/{len(payload['queries'])}", flush=True)

    gallery_matrix, gallery_valid = _pack_vectors(gallery_vectors)
    a_matrix, a_valid = _pack_vectors(query_a)
    b_matrix, b_valid = _pack_vectors(query_b)
    c_matrix, c_valid = _pack_vectors(query_c)
    npz_path = artifact_dir / "embeddings.npz"
    np.savez_compressed(
        npz_path,
        gallery_sample_ids=np.asarray(
            [row["sample_id"] for row in gallery_records], dtype="U128"
        ),
        gallery_vectors=gallery_matrix,
        gallery_valid=gallery_valid,
        query_sample_ids=np.asarray(
            [row["sample_id"] for row in query_records], dtype="U128"
        ),
        A_original=a_matrix,
        A_original_valid=a_valid,
        B_all_superres=b_matrix,
        B_all_superres_valid=b_valid,
        C_gated_superres=c_matrix,
        C_gated_superres_valid=c_valid,
    )
    cache = {
        "schema_version": 2,
        "kind": "checkin_superres_embedding_cache",
        "manifest_id": payload["manifest_id"],
        "prepare_config_hash": payload["prepare_config_hash"],
        "model_provenance_hash": payload["model_provenance_hash"],
        "evaluation_model_provenance": _model_provenance(settings),
        "npz_path": _relative(npz_path, artifact_dir),
        "npz_sha256": file_sha256(npz_path),
        "gallery_records": gallery_records,
        "query_records": query_records,
        "runtime": {
            "superres_requests": superres_requests,
            "gfpgan_calls": gfpgan_invocations,
            "gfpgan_outputs": gfpgan_outputs,
            "gfpgan_no_output": gfpgan_no_output,
            "startup_blocked": startup_blocked,
            "enhanced_embedding_success": enhanced_embedding_success,
            "enhanced_embedding_failure": enhanced_embedding_failure,
            "gfpgan_seconds": round(gfpgan_seconds, 6),
            "gfpgan_mean_seconds": (
                round(gfpgan_seconds / gfpgan_invocations, 6)
                if gfpgan_invocations
                else None
            ),
            "superres_startup_error": startup_error,
        },
    }
    cache["evaluation_model_provenance_hash"] = stable_hash(
        cache["evaluation_model_provenance"]
    )
    cache_path = artifact_dir / "embedding_cache.json"
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return gallery_records, query_records, cache


def _load_embedding_cache(
    payload: dict,
    artifact_dir: Path,
    settings,
) -> tuple[list[dict], list[dict], dict]:
    cache_path = artifact_dir / "embedding_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("schema_version") != 2:
        raise RuntimeError(
            "embedding cache schema已过期；请使用--force-recompute重建"
        )
    for key in ("manifest_id", "prepare_config_hash", "model_provenance_hash"):
        if cache.get(key) != payload.get(key):
            raise RuntimeError(f"embedding cache provenance不匹配：{key}")
    if stable_hash(cache["evaluation_model_provenance"]) != cache.get(
        "evaluation_model_provenance_hash"
    ):
        raise RuntimeError("evaluation model provenance hash不匹配")
    if _model_provenance(settings) != cache["evaluation_model_provenance"]:
        raise RuntimeError("运行时模型文件与embedding cache provenance不匹配")
    npz_path = _resolve(cache["npz_path"], artifact_dir)
    assert npz_path is not None
    if file_sha256(npz_path) != cache["npz_sha256"]:
        raise RuntimeError("embedding NPZ hash不匹配")
    arrays = np.load(npz_path, allow_pickle=False)
    expected_gallery = [row["sample_id"] for row in cache["gallery_records"]]
    expected_query = [row["sample_id"] for row in cache["query_records"]]
    manifest_gallery = [row["sample_id"] for row in payload["gallery"]]
    manifest_queries = [row["sample_id"] for row in payload["queries"]]
    if expected_gallery != manifest_gallery or expected_query != manifest_queries:
        raise RuntimeError("embedding cache样本顺序与manifest不匹配")
    for cached, frozen in zip(cache["gallery_records"], payload["gallery"]):
        for key in ("sample_id", "pid", "aligned_sha256", "source_sha256"):
            if cached.get(key) != frozen.get(key):
                raise RuntimeError(f"Gallery cache字段与manifest不匹配：{key}")
    for cached, frozen in zip(cache["query_records"], payload["queries"]):
        for key in (
            "sample_id",
            "pid",
            "track",
            "eligibility",
            "face_best_frame_index",
            "aligned_sha256",
            "source_sha256",
            "quality",
        ):
            if cached.get(key) != frozen.get(key):
                raise RuntimeError(f"Query cache字段与manifest不匹配：{key}")
    if arrays["gallery_sample_ids"].tolist() != expected_gallery:
        raise RuntimeError("Gallery embedding顺序不匹配")
    if arrays["query_sample_ids"].tolist() != expected_query:
        raise RuntimeError("Query embedding顺序不匹配")
    unpacked_gallery = _unpack_vectors(
        arrays["gallery_vectors"], arrays["gallery_valid"]
    )
    for index, row in enumerate(cache["gallery_records"]):
        row["vectors"] = {
            "A_original": unpacked_gallery[index]
        }
    unpacked = {
        arm: _unpack_vectors(arrays[arm], arrays[f"{arm}_valid"])
        for arm in ARMS
    }
    frozen_poor_threshold = float(
        payload["prepare_config"]["face_fiqa_poor_thresh"]
    )
    for index, row in enumerate(cache["query_records"]):
        original = unpacked["A_original"][index]
        superres = unpacked["B_all_superres"][index]
        if row.get("restoration_output"):
            expected_post_accepted, expected_post_reason = superres_quality_ok(
                row.get("fiqa_after"),
                poor_threshold=frozen_poor_threshold,
            )
        else:
            expected_post_accepted, expected_post_reason = False, None
        if bool(row.get("post_superres_accepted")) != expected_post_accepted:
            raise RuntimeError(
                "cache post-SR门控不等于manifest冻结FIQA阈值的派生结果"
            )
        if row.get("restoration_output") and row.get(
            "post_superres_reason"
        ) != expected_post_reason:
            raise RuntimeError("cache post-SR拒绝原因与冻结门控不匹配")
        derived = select_arm_embeddings(
            original,
            superres,
            eligibility=payload["queries"][index].get("eligibility", "none"),
            superres_succeeded=bool(row.get("superres_succeeded")),
            post_superres_accepted=expected_post_accepted,
        )
        persisted_c = unpacked["C_gated_superres"][index]
        derived_c = derived["C_gated_superres"]
        if (persisted_c is None) != (derived_c is None) or (
            persisted_c is not None
            and derived_c is not None
            and not np.array_equal(persisted_c, derived_c)
        ):
            raise RuntimeError("持久化C向量不等于由缓存A/B及固定门控派生的C")
        row["vectors"] = derived
        for path_key, hash_key in (
            ("original_aligned_path", "original_aligned_sha256"),
            ("superres_aligned_path", "superres_aligned_sha256"),
        ):
            path = _resolve(row.get(path_key), artifact_dir)
            if path is not None and file_sha256(path) != row.get(hash_key):
                raise RuntimeError(f"cache图像hash不匹配：{path}")
    return cache["gallery_records"], cache["query_records"], cache


def _templates(
    gallery_records: list[dict],
    gallery_vectors: list[np.ndarray | None],
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, vector in zip(gallery_records, gallery_vectors):
        if vector is not None:
            grouped[row["pid"]].append(vector)
    return {
        pid: common.l2norm(np.mean(np.stack(vectors), axis=0))
        for pid, vectors in grouped.items()
    }


def _score(vector: np.ndarray | None, templates: dict[str, np.ndarray], gt: str) -> dict:
    if vector is None or not templates:
        return {
            "pred": None,
            "score": None,
            "rank": None,
            "rank1_correct": False,
            "rank5_correct": False,
            "gt_template_available": gt in templates,
        }
    ranked = sorted(
        ((pid, float(template @ vector)) for pid, template in templates.items()),
        key=lambda item: (-item[1], item[0]),
    )
    rank = next(
        (index + 1 for index, (pid, _) in enumerate(ranked) if pid == gt),
        None,
    )
    return {
        "pred": ranked[0][0],
        "score": round(ranked[0][1], 6),
        "rank": rank,
        "rank1_correct": rank == 1,
        "rank5_correct": rank is not None and rank <= 5,
        "gt_template_available": gt in templates,
    }


def summarize_scores(rows: list[dict], threshold: float) -> dict:
    count = len(rows)
    correct_accept = sum(
        row["rank1_correct"]
        and row["score"] is not None
        and row["score"] >= threshold
        for row in rows
    )
    wrong_accept = sum(
        not row["rank1_correct"]
        and row["pred"] is not None
        and row["score"] is not None
        and row["score"] >= threshold
        for row in rows
    )
    return {
        "queries": count,
        "vector_available": sum(row["pred"] is not None for row in rows),
        "gt_template_available": sum(
            row["gt_template_available"] for row in rows
        ),
        "rank1": sum(row["rank1_correct"] for row in rows),
        "rank1_rate": (
            round(sum(row["rank1_correct"] for row in rows) / count, 6)
            if count
            else None
        ),
        "rank5": sum(row["rank5_correct"] for row in rows),
        "rank5_rate": (
            round(sum(row["rank5_correct"] for row in rows) / count, 6)
            if count
            else None
        ),
        "fixed_threshold": threshold,
        "correct_accept": correct_accept,
        "correct_accept_rate": round(correct_accept / count, 6) if count else None,
        "wrong_accept_among_genuine_queries": wrong_accept,
        "wrong_accept_rate_among_genuine_queries": (
            round(wrong_accept / count, 6) if count else None
        ),
        "note": "wrong accept among genuine Query samples; this is not true FMR",
    }


def _exact_paired_p(improved: int, degraded: int) -> float:
    discordant = improved + degraded
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(improved, degraded) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def paired_uncertainty(
    before: list[dict],
    after: list[dict],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    if [row["sample_id"] for row in before] != [
        row["sample_id"] for row in after
    ]:
        raise ValueError("paired rows顺序不一致")
    deltas = [
        int(candidate["rank1_correct"]) - int(base["rank1_correct"])
        for base, candidate in zip(before, after)
    ]
    improved = sum(delta == 1 for delta in deltas)
    degraded = sum(delta == -1 for delta in deltas)
    correct_to_correct = sum(
        base["rank1_correct"] and candidate["rank1_correct"]
        for base, candidate in zip(before, after)
    )
    wrong_to_wrong = sum(
        not base["rank1_correct"] and not candidate["rank1_correct"]
        for base, candidate in zip(before, after)
    )
    rng = random.Random(seed)
    boot = []
    if deltas:
        for _ in range(max(0, bootstrap_samples)):
            boot.append(
                sum(deltas[rng.randrange(len(deltas))] for _ in deltas)
                / len(deltas)
            )
    boot.sort()
    lower = boot[int(0.025 * (len(boot) - 1))] if boot else None
    upper = boot[int(0.975 * (len(boot) - 1))] if boot else None
    return {
        "samples": len(deltas),
        "rank1_improved": improved,
        "rank1_degraded": degraded,
        "correct_to_correct": correct_to_correct,
        "correct_to_wrong": degraded,
        "wrong_to_correct": improved,
        "wrong_to_wrong": wrong_to_wrong,
        "prediction_changed": sum(
            base["pred"] != candidate["pred"]
            for base, candidate in zip(before, after)
        ),
        "rank1_net_delta": round(sum(deltas) / len(deltas), 6) if deltas else None,
        "paired_bootstrap_95ci": (
            [round(lower, 6), round(upper, 6)]
            if lower is not None and upper is not None
            else None
        ),
        "bootstrap_samples": max(0, bootstrap_samples),
        "exact_paired_two_sided_p": round(_exact_paired_p(improved, degraded), 8),
    }


def _font(size: int):
    for path in (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _panel(image: Image.Image | None, title: str, detail: str, size: int = 240) -> Image.Image:
    panel = Image.new("RGB", (size, size + 76), "white")
    if image is None:
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, size - 1, size - 1), outline="#999999", width=2)
        draw.text((20, size // 2 - 10), "NO IMAGE", fill="#aa0000", font=_font(20))
    else:
        panel.paste(
            image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS),
            (0, 0),
        )
    draw = ImageDraw.Draw(panel)
    draw.text((4, size + 8), title[:34], fill="black", font=_font(16))
    draw.text((4, size + 38), detail[:44], fill="#333333", font=_font(13))
    return panel


def _comparison(
    output: Path,
    record: dict,
    scores: dict[str, dict],
    gallery_examples: dict[str, dict],
    artifact_dir: Path,
    manifest_path: Path,
) -> None:
    panels = []
    gt_gallery = gallery_examples.get(record["pid"])
    gt_image = (
        Image.open(_resolve(gt_gallery["aligned_path"], manifest_path.parent))
        if gt_gallery
        else None
    )
    panels.append(_panel(gt_image, f"GT check-in {record['pid']}", "original Gallery"))
    original_path = _resolve(record.get("original_aligned_path"), artifact_dir)
    panels.append(
        _panel(
            Image.open(original_path) if original_path else None,
            "Original Query",
            f"frame={record.get('face_best_frame_index')} {record.get('eligibility')}",
        )
    )
    sr_path = _resolve(record.get("superres_aligned_path"), artifact_dir)
    panels.append(
        _panel(
            Image.open(sr_path) if sr_path else None,
            "GFPGAN Query",
            (
                f"success={record.get('superres_succeeded')} "
                f"accept={record.get('post_superres_accepted')}"
            ),
        )
    )
    added_pred = set()
    for arm in ("A_original", "B_all_superres"):
        score = scores[arm]
        pred = score["pred"]
        if pred and pred != record["pid"] and pred not in added_pred:
            predicted = gallery_examples.get(pred)
            predicted_image = (
                Image.open(_resolve(predicted["aligned_path"], manifest_path.parent))
                if predicted
                else None
            )
            panels.append(
                _panel(
                    predicted_image,
                    f"{arm[0]} predicted {pred}",
                    f"score={score['score']}",
                )
            )
            added_pred.add(pred)
    width = 20 + len(panels) * 260
    canvas = Image.new("RGB", (width, 386), "#f3f3f3")
    draw = ImageDraw.Draw(canvas)
    transition = (
        f"A={'correct' if scores['A_original']['rank1_correct'] else 'wrong'} -> "
        f"B={'correct' if scores['B_all_superres']['rank1_correct'] else 'wrong'}"
    )
    draw.text(
        (16, 10),
        (
            f"{record['sample_id']} | eligibility={record.get('eligibility')} | "
            f"{transition}"
        ),
        fill="black",
        font=_font(16),
    )
    draw.text(
        (16, 38),
        (
            f"A pred={scores['A_original']['pred']} score={scores['A_original']['score']} | "
            f"B pred={scores['B_all_superres']['pred']} score={scores['B_all_superres']['score']}"
        ),
        fill="#333333",
        font=_font(14),
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, (16 + index * 260, 68))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, subsampling=0)


def evaluate(args: argparse.Namespace) -> int:
    from app import face as face_mod
    from app.core.config import settings

    manifest_path = Path(args.manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_manifest(payload, manifest_path)
    settings.face_rec_backend = "arcface"
    runtime_provenance = _model_provenance(settings)
    if not _provenance_compatible(payload["model_provenance"], runtime_provenance):
        raise RuntimeError(
            "运行时模型/权重provenance与prepare不一致"
        )
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_path = artifact_dir / "embedding_cache.json"
    if cache_path.is_file() and not args.force_recompute:
        gallery_records, query_records, cache = _load_embedding_cache(
            payload, artifact_dir, settings
        )
        cache_reused = True
    else:
        gallery_records, query_records, cache = _compute_embedding_cache(
            payload,
            manifest_path,
            artifact_dir,
            face_mod,
            settings,
        )
        cache_reused = False
        gallery_records, query_records, cache = _load_embedding_cache(
            payload, artifact_dir, settings
        )

    gallery_vectors = [
        row["vectors"]["A_original"] for row in gallery_records
    ]
    templates = _templates(gallery_records, gallery_vectors)
    threshold = float(payload["prepare_config"]["face_hit_thresh"])
    scores_by_arm = {}
    for arm in ARMS:
        rows = []
        for record in query_records:
            score = _score(record["vectors"][arm], templates, record["pid"])
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "pid": record["pid"],
                    "track": record["track"],
                    "eligibility": record.get("eligibility", "none"),
                    "category": (record.get("quality") or {}).get(
                        "category", "none"
                    ),
                    **score,
                }
            )
        scores_by_arm[arm] = rows

    summaries = {}
    for arm, rows in scores_by_arm.items():
        summaries[arm] = {
            "full_cohort": summarize_scores(rows, threshold),
            "by_eligibility": {
                value: summarize_scores(
                    [row for row in rows if row["eligibility"] == value],
                    threshold,
                )
                for value in ("direct", "recoverable", "unusable", "none")
            },
            "by_category": {
                value: summarize_scores(
                    [row for row in rows if row["category"] == value],
                    threshold,
                )
                for value in ("clear", "marginal", "poor", "none")
            },
        }

    paired = {}
    for before, after in (
        ("A_original", "B_all_superres"),
        ("A_original", "C_gated_superres"),
        ("B_all_superres", "C_gated_superres"),
    ):
        paired[f"{before}_vs_{after}"] = paired_uncertainty(
            scores_by_arm[before],
            scores_by_arm[after],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    recoverable_indices = [
        index
        for index, row in enumerate(scores_by_arm["A_original"])
        if row["eligibility"] == "recoverable"
    ]
    paired["primary_recoverable_A_vs_B"] = paired_uncertainty(
        [scores_by_arm["A_original"][index] for index in recoverable_indices],
        [scores_by_arm["B_all_superres"][index] for index in recoverable_indices],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    gallery_examples = {}
    for row in gallery_records:
        gallery_examples.setdefault(row["pid"], row)
    comparison_dir = artifact_dir / "comparisons"
    artifacts_by_sample = {}
    for index, record in enumerate(query_records):
        score_map = {
            arm: scores_by_arm[arm][index]
            for arm in ARMS
        }
        artifact = {
            "status": (
                "processed"
                if record.get("superres_attempted")
                else "non_processed"
            ),
            "reason": record.get("superres_failure_reason"),
            "original_aligned_path": record.get("original_aligned_path"),
            "superres_aligned_path": record.get("superres_aligned_path"),
        }
        if record.get("superres_attempted"):
            comparison_path = comparison_dir / f"{record['sample_id']}.jpg"
            _comparison(
                comparison_path,
                record,
                score_map,
                gallery_examples,
                artifact_dir,
                manifest_path,
            )
            artifact["comparison_path"] = _relative(
                comparison_path, artifact_dir
            )
        artifacts_by_sample[record["sample_id"]] = artifact
    image_manifest_rows = build_image_manifest_records(
        payload["queries"], artifacts_by_sample
    )
    image_manifest = {
        "schema_version": 1,
        "kind": "checkin_superres_image_manifest",
        "manifest_id": payload["manifest_id"],
        "rows": image_manifest_rows,
    }
    image_manifest_path = artifact_dir / "image_manifest.json"
    image_manifest_path.write_text(
        json.dumps(image_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    drift = [
        row["embedding_cosine_original_superres"]
        for row in query_records
        if row.get("embedding_cosine_original_superres") is not None
    ]
    fiqa_pairs = [
        (float(row["fiqa_before"]), float(row["fiqa_after"]))
        for row in query_records
        if row.get("fiqa_before") is not None
        and row.get("fiqa_after") is not None
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "checkin_superres_abc_result",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "manifest_id": payload["manifest_id"],
        "protocol": {
            "gallery": "fixed original actor check-in front images only",
            "A_original": "original aligned Query embedding for every aligned face; diagnostic control",
            "B_all_superres": "GFPGAN every aligned Query; failures have no vector and never fall back to A",
            "C_gated_superres": (
                "cache-derived: direct=A; recoverable=accepted successful B; "
                "unusable/none=no vector"
            ),
            "broad_query_universe": "every official MEVID Query tracklet",
            "primary_causal_subgroup": "recoverable A-vs-B",
            "recognition_outcome_used_for_selection": False,
            "threshold_source": "frozen FACE_HIT_THRESH",
            "threshold": threshold,
            "wrong_accept_is_not_labeled_fmr": True,
        },
        "gallery": {
            "selected_images": len(gallery_records),
            "template_pids": len(templates),
            "template_pid_list": sorted(templates),
        },
        "results": summaries,
        "paired": paired,
        "diagnostics": {
            "embedding_cosine_original_superres": {
                "count": len(drift),
                "mean": round(float(np.mean(drift)), 6) if drift else None,
                "median": round(float(np.median(drift)), 6) if drift else None,
            },
            "fiqa": {
                "paired_count": len(fiqa_pairs),
                "before_mean": (
                    round(float(np.mean([pair[0] for pair in fiqa_pairs])), 6)
                    if fiqa_pairs
                    else None
                ),
                "after_mean": (
                    round(float(np.mean([pair[1] for pair in fiqa_pairs])), 6)
                    if fiqa_pairs
                    else None
                ),
                "delta_mean": (
                    round(
                        float(np.mean([pair[1] - pair[0] for pair in fiqa_pairs])),
                        6,
                    )
                    if fiqa_pairs
                    else None
                ),
            },
        },
        "runtime": {
            **cache["runtime"],
            "embedding_cache_reused": cache_reused,
            "embedding_cache_schema": cache["schema_version"],
            "evaluation_model_provenance_hash": cache[
                "evaluation_model_provenance_hash"
            ],
            "embedding_npz": cache["npz_path"],
            "image_manifest": _relative(image_manifest_path, artifact_dir),
            "comparison_records": sum(
                row["comparison_path"] is not None for row in image_manifest_rows
            ),
            "non_processed_records": sum(
                row["status"] == "non_processed" for row in image_manifest_rows
            ),
        },
        "scores": scores_by_arm,
    }
    output = (
        Path(args.output).resolve()
        if args.output
        else SUPERRES_DIR
        / "results"
        / f"checkin_superres_abc_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[saved] {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MEVID actor check-in固定Gallery超分A/B/C实验"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser(
        "prepare", help="冻结schema-v3 check-in Gallery与全部官方Query"
    )
    prepare_parser.add_argument("--data", required=True, help="MEVID根目录")
    prepare_parser.add_argument(
        "--checkin",
        default="",
        help="actor check-in目录；默认<MEVID_ROOT>/actor_checkin",
    )
    prepare_parser.add_argument(
        "--manifest",
        default=str(SUPERRES_DIR / "manifests" / "checkin_superres_abc_v3.json"),
    )
    prepare_parser.add_argument(
        "--cache",
        default=str(SUPERRES_DIR / "results" / "checkin_superres_prepare"),
    )
    prepare_parser.add_argument(
        "--frames-per-track",
        type=int,
        default=24,
        help="正式默认：每条Query确定性均匀抽24帧",
    )
    prepare_parser.add_argument("--top-k", type=int, default=3)
    prepare_parser.add_argument("--min-gap-frames", type=int, default=2)
    prepare_parser.add_argument("--gallery-shots", type=int, default=3)
    prepare_parser.set_defaults(handler=prepare)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="在固定manifest上计算/复用A/B缓存并派生C"
    )
    evaluate_parser.add_argument("--manifest", required=True)
    evaluate_parser.add_argument("--output", default="")
    evaluate_parser.add_argument(
        "--artifact-dir",
        default=str(SUPERRES_DIR / "results" / "checkin_superres_artifacts"),
    )
    evaluate_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    evaluate_parser.add_argument("--seed", type=int, default=0)
    evaluate_parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="显式丢弃可复用embedding cache并重跑GFPGAN",
    )
    evaluate_parser.set_defaults(handler=evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("frames_per_track", "top_k", "min_gap_frames", "gallery_shots"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')}必须大于0")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
