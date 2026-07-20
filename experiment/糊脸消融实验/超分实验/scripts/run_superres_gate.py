"""MEVID超分门控实验：Train校准FIQA阈值，冻结Test输入后比较A/B/C。"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

SUPERRES_DIR = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = SUPERRES_DIR.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from common import mevid_eval_common as common  # noqa: E402

VARIANTS = ("A_original", "B_all_superres", "C_gated_superres")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_saved_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sample_id(tracklet: common.Tracklet) -> str:
    return f"{tracklet.pid}_c{tracklet.cam}_o{tracklet.outfit}_t{tracklet.track}"


def _sample_evenly_indexed(items: list[Path], count: int) -> list[tuple[int, Path]]:
    if count <= 0 or len(items) <= count:
        return list(enumerate(items))
    step = len(items) / count
    indices = [int(i * step) for i in range(count)]
    return [(index, items[index]) for index in indices]


def product_person_quality_score(quality: dict) -> float:
    blur_var = float(quality.get("blur_var") or 0.0)
    area = float(quality.get("area") or 0.0)
    return blur_var * min(1.0, area / 20000.0)


def select_product_best_frame(
    tracklet: common.Tracklet,
    frames_per_track: int,
    image_loader: Callable[[Path], Image.Image],
    quality_fn: Callable[[Image.Image], dict],
) -> dict:
    candidates = _sample_evenly_indexed(tracklet.frames, frames_per_track)
    if not candidates:
        raise ValueError(f"tracklet没有可用帧：{tracklet.pid}/{tracklet.track}")

    best = None
    for frame_index, path in candidates:
        image = image_loader(path)
        person_quality = dict(quality_fn(image) or {})
        score = product_person_quality_score(person_quality)
        candidate = {
            "path": path,
            "best_idx": frame_index,
            "person_quality": person_quality,
            "score": score,
            "candidate_count": len(candidates),
        }
        if best is None or score > best["score"]:
            best = candidate
    assert best is not None
    return best


def _quality_tags(quality: dict) -> list[str]:
    tags = []
    defects = set(quality.get("defects") or [])
    if "small_face" in defects:
        tags.append("small-face")
    if "blur" in defects or "low_fiqa" in defects:
        tags.append("blur")
    if {"extreme_yaw", "extreme_pitch", "pose_yaw", "pose_pitch"} & defects:
        tags.append("pose")
    category = quality.get("category", "poor")
    if not tags:
        tags.append(category if category in {"clear", "marginal"} else "other-poor")
    return tags


def _cause_bin(category: str, tags: list[str]) -> str:
    if category in {"clear", "marginal", "none"}:
        return category
    for candidate in ("small-face", "blur", "pose"):
        if candidate in tags:
            return candidate
    return "other-poor"


def _best_detected_face(faces: list[dict]) -> dict | None:
    return max(faces, key=lambda item: float(item.get("det_score", 0.0))) if faces else None


def _threshold_between(lower: float, upper: float | None) -> float:
    if upper is None:
        return float(np.nextafter(np.float64(lower), np.float64(np.inf)))
    return float((lower + upper) / 2.0)


def calibrate_fiqa_thresholds(
    rows: list[dict],
    poor_precision: float = 0.80,
    clear_precision: float = 0.90,
) -> dict:
    valid_rows = [
        row for row in rows
        if row.get("fiqa") is not None and row.get("usable") is not None
    ]
    if not valid_rows:
        raise ValueError("没有可用于FIQA校准的样本")

    if not (0.0 < poor_precision <= 1.0 and 0.0 < clear_precision <= 1.0):
        raise ValueError("precision目标必须在 (0, 1] 区间内")

    sorted_rows = sorted(valid_rows, key=lambda row: float(row["fiqa"]))
    unique_fiqas = sorted({float(row["fiqa"]) for row in sorted_rows})
    total = len(sorted_rows)
    identities = len({row.get("pid") for row in sorted_rows if row.get("pid") is not None})

    poor_candidates = []
    for index, fiqa_value in enumerate(unique_fiqas):
        selected = [row for row in sorted_rows if float(row["fiqa"]) <= fiqa_value]
        if not selected:
            continue
        unusable = sum(not bool(row["usable"]) for row in selected)
        precision = unusable / len(selected)
        if precision >= poor_precision:
            poor_candidates.append(
                {
                    "threshold": _threshold_between(
                        fiqa_value,
                        unique_fiqas[index + 1] if index + 1 < len(unique_fiqas) else None,
                    ),
                    "precision": precision,
                    "coverage": len(selected) / total,
                    "count": len(selected),
                }
            )
    if not poor_candidates:
        raise ValueError(f"无法找到满足 poor_precision>={poor_precision:.3f} 的阈值")

    clear_candidates = []
    for fiqa_value in unique_fiqas:
        selected = [row for row in sorted_rows if float(row["fiqa"]) >= fiqa_value]
        if not selected:
            continue
        usable = sum(bool(row["usable"]) for row in selected)
        precision = usable / len(selected)
        if precision >= clear_precision:
            clear_candidates.append(
                {
                    "threshold": float(fiqa_value),
                    "precision": precision,
                    "coverage": len(selected) / total,
                    "count": len(selected),
                }
            )
    if not clear_candidates:
        raise ValueError(f"无法找到满足 clear_precision>={clear_precision:.3f} 的阈值")

    poor = max(
        poor_candidates,
        key=lambda item: (item["count"], item["precision"], item["threshold"]),
    )
    clear = max(
        clear_candidates,
        key=lambda item: (item["count"], item["precision"], -item["threshold"]),
    )
    if not poor["threshold"] < clear["threshold"]:
        raise ValueError(
            "校准结果无法分开 poor 与 clear 阈值："
            f"poor={poor['threshold']:.6f}, clear={clear['threshold']:.6f}"
        )

    return {
        "poor": {
            "threshold": round(float(poor["threshold"]), 6),
            "precision": round(float(poor["precision"]), 6),
            "coverage": round(float(poor["coverage"]), 6),
            "count": int(poor["count"]),
        },
        "clear": {
            "threshold": round(float(clear["threshold"]), 6),
            "precision": round(float(clear["precision"]), 6),
            "coverage": round(float(clear["coverage"]), 6),
            "count": int(clear["count"]),
        },
        "total_rows": total,
        "identities": identities,
        "poor_precision_target": float(poor_precision),
        "clear_precision_target": float(clear_precision),
        "method": (
            "poor阈值在满足 fiqa < t 中 unusable precision 目标的候选里选覆盖最大的前缀；"
            "clear阈值在满足 fiqa >= t 中 usable precision 目标的候选里选覆盖最大的后缀；"
            "两者必须满足 poor < clear。"
        ),
    }


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("RGB")


def calibrate_manifest(args: argparse.Namespace) -> int:
    from app import body_reid as body_mod
    from app import face as face_mod
    from app.core.config import settings

    data_dir = Path(args.data).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if settings.face_fiqa_backend in {"off", "none", ""}:
        raise ValueError("FACE_FIQA_BACKEND 未启用，无法校准 CR-FIQA 阈值")

    settings.face_rec_backend = "arcface"
    settings.face_superres = "off"

    tracklets = common.load_mevid_train(data_dir)
    if args.max_tracklets > 0:
        tracklets = tracklets[: args.max_tracklets]

    rows_by_pid = defaultdict(list)
    started = time.perf_counter()
    valid_tracks = 0
    skipped_tracks = 0
    for track_index, tracklet in enumerate(tracklets, start=1):
        try:
            chosen = select_product_best_frame(
                tracklet,
                args.frames_per_track,
                image_loader=_load_rgb_image,
                quality_fn=body_mod.assess_quality,
            )
            image = _load_rgb_image(chosen["path"])
            faces = face_mod.detect(
                image,
                with_quality=True,
                enhance_blurry=False,
                with_identity=True,
                with_geometry=False,
            )
            best_face = _best_detected_face(faces)
            quality = dict((best_face or {}).get("quality") or {})
            embedding = (best_face or {}).get("embedding")
            fiqa = quality.get("fiqa")
            if best_face is None or embedding is None or fiqa is None:
                skipped_tracks += 1
            else:
                rows_by_pid[tracklet.pid].append(
                    {
                        "pid": tracklet.pid,
                        "track": tracklet.track,
                        "best_idx": chosen["best_idx"],
                        "source_relpath": _relative_or_absolute(chosen["path"], data_dir),
                        "person_quality": chosen["person_quality"],
                        "person_quality_score": round(float(chosen["score"]), 6),
                        "fiqa": float(fiqa),
                        "embedding": common.l2norm(embedding),
                    }
                )
                valid_tracks += 1
        except Exception:
            skipped_tracks += 1

        if track_index % 20 == 0 or track_index == len(tracklets):
            print(f"    calibrate {track_index}/{len(tracklets)} tracks", flush=True)

    anchor_vectors = defaultdict(list)
    probe_rows = []
    eligible_pids = []
    anchor_tracks = 0
    for pid, rows in sorted(rows_by_pid.items()):
        ordered = sorted(rows, key=lambda row: (row["track"], row["best_idx"], row["source_relpath"]))
        if len(ordered) < 2:
            continue
        anchor_count = max(1, len(ordered) // 2)
        anchor_count = min(anchor_count, len(ordered) - 1)
        anchors = ordered[:anchor_count]
        probes = ordered[anchor_count:]
        anchor_vectors[pid].extend(row["embedding"] for row in anchors)
        probe_rows.extend(probes)
        eligible_pids.append(pid)
        anchor_tracks += len(anchors)

    templates = common.build_mean_templates(anchor_vectors)
    if not templates or not probe_rows:
        raise ValueError("校准样本不足：至少需要两个有效track且能形成anchor/probe划分")

    calibration_rows = []
    for row in probe_rows:
        pred, score = common.top1(row["embedding"], templates)
        calibration_rows.append(
            {
                "pid": row["pid"],
                "track": row["track"],
                "best_idx": row["best_idx"],
                "source_relpath": row["source_relpath"],
                "person_quality": row["person_quality"],
                "person_quality_score": row["person_quality_score"],
                "fiqa": round(float(row["fiqa"]), 6),
                "pred": pred,
                "confidence": round(float(score), 6),
                "usable": bool(pred == row["pid"] and score >= settings.face_hit_thresh),
            }
        )

    threshold_info = calibrate_fiqa_thresholds(
        calibration_rows,
        poor_precision=args.poor_precision,
        clear_precision=args.clear_precision,
    )
    payload = {
        "schema_version": 1,
        "kind": "fiqa_calibration",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_root_hint": data_dir.name,
        "calibration_config": {
            "frames_per_track": args.frames_per_track,
            "max_tracklets": args.max_tracklets,
            "face_rec_backend": settings.face_rec_backend,
            "face_superres": settings.face_superres,
            "face_fiqa_backend": settings.face_fiqa_backend,
            "face_fiqa_arch": settings.face_fiqa_arch,
            "poor_precision_target": args.poor_precision,
            "clear_precision_target": args.clear_precision,
        },
        "thresholds": {
            "poor": threshold_info["poor"],
            "clear": threshold_info["clear"],
        },
        "stats": {
            "total_rows": threshold_info["total_rows"],
            "identities": threshold_info["identities"],
            "eligible_identities": len(eligible_pids),
            "train_tracklets": len(tracklets),
            "valid_tracks": valid_tracks,
            "skipped_tracks": skipped_tracks,
            "anchor_tracks": anchor_tracks,
            "probe_tracks": len(calibration_rows),
            "face_hit_thresh": float(settings.face_hit_thresh),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        },
        "method": threshold_info["method"],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {output_path}")
    return 0


def prepare_manifest(args: argparse.Namespace) -> int:
    from app import body_reid as body_mod
    from app import face as face_mod
    from app.core.config import settings

    data_dir = Path(args.data).resolve()
    manifest_path = Path(args.manifest).resolve()
    calibration_path = Path(args.calibration).resolve()
    cache_root = Path(args.cache).resolve()
    aligned_dir = cache_root / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not calibration_path.is_file():
        raise FileNotFoundError(f"校准文件不存在：{calibration_path}")
    calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration_payload.get("kind") != "fiqa_calibration":
        raise ValueError("校准文件类型非法：必须是fiqa_calibration")
    calibration_config = calibration_payload.get("calibration_config") or {}
    calibrated_backend = str(
        calibration_config.get("face_fiqa_backend") or ""
    ).strip().lower()
    runtime_backend = str(settings.face_fiqa_backend or "").strip().lower()
    if calibrated_backend in {"", "off", "none"}:
        raise ValueError("校准文件没有记录已启用的FIQA后端")
    if runtime_backend != calibrated_backend:
        raise ValueError(
            "FIQA后端与Train校准不一致："
            f"calibration={calibrated_backend}, runtime={runtime_backend}"
        )
    calibrated_arch = str(calibration_config.get("face_fiqa_arch") or "").strip().lower()
    runtime_arch = str(settings.face_fiqa_arch or "").strip().lower()
    if calibrated_arch and runtime_arch != calibrated_arch:
        raise ValueError(
            "FIQA网络结构与Train校准不一致："
            f"calibration={calibrated_arch}, runtime={runtime_arch}"
        )
    poor_threshold = float(calibration_payload["thresholds"]["poor"]["threshold"])
    clear_threshold = float(calibration_payload["thresholds"]["clear"]["threshold"])
    if not poor_threshold < clear_threshold:
        raise ValueError("校准文件中的FIQA阈值非法：poor必须小于clear")

    settings.face_rec_backend = "arcface"
    settings.face_superres = "off"
    settings.face_fiqa_poor_thresh = poor_threshold
    settings.face_fiqa_clear_thresh = clear_threshold

    tracklets = common.load_mevid(data_dir)
    if args.max_tracklets > 0:
        tracklets = tracklets[: args.max_tracklets]

    samples = []
    started = time.perf_counter()
    calibration_sha = _sha256(calibration_path)
    for track_index, tracklet in enumerate(tracklets, start=1):
        sample_id = _sample_id(tracklet)
        row = {
            "sample_id": sample_id,
            "pid": tracklet.pid,
            "cam": tracklet.cam,
            "outfit": tracklet.outfit,
            "track": tracklet.track,
            "role": "query" if tracklet.is_query else "gallery",
            "best_idx": None,
            "candidate_count": 0,
            "person_quality": None,
            "person_quality_score": None,
            "source_relpath": None,
            "face_status": "none",
            "quality_bin": "none",
            "cause_bin": "none",
            "degradation_tags": ["none"],
            "quality": {},
            "bbox": None,
            "kps": None,
            "aligned_relpath": None,
            "aligned_sha256": None,
        }
        try:
            chosen = select_product_best_frame(
                tracklet,
                args.frames_per_track,
                image_loader=_load_rgb_image,
                quality_fn=body_mod.assess_quality,
            )
            row.update(
                {
                    "best_idx": chosen["best_idx"],
                    "candidate_count": chosen["candidate_count"],
                    "person_quality": chosen["person_quality"],
                    "person_quality_score": round(float(chosen["score"]), 6),
                    "source_relpath": _relative_or_absolute(chosen["path"], data_dir),
                }
            )
            image = _load_rgb_image(chosen["path"])
            faces = face_mod.detect(
                image,
                with_quality=True,
                enhance_blurry=False,
                with_identity=False,
                with_geometry=False,
            )
            best_face = _best_detected_face(faces)
            if best_face is not None:
                quality = dict(best_face.get("quality") or {})
                aligned = face_mod.align_face(image, best_face.get("kps"))
                category = quality.get("category", "poor")
                tags = _quality_tags(quality)
                row.update(
                    {
                        "face_status": "detected" if aligned is not None else "unaligned",
                        "quality_bin": category,
                        "cause_bin": _cause_bin(category, tags),
                        "degradation_tags": tags,
                        "quality": quality,
                        "bbox": best_face.get("bbox"),
                        "kps": best_face.get("kps"),
                    }
                )
                if aligned is not None:
                    aligned_path = aligned_dir / f"{sample_id}.png"
                    Image.fromarray(aligned[:, :, ::-1]).save(aligned_path)
                    row["aligned_relpath"] = _relative_or_absolute(aligned_path, common.ROOT)
                    row["aligned_sha256"] = _sha256(aligned_path)
        except Exception as exc:
            row["prepare_error"] = f"{type(exc).__name__}: {exc}"
        samples.append(row)

        if track_index % 20 == 0 or track_index == len(tracklets):
            print(f"    prepare {track_index}/{len(tracklets)} tracks", flush=True)

    payload = {
        "schema_version": 2,
        "kind": "superres_gate_manifest",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_root_hint": data_dir.name,
        "cache_root": _relative_or_absolute(cache_root, common.ROOT),
        "calibration": {
            "path": _relative_or_absolute(calibration_path, common.ROOT),
            "sha256": calibration_sha,
            "face_fiqa_poor_thresh": poor_threshold,
            "face_fiqa_clear_thresh": clear_threshold,
        },
        "prepare_config": {
            "frames_per_track": args.frames_per_track,
            "face_model": settings.face_model,
            "face_det_size": settings.face_det_size,
            "face_fiqa_backend": settings.face_fiqa_backend,
            "face_fiqa_poor_thresh": settings.face_fiqa_poor_thresh,
            "face_fiqa_clear_thresh": settings.face_fiqa_clear_thresh,
        },
        "samples": samples,
        "summary": {
            "tracklets": len(tracklets),
            "samples": len(samples),
            "detected": sum(row["face_status"] == "detected" for row in samples),
            "none": sum(row["quality_bin"] == "none" for row in samples),
            "by_quality_bin": {
                category: sum(row["quality_bin"] == category for row in samples)
                for category in common.BIN_ORDER
            },
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        },
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {manifest_path}")
    return 0


def select_variant_embeddings(
    original: np.ndarray | None,
    enhanced: np.ndarray | None,
    can_superres: bool,
) -> dict[str, np.ndarray | None]:
    """A/B/C固定样本选择；GFPGAN失败时B/C回退原图，不删除样本。"""
    enhanced_or_original = enhanced if enhanced is not None else original
    return {
        "A_original": original,
        "B_all_superres": enhanced_or_original,
        "C_gated_superres": enhanced_or_original if can_superres else original,
    }


def _score_rows(records, variant: str, templates: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for record in records:
        pred, score = common.top1(record["vectors"].get(variant), templates)
        rows.append(
            {
                "sample_id": record["sample_id"],
                "gt": record["pid"],
                "genuine": record["pid"] in templates,
                "quality_bin": record["quality_bin"],
                "cause_bin": record["cause_bin"],
                "degradation_tags": record["degradation_tags"],
                "pred": pred,
                "confidence": round(float(score), 6),
            }
        )
    return rows


def paired_transition_stats(
    base_rows: list[dict],
    candidate_rows: list[dict],
    thresholds: dict[str, float | None] | None = None,
) -> dict:
    base = {row["sample_id"]: row for row in base_rows}
    improved = degraded = changed = 0
    false_accepts = {
        key: {"increased": 0, "decreased": 0}
        for key in (thresholds or {})
    }
    for row in candidate_rows:
        before = base[row["sample_id"]]
        changed += int(before["pred"] != row["pred"])
        if row["genuine"]:
            before_correct = before["pred"] == before["gt"]
            after_correct = row["pred"] == row["gt"]
            improved += int(not before_correct and after_correct)
            degraded += int(before_correct and not after_correct)
        else:
            for key, threshold in (thresholds or {}).items():
                if threshold is None:
                    continue
                before_accept = (
                    before["pred"] is not None
                    and before["confidence"] >= threshold
                )
                after_accept = (
                    row["pred"] is not None
                    and row["confidence"] >= threshold
                )
                false_accepts[key]["increased"] += int(
                    not before_accept and after_accept
                )
                false_accepts[key]["decreased"] += int(
                    before_accept and not after_accept
                )
    return {
        "samples": len(candidate_rows),
        "prediction_changed": changed,
        "rank1_improved": improved,
        "rank1_degraded": degraded,
        "imposter_false_accept_transitions": false_accepts,
    }


def evaluate_manifest(args: argparse.Namespace) -> int:
    from app import face as face_mod
    from app import face_fiqa
    from app.core.config import settings

    manifest_path = Path(args.manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("kind") != "superres_gate_manifest":
        raise ValueError("不是superres_gate_manifest")

    settings.face_rec_backend = "arcface"
    settings.face_superres = "off"

    frozen_by_pid = defaultdict(list)
    for frozen in payload["samples"]:
        frozen_by_pid[frozen["pid"]].append(frozen)
    candidate_pids = sorted(
        pid
        for pid, rows in frozen_by_pid.items()
        if any(
            row["role"] == "gallery"
            and row.get("aligned_relpath")
            and (row.get("quality") or {}).get("can_match")
            for row in rows
        )
        and any(
            row["role"] == "query"
            and row.get("aligned_relpath")
            and (row.get("quality") or {}).get("can_match")
            for row in rows
        )
    )
    enroll_pids, imposter_pids = common.split_subjects(
        candidate_pids,
        args.enroll_subjects,
        args.imposter_subjects,
        args.seed,
    )
    enroll_set = set(enroll_pids)
    imposter_set = set(imposter_pids)
    selected_frozen = [
        frozen
        for frozen in payload["samples"]
        if frozen["pid"] in enroll_set or frozen["pid"] in imposter_set
    ]

    records = []
    setup_timing = defaultdict(float)
    for index, frozen in enumerate(selected_frozen, start=1):
        record = dict(frozen)
        record["vectors"] = {name: None for name in VARIANTS}
        aligned_relpath = frozen.get("aligned_relpath")
        aligned_path = _resolve_saved_path(aligned_relpath, common.ROOT) if aligned_relpath else None
        if aligned_path and not aligned_path.is_file():
            raise FileNotFoundError(f"固定输入不存在：{aligned_path}")
        if aligned_path and aligned_path.is_file():
            if frozen.get("aligned_sha256") and _sha256(aligned_path) != frozen["aligned_sha256"]:
                raise RuntimeError(f"固定输入被修改：{aligned_path}")
            aligned_bgr = np.asarray(Image.open(aligned_path).convert("RGB"))[:, :, ::-1].copy()
            if (frozen.get("quality") or {}).get("can_match"):
                t0 = time.perf_counter()
                original = face_mod.embed_aligned_face(aligned_bgr, "arcface")
                elapsed = time.perf_counter() - t0
                timing_key = (
                    "gallery_arcface_seconds"
                    if frozen["pid"] in enroll_set and frozen["role"] == "gallery"
                    else "evaluation_arcface_seconds"
                )
                setup_timing[timing_key] += elapsed
                record["arcface_original_seconds"] = elapsed
                record["vectors"]["A_original"] = (
                    common.l2norm(original) if original is not None else None
                )
        records.append(record)
        if index % 100 == 0 or index == len(selected_frozen):
            print(f"    original embeddings {index}/{len(selected_frozen)}", flush=True)

    gallery_by_pid = defaultdict(list)
    query_by_pid = defaultdict(list)
    gallery_records_by_pid = defaultdict(list)
    for record in records:
        if record["role"] == "gallery":
            gallery_records_by_pid[record["pid"]].append(record)
            vector = record["vectors"]["A_original"]
            if vector is not None:
                gallery_by_pid[record["pid"]].append(vector)
        else:
            query_by_pid[record["pid"]].append(record)

    templates = common.build_mean_templates(gallery_by_pid)
    missing_templates = [pid for pid in enroll_pids if pid not in templates]
    if missing_templates:
        raise RuntimeError(
            "固定建档身份无法生成A模板：" + ",".join(missing_templates)
        )
    fixed_templates = {pid: templates[pid] for pid in enroll_pids}

    evaluation_genuine = [
        record
        for pid in enroll_pids
        for record in query_by_pid[pid]
    ]
    evaluation_imposter = [
        record
        for pid in imposter_pids
        for record in [*gallery_records_by_pid[pid], *query_by_pid[pid]]
    ]
    evaluation_records = evaluation_genuine + evaluation_imposter

    settings.face_superres = "gfpgan"
    face_mod._ensure_superres()
    superres_startup_error = face_mod.superres_error()
    if superres_startup_error:
        print(f"[warn] GFPGAN不可用，B/C将回退A：{superres_startup_error}", flush=True)

    timing = defaultdict(float)
    counts = defaultdict(int)
    fiqa_deltas = []
    embedding_drifts = []
    for record in evaluation_records:
        original = record["vectors"].get("A_original")
        if original is None:
            continue
        timing["arcface_original_seconds"] += float(
            record.get("arcface_original_seconds") or 0.0
        )
        can_match = bool((record.get("quality") or {}).get("can_match"))
        can_superres = bool((record.get("quality") or {}).get("can_superres"))
        enhanced_embedding = None
        enhanced_bgr = None
        if can_match:
            counts["B_gfpgan_calls"] += 1
            aligned_path = _resolve_saved_path(record["aligned_relpath"], common.ROOT)
            aligned_bgr = np.asarray(
                Image.open(aligned_path).convert("RGB")
            )[:, :, ::-1].copy()
            t0 = time.perf_counter()
            aligned_rgb = Image.fromarray(aligned_bgr[:, :, ::-1])
            restored = face_mod.enhance(aligned_rgb, aligned=True)
            superres_elapsed = time.perf_counter() - t0
            timing["B_gfpgan_seconds"] += superres_elapsed
            if can_superres:
                timing["C_gfpgan_seconds"] += superres_elapsed
            if restored is not aligned_rgb:
                enhanced_bgr = np.asarray(
                    restored.convert("RGB")
                )[:, :, ::-1].copy()
                t0 = time.perf_counter()
                enhanced_embedding = face_mod.embed_aligned_face(
                    enhanced_bgr, "arcface"
                )
                enhanced_arcface_elapsed = time.perf_counter() - t0
                timing["B_arcface_enhanced_seconds"] += enhanced_arcface_elapsed
                if can_superres:
                    timing["C_arcface_enhanced_seconds"] += enhanced_arcface_elapsed
                counts["B_gfpgan_success"] += 1

        counts["C_gfpgan_calls"] += int(can_match and can_superres)
        variants = select_variant_embeddings(
            original, enhanced_embedding, can_superres
        )
        record["vectors"].update(
            {
                name: common.l2norm(vector) if vector is not None else None
                for name, vector in variants.items()
            }
        )
        if original is not None and enhanced_embedding is not None:
            embedding_drifts.append(float(original @ enhanced_embedding))
        if enhanced_bgr is not None:
            try:
                before = (record.get("quality") or {}).get("fiqa")
                after = face_fiqa.score(enhanced_bgr)
                if before is not None and after is not None:
                    fiqa_deltas.append(float(after) - float(before))
            except Exception:
                pass

    match_threshold = settings.face_hit_thresh
    thresholds = {"product": float(match_threshold)}
    evaluation_rows = {
        name: _score_rows(evaluation_records, name, fixed_templates)
        for name in VARIANTS
    }
    summaries = {
        name: common.summarize_with_thresholds(rows, thresholds)
        for name, rows in evaluation_rows.items()
    }

    result = {
        "schema_version": 2,
        "kind": "superres_gate_result",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "protocol": {
            "gallery_variant": "A_original",
            "threshold_source": "FACE_HIT_THRESH",
            "variants": {
                "A_original": "原始对齐脸→ArcFace",
                "B_all_superres": "所有can_match人脸→GFPGAN→ArcFace",
                "C_gated_superres": "仅冻结can_superres=true时使用超分ArcFace",
            },
            "enroll_subjects": enroll_pids,
            "imposter_subjects": imposter_pids,
            "match_threshold": float(match_threshold),
        },
        "results": summaries,
        "paired": {
            "B_vs_A": paired_transition_stats(
                evaluation_rows["A_original"],
                evaluation_rows["B_all_superres"],
                thresholds,
            ),
            "C_vs_A": paired_transition_stats(
                evaluation_rows["A_original"],
                evaluation_rows["C_gated_superres"],
                thresholds,
            ),
            "C_vs_B": paired_transition_stats(
                evaluation_rows["B_all_superres"],
                evaluation_rows["C_gated_superres"],
                thresholds,
            ),
        },
        "runtime": {
            **{key: round(value, 4) for key, value in setup_timing.items()},
            **{key: round(value, 4) for key, value in timing.items()},
            **counts,
            "superres_startup_error": superres_startup_error,
            "embedding_cosine_original_vs_superres_mean": (
                round(float(np.mean(embedding_drifts)), 6)
                if embedding_drifts
                else None
            ),
            "fiqa_delta_mean": (
                round(float(np.mean(fiqa_deltas)), 6) if fiqa_deltas else None
            ),
        },
    }
    output = (
        Path(args.output).resolve()
        if args.output
        else SUPERRES_DIR
        / "results"
        / f"superres_gate_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {output}")
    print(f"{'variant':>20}  {'rank1':>8}  {'TPIR':>8}  {'FMR':>8}")
    for name, summary in summaries.items():
        point = summary["operating_points"]["product"]
        rank1 = summary["rank1_rate"]
        print(
            f"{name:>20}  "
            f"{(100 * rank1 if rank1 is not None else 0):7.2f}%  "
            f"{100 * point['tpir']:7.2f}%  "
            f"{100 * point['fmr']:7.2f}%"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MEVID超分门控A/B/C实验")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate", help="用Train集校准CR-FIQA阈值")
    calibrate.add_argument("--data", required=True, help="MEVID根目录")
    calibrate.add_argument(
        "--output",
        default=str(SUPERRES_DIR / "manifests" / "fiqa_calibration.json"),
    )
    calibrate.add_argument(
        "--frames-per-track",
        type=int,
        default=0,
        help="0表示扫描整条轨迹；非0仅用于调试抽样",
    )
    calibrate.add_argument("--poor-precision", type=float, default=0.80)
    calibrate.add_argument("--clear-precision", type=float, default=0.90)
    calibrate.add_argument("--max-tracklets", type=int, default=0, help="0表示全部；smoke可设小值")
    calibrate.set_defaults(handler=calibrate_manifest)

    prepare = subparsers.add_parser("prepare", help="按产品best_idx冻结Test唯一对齐脸")
    prepare.add_argument("--data", required=True, help="MEVID根目录")
    prepare.add_argument(
        "--calibration",
        default=str(SUPERRES_DIR / "manifests" / "fiqa_calibration.json"),
        help="Train集FIQA校准JSON",
    )
    prepare.add_argument(
        "--manifest",
        default=str(SUPERRES_DIR / "manifests" / "superres_gate_test.json"),
    )
    prepare.add_argument(
        "--cache",
        default=str(common.ROOT / "data" / "generated" / "superres_gate"),
    )
    prepare.add_argument(
        "--frames-per-track",
        type=int,
        default=0,
        help="0表示扫描整条轨迹；非0仅用于调试抽样",
    )
    prepare.add_argument("--max-tracklets", type=int, default=0, help="0表示全部；smoke可设小值")
    prepare.set_defaults(handler=prepare_manifest)

    evaluate = subparsers.add_parser("evaluate", help="读取固定manifest运行A/B/C")
    evaluate.add_argument(
        "--manifest",
        default=str(SUPERRES_DIR / "manifests" / "superres_gate_test.json"),
    )
    evaluate.add_argument("--output", default="")
    evaluate.add_argument("--enroll-subjects", type=int, default=27)
    evaluate.add_argument(
        "--imposter-subjects",
        type=int,
        default=-1,
        help="-1表示使用建档后全部剩余可用身份",
    )
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.set_defaults(handler=evaluate_manifest)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
