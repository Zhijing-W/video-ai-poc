"""MEVID超分门控实验：固定输入后比较原图、全量超分和产品门控超分。"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SUPERRES_DIR = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = SUPERRES_DIR.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from common import mevid_eval_common as common  # noqa: E402

VARIANTS = ("A_original", "B_all_superres", "C_gated_superres")
CATEGORY_RANK = {"none": 0, "poor": 1, "marginal": 2, "clear": 3}


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


def _sample_id(tracklet: common.Tracklet, source: Path) -> str:
    suffix = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    return (
        f"{tracklet.pid}_c{tracklet.cam}_o{tracklet.outfit}_"
        f"t{tracklet.track}_{source.stem}_{suffix}"
    )


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


def prepare_manifest(args: argparse.Namespace) -> int:
    from app import face as face_mod
    from app.core.config import settings

    data_dir = Path(args.data).resolve()
    manifest_path = Path(args.manifest).resolve()
    cache_root = Path(args.cache).resolve()
    aligned_dir = cache_root / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # 分类阶段固定关闭身份识别和超分；FIQA是否启用由产品配置决定。
    settings.face_rec_backend = "arcface"
    settings.face_superres = "off"
    tracklets = common.load_mevid(data_dir)
    if args.max_tracklets > 0:
        tracklets = tracklets[: args.max_tracklets]

    samples = []
    started = time.perf_counter()
    for track_index, tracklet in enumerate(tracklets, start=1):
        chosen = common.sample_evenly(tracklet.frames, args.frames_per_track)
        for source in chosen:
            sample_id = _sample_id(tracklet, source)
            row = {
                "sample_id": sample_id,
                "pid": tracklet.pid,
                "cam": tracklet.cam,
                "outfit": tracklet.outfit,
                "track": tracklet.track,
                "role": "query" if tracklet.is_query else "gallery",
                "source_relpath": _relative_or_absolute(source, data_dir),
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
                image = Image.open(source).convert("RGB")
                faces = face_mod.detect(
                    image,
                    with_quality=True,
                    enhance_blurry=False,
                    with_identity=False,
                    with_geometry=False,
                )
                best = max(faces, key=lambda item: float(item.get("det_score", 0.0))) if faces else None
                if best is not None:
                    quality = dict(best.get("quality") or {})
                    aligned = face_mod.align_face(image, best.get("kps"))
                    category = quality.get("category", "poor")
                    tags = _quality_tags(quality)
                    row.update(
                        {
                            "face_status": "detected" if aligned is not None else "unaligned",
                            "quality_bin": category,
                            "cause_bin": _cause_bin(category, tags),
                            "degradation_tags": tags,
                            "quality": quality,
                            "bbox": best.get("bbox"),
                            "kps": best.get("kps"),
                        }
                    )
                    if aligned is not None:
                        aligned_path = aligned_dir / f"{sample_id}.png"
                        Image.fromarray(aligned[:, :, ::-1]).save(aligned_path)
                        row["aligned_relpath"] = _relative_or_absolute(aligned_path, common.ROOT)
                        row["aligned_sha256"] = _sha256(aligned_path)
            except Exception as exc:  # 单帧失败仍保留在固定输入清单
                row["prepare_error"] = f"{type(exc).__name__}: {exc}"
            samples.append(row)

        if track_index % 20 == 0 or track_index == len(tracklets):
            print(f"    prepare {track_index}/{len(tracklets)} tracks", flush=True)

    payload = {
        "schema_version": 1,
        "kind": "superres_gate_manifest",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_root_hint": data_dir.name,
        "cache_root": _relative_or_absolute(cache_root, common.ROOT),
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


def _aggregate_track(samples: list[dict], variant: str) -> np.ndarray | None:
    vectors = []
    weights = []
    for sample in samples:
        vector = sample["embeddings"].get(variant)
        if vector is None:
            continue
        vectors.append(vector)
        weights.append(max(0.05, float((sample.get("quality") or {}).get("quality") or 0.0)))
    if not vectors:
        return None
    return common.l2norm(np.average(np.stack(vectors), axis=0, weights=np.asarray(weights)))


def _best_quality(samples: list[dict]) -> dict:
    return max(
        samples,
        key=lambda item: (
            CATEGORY_RANK.get(item.get("quality_bin", "none"), 0),
            float((item.get("quality") or {}).get("quality") or 0.0),
        ),
    )


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

    samples = []
    setup_timing = defaultdict(float)
    for index, frozen in enumerate(selected_frozen, start=1):
        sample = dict(frozen)
        sample["embeddings"] = {name: None for name in VARIANTS}
        aligned_relpath = frozen.get("aligned_relpath")
        aligned_path = _resolve_saved_path(aligned_relpath, common.ROOT) if aligned_relpath else None
        if aligned_path and not aligned_path.is_file():
            raise FileNotFoundError(f"固定输入不存在：{aligned_path}")
        if aligned_path and aligned_path.is_file():
            if frozen.get("aligned_sha256") and _sha256(aligned_path) != frozen["aligned_sha256"]:
                raise RuntimeError(f"固定输入被修改：{aligned_path}")
            aligned_bgr = np.asarray(Image.open(aligned_path).convert("RGB"))[:, :, ::-1].copy()

            can_match = bool((frozen.get("quality") or {}).get("can_match"))
            original = None
            if can_match:
                t0 = time.perf_counter()
                original = face_mod.embed_aligned_face(aligned_bgr, "arcface")
                elapsed = time.perf_counter() - t0
                timing_key = (
                    "gallery_arcface_seconds"
                    if frozen["pid"] in enroll_set and frozen["role"] == "gallery"
                    else "evaluation_arcface_seconds"
                )
                setup_timing[timing_key] += elapsed
                sample["arcface_original_seconds"] = elapsed
                sample["embeddings"]["A_original"] = (
                    common.l2norm(original) if original is not None else None
                )
        samples.append(sample)
        if index % 100 == 0 or index == len(selected_frozen):
            print(f"    original embeddings {index}/{len(selected_frozen)}", flush=True)

    grouped = defaultdict(list)
    for sample in samples:
        key = (
            sample["pid"],
            sample["cam"],
            sample["outfit"],
            sample["track"],
            sample["role"],
        )
        grouped[key].append(sample)

    track_records = []
    for (pid, cam, outfit, track, role), rows in grouped.items():
        best = _best_quality(rows)
        track_records.append(
            {
                "sample_id": f"{pid}:{cam}:{outfit}:{track}:{role}",
                "pid": pid,
                "cam": cam,
                "outfit": outfit,
                "track": track,
                "role": role,
                "quality_bin": best["quality_bin"],
                "cause_bin": best["cause_bin"],
                "degradation_tags": best["degradation_tags"],
                "vectors": {name: _aggregate_track(rows, name) for name in VARIANTS},
                "samples": rows,
            }
        )

    gallery_by_pid = defaultdict(list)
    query_by_pid = defaultdict(list)
    gallery_records_by_pid = defaultdict(list)
    for record in track_records:
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

    # 不再拆校准集：使用产品固定阈值，所有query都进入最终评测。
    evaluation_genuine = [
        record
        for pid in enroll_pids
        for record in query_by_pid[pid]
    ]
    # 陌生身份未进入数据库，因此其官方gallery/query轨迹都可作为未知测试片段。
    evaluation_imposter = [
        record
        for pid in imposter_pids
        for record in [*gallery_records_by_pid[pid], *query_by_pid[pid]]
    ]
    evaluation_records = evaluation_genuine + evaluation_imposter

    # B/C只在最终评测样本上运行。Gallery始终只使用A原图向量。
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
        for sample in record["samples"]:
            original = sample["embeddings"].get("A_original")
            if original is None:
                continue
            timing["arcface_original_seconds"] += float(
                sample.get("arcface_original_seconds") or 0.0
            )
            can_match = bool((sample.get("quality") or {}).get("can_match"))
            can_superres = bool((sample.get("quality") or {}).get("can_superres"))
            enhanced_embedding = None
            enhanced_bgr = None
            if can_match:
                counts["B_gfpgan_calls"] += 1
                aligned_path = _resolve_saved_path(sample["aligned_relpath"], common.ROOT)
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
            sample["embeddings"].update(
                {
                    name: common.l2norm(vector) if vector is not None else None
                    for name, vector in variants.items()
                }
            )
            if original is not None and enhanced_embedding is not None:
                embedding_drifts.append(float(original @ enhanced_embedding))
            if enhanced_bgr is not None:
                try:
                    before = (sample.get("quality") or {}).get("fiqa")
                    after = face_fiqa.score(enhanced_bgr)
                    if before is not None and after is not None:
                        fiqa_deltas.append(float(after) - float(before))
                except Exception:
                    pass

        record["vectors"]["B_all_superres"] = _aggregate_track(
            record["samples"], "B_all_superres"
        )
        record["vectors"]["C_gated_superres"] = _aggregate_track(
            record["samples"], "C_gated_superres"
        )

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
        "schema_version": 1,
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

    prepare = subparsers.add_parser("prepare", help="只检测、质量分桶并冻结112×112原始对齐脸")
    prepare.add_argument("--data", required=True, help="MEVID根目录")
    prepare.add_argument(
        "--manifest",
        default=str(SUPERRES_DIR / "manifests" / "superres_gate.json"),
    )
    prepare.add_argument(
        "--cache",
        default=str(common.ROOT / "data" / "generated" / "superres_gate"),
    )
    prepare.add_argument("--frames-per-track", type=int, default=8)
    prepare.add_argument("--max-tracklets", type=int, default=0, help="0表示全部；smoke可设小值")
    prepare.set_defaults(handler=prepare_manifest)

    evaluate = subparsers.add_parser("evaluate", help="读取固定manifest运行A/B/C")
    evaluate.add_argument(
        "--manifest",
        default=str(SUPERRES_DIR / "manifests" / "superres_gate.json"),
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
