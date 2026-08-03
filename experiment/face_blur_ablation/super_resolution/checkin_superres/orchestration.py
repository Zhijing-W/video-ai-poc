"""Evaluate orchestration and the stable prepare/evaluate CLI."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from .common import (
    ARMS,
    SCHEMA_VERSION,
    SUPERRES_DIR,
    build_image_manifest_records,
    _relative,
)
from .embeddings import (
    _compute_embedding_cache,
    _load_embedding_cache,
    _verify_manifest,
)
from .metrics import _score, _templates, paired_uncertainty, summarize_scores
from .matrix import FROZEN_MANIFEST_ID, evaluate_matrix
from .preparation import (
    _model_provenance,
    _provenance_compatible,
    prepare,
)
from .visualization import _comparison


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

    matrix_parser = subparsers.add_parser(
        "evaluate-matrix",
        help="仅在指定冻结manifest上运行GFPGAN/CodeFormer/RealESRGAN固定矩阵",
    )
    matrix_parser.add_argument(
        "--manifest",
        required=True,
        help=f"必须是manifest ID {FROZEN_MANIFEST_ID}",
    )
    matrix_parser.add_argument("--output", required=True)
    matrix_parser.add_argument(
        "--artifact-root",
        "--artifact-dir",
        dest="artifact_root",
        required=True,
    )
    matrix_parser.add_argument("--seed", type=int, default=0)
    matrix_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    matrix_parser.add_argument("--permutation-samples", type=int, default=20000)
    matrix_parser.add_argument("--force-recompute", action="store_true")
    matrix_parser.set_defaults(handler=evaluate_matrix)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("frames_per_track", "top_k", "min_gap_frames", "gallery_shots"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')}必须大于0")
    return args.handler(args)
