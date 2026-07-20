"""固定真实难脸，配对比较原图与GFPGAN后的ArcFace身份特征。"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from common import mevid_eval_common as common  # noqa: E402

HARD_TAGS = {"blur", "small-face"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else common.ROOT / path


def _gallery_rank(row: dict) -> tuple:
    quality = row.get("quality") or {}
    defects = set(quality.get("defects") or [])
    return (
        not bool({"extreme_yaw", "extreme_pitch", "low_detection"} & defects),
        float(quality.get("det_score") or 0.0),
        float(quality.get("blur_var") or 0.0),
        float(quality.get("area") or 0.0),
        -abs(float(quality.get("yaw") or 0.0)),
        -abs(float(quality.get("pitch") or 0.0)),
    )


def _hardness(row: dict) -> tuple:
    quality = row.get("quality") or {}
    tags = set(row.get("degradation_tags") or [])
    defects = set(quality.get("defects") or [])
    return (
        "blur" in tags or "blur" in defects or "low_fiqa" in defects,
        "small-face" in tags or "small_face" in defects,
        -float(quality.get("blur_var") or 0.0),
        -float(quality.get("area") or 0.0),
    )


def is_hard_matchable_query(row: dict) -> bool:
    quality = row.get("quality") or {}
    tags = set(row.get("degradation_tags") or [])
    defects = set(quality.get("defects") or [])
    direct_tags = tags & HARD_TAGS
    direct_defects = defects & {"blur", "small_face"}
    return bool(
        row.get("role") == "query"
        and row.get("aligned_relpath")
        and quality.get("can_match", True)
        and (direct_tags or direct_defects)
    )


def select_one_hard_face_per_track(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if is_hard_matchable_query(row):
            grouped[
                (
                    row["pid"],
                    row.get("cam"),
                    row.get("outfit"),
                    row.get("track"),
                )
            ].append(row)
    return [
        max(group, key=_hardness)
        for _, group in sorted(grouped.items())
    ]


def _template(vectors: list[np.ndarray]) -> np.ndarray | None:
    return common.l2norm(np.mean(np.stack(vectors), axis=0)) if vectors else None


def _scores(
    vector: np.ndarray,
    templates: dict[str, np.ndarray],
    pid: str,
) -> dict:
    all_scores = {
        template_pid: float(template @ vector)
        for template_pid, template in templates.items()
    }
    pred = max(all_scores, key=all_scores.get)
    own = all_scores[pid]
    impostor = max(
        score for template_pid, score in all_scores.items()
        if template_pid != pid
    )
    return {
        "pred": pred,
        "top1_score": all_scores[pred],
        "genuine_score": own,
        "max_impostor_score": impostor,
        "margin": own - impostor,
        "rank1_correct": pred == pid,
    }


def summarize(rows: list[dict], prefix: str, threshold: float) -> dict:
    correct = sum(row[prefix]["rank1_correct"] for row in rows)
    accepted_correct = sum(
        row[prefix]["rank1_correct"]
        and row[prefix]["top1_score"] >= threshold
        for row in rows
    )
    false_accept = sum(
        not row[prefix]["rank1_correct"]
        and row[prefix]["top1_score"] >= threshold
        for row in rows
    )
    return {
        "queries": len(rows),
        "rank1_correct": correct,
        "rank1_rate": round(correct / len(rows), 6),
        "correct_accept_rate": round(accepted_correct / len(rows), 6),
        "false_accept_rate": round(false_accept / len(rows), 6),
        "mean_genuine_score": round(
            float(np.mean([row[prefix]["genuine_score"] for row in rows])), 6
        ),
        "median_genuine_score": round(
            float(np.median([row[prefix]["genuine_score"] for row in rows])), 6
        ),
        "mean_max_impostor_score": round(
            float(np.mean([row[prefix]["max_impostor_score"] for row in rows])), 6
        ),
        "mean_margin": round(
            float(np.mean([row[prefix]["margin"] for row in rows])), 6
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="真实难脸原图/GFPGAN ArcFace配对实验"
    )
    parser.add_argument(
        "--manifest",
        default="/results/superres_gate_manifest.json",
    )
    parser.add_argument(
        "--output",
        default="/results/paired_superres_identity.json",
    )
    parser.add_argument("--gallery-shots", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from app import face as face_mod
    from app.core.config import settings

    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    samples = payload["samples"]
    settings.face_rec_backend = "arcface"
    settings.face_superres = "gfpgan"

    gallery_rows = defaultdict(list)
    for row in samples:
        if (
            row.get("role") == "gallery"
            and row.get("aligned_relpath")
            and (row.get("quality") or {}).get("can_match", True)
        ):
            gallery_rows[row["pid"]].append(row)

    templates = {}
    gallery_manifest = {}
    for pid, rows in gallery_rows.items():
        chosen = sorted(rows, key=_gallery_rank, reverse=True)[: args.gallery_shots]
        vectors = []
        used = []
        for row in chosen:
            path = _path(row["aligned_relpath"])
            if not path.is_file():
                continue
            if row.get("aligned_sha256") and _sha256(path) != row["aligned_sha256"]:
                raise RuntimeError(f"Gallery固定输入被修改：{path}")
            bgr = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
            vector = face_mod.embed_aligned_face(bgr, "arcface")
            if vector is not None:
                vectors.append(common.l2norm(vector))
                used.append(row["sample_id"])
        template = _template(vectors)
        if template is not None:
            templates[pid] = template
            gallery_manifest[pid] = used

    queries = [
        row for row in select_one_hard_face_per_track(samples)
        if row["pid"] in templates
    ]
    if args.max_queries > 0 and len(queries) > args.max_queries:
        random.Random(args.seed).shuffle(queries)
        queries = queries[: args.max_queries]
    if not queries:
        raise RuntimeError("没有同时具备原图Gallery模板的真实难脸Query")

    face_mod._ensure_superres()
    superres_error = face_mod.superres_error()
    if superres_error:
        raise RuntimeError(f"GFPGAN不可用：{superres_error}")

    rows = []
    started = time.perf_counter()
    for index, query in enumerate(queries, start=1):
        path = _path(query["aligned_relpath"])
        if not path.is_file():
            raise FileNotFoundError(f"Query固定输入不存在：{path}")
        if query.get("aligned_sha256") and _sha256(path) != query["aligned_sha256"]:
            raise RuntimeError(f"Query固定输入被修改：{path}")

        original_bgr = np.asarray(
            Image.open(path).convert("RGB")
        )[:, :, ::-1].copy()
        original_vector = face_mod.embed_aligned_face(original_bgr, "arcface")
        if original_vector is None:
            continue
        original_vector = common.l2norm(original_vector)

        original_rgb = Image.fromarray(original_bgr[:, :, ::-1])
        t0 = time.perf_counter()
        restored = face_mod.enhance(original_rgb, aligned=True)
        superres_seconds = time.perf_counter() - t0
        if restored is original_rgb:
            continue
        enhanced_bgr = np.asarray(
            restored.convert("RGB")
        )[:, :, ::-1].copy()
        enhanced_vector = face_mod.embed_aligned_face(enhanced_bgr, "arcface")
        if enhanced_vector is None:
            continue
        enhanced_vector = common.l2norm(enhanced_vector)

        original_scores = _scores(original_vector, templates, query["pid"])
        enhanced_scores = _scores(enhanced_vector, templates, query["pid"])
        rows.append(
            {
                "sample_id": query["sample_id"],
                "pid": query["pid"],
                "track": query.get("track"),
                "quality_bin": query.get("quality_bin"),
                "degradation_tags": query.get("degradation_tags") or [],
                "blur_var": (query.get("quality") or {}).get("blur_var"),
                "face_area": (query.get("quality") or {}).get("area"),
                "feature_cosine": round(float(original_vector @ enhanced_vector), 6),
                "superres_seconds": round(superres_seconds, 6),
                "original": original_scores,
                "enhanced": enhanced_scores,
            }
        )
        if index % 50 == 0 or index == len(queries):
            print(f"    paired {index}/{len(queries)}", flush=True)

    threshold = settings.face_hit_thresh
    improved = sum(
        not row["original"]["rank1_correct"]
        and row["enhanced"]["rank1_correct"]
        for row in rows
    )
    degraded = sum(
        row["original"]["rank1_correct"]
        and not row["enhanced"]["rank1_correct"]
        for row in rows
    )
    result = {
        "schema_version": 1,
        "kind": "paired_superres_identity",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": args.manifest,
        "gallery_identities": len(templates),
        "gallery_shots": args.gallery_shots,
        "queries": len(rows),
        "query_selection": "one real blur/small-face aligned query per track",
        "match_threshold": threshold,
        "original": summarize(rows, "original", threshold),
        "enhanced": summarize(rows, "enhanced", threshold),
        "paired": {
            "wrong_to_correct": improved,
            "correct_to_wrong": degraded,
            "genuine_score_increased": sum(
                row["enhanced"]["genuine_score"]
                > row["original"]["genuine_score"]
                for row in rows
            ),
            "genuine_score_decreased": sum(
                row["enhanced"]["genuine_score"]
                < row["original"]["genuine_score"]
                for row in rows
            ),
            "margin_increased": sum(
                row["enhanced"]["margin"] > row["original"]["margin"]
                for row in rows
            ),
            "margin_decreased": sum(
                row["enhanced"]["margin"] < row["original"]["margin"]
                for row in rows
            ),
            "mean_feature_cosine": round(
                float(np.mean([row["feature_cosine"] for row in rows])), 6
            ),
        },
        "runtime": {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "gfpgan_seconds": round(
                sum(row["superres_seconds"] for row in rows), 3
            ),
        },
        "rows": rows,
        "gallery_manifest": gallery_manifest,
    }
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[saved] {args.output}")
    print(
        f"A rank1={result['original']['rank1_rate']:.3f} "
        f"B rank1={result['enhanced']['rank1_rate']:.3f} "
        f"improved={improved} degraded={degraded}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
