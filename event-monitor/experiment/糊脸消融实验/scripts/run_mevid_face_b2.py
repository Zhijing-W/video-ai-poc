"""MEVID 协议 B2：共同多样化锚点、独立阈值校准和重复划分的人脸模型对比。"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import mevid_eval_common as common


sys.path.insert(0, str(common.ROOT))

VARIANTS = {
    "FR0": {"backend": "arcface", "superres": False, "note": "ArcFace"},
    "FR1": {"backend": "adaface", "superres": False, "note": "AdaFace"},
    "FR2": {"backend": "arcface", "superres": True, "note": "ArcFace + GFPGAN"},
    "FR3": {"backend": "adaface", "superres": True, "note": "AdaFace + GFPGAN"},
}


def parse_targets(value: str) -> tuple[float, ...]:
    targets = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not targets or any(item < 0 or item > 1 for item in targets):
        raise argparse.ArgumentTypeError("FMR targets 必须是 0..1 的逗号分隔小数")
    return targets


def _query_rows(records, templates, variant: str, quality_variant: str):
    rows = []
    for pid, track_id, record in records:
        slot = record["face"][variant]
        quality_slot = record["face"][quality_variant]
        vector = common.aggregate_face(slot)
        pred, score = common.top1(vector, templates)
        rows.append(
            {
                "sample_id": f"{pid}:{track_id}",
                "gt": pid,
                "genuine": pid in templates,
                "quality_bin": quality_slot["best_cat"],
                "cause_bin": common.quality_cause_bucket(quality_slot),
                "degradation_tags": common.quality_degradation_tags(quality_slot),
                "pred": pred,
                "confidence": round(float(score), 6),
                "detected_frames": sum(
                    1 for frame in quality_slot["frames"] if frame["emb"] is not None
                ),
                "enhanced_frames": slot["enhanced"],
                "superres_attempted_frames": slot["superres_attempted"],
            }
        )
    return rows


def _split_query_records(
    records_by_pid,
    pids,
    calibration_tracks: int,
    evaluation_tracks: int,
    seed: int,
):
    calibration = []
    evaluation = []
    for pid in pids:
        records = list(records_by_pid[pid])
        random.Random(seed * 1009 + int(pid)).shuffle(records)
        calibration.extend(records[:calibration_tracks])
        evaluation.extend(records[calibration_tracks:calibration_tracks + evaluation_tracks])
    return calibration, evaluation


def _paired_superres_stats(base_rows: list[dict], enhanced_rows: list[dict]) -> dict:
    base = {row["sample_id"]: row for row in base_rows}
    attempted = [
        row
        for row in enhanced_rows
        if row["superres_attempted_frames"] > 0 and row["sample_id"] in base
    ]
    selected = [
        row
        for row in attempted
        if row["enhanced_frames"] > 0 and row["sample_id"] in base
    ]
    changed = improved = degraded = 0
    for row in selected:
        before = base[row["sample_id"]]
        if before["pred"] != row["pred"]:
            changed += 1
        before_correct = before["pred"] == before["gt"]
        after_correct = row["pred"] == row["gt"]
        improved += int(not before_correct and after_correct)
        degraded += int(before_correct and not after_correct)
    return {
        "attempted_tracks": len(attempted),
        "triggered_tracks": len(selected),
        "attempted_but_not_enhanced": len(attempted) - len(selected),
        "prediction_changed": changed,
        "rank1_improved": improved,
        "rank1_degraded": degraded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MEVID B2：共同多样化锚点、轨迹级独立校准、重复划分的人脸模型对比"
    )
    parser.add_argument("--data", required=True, help="MEVID 根目录")
    parser.add_argument("--variants", default="FR0,FR1,FR2,FR3")
    parser.add_argument("--enroll-subjects", type=int, default=27)
    parser.add_argument("--imposter-subjects", type=int, default=-1)
    parser.add_argument("--anchors-per-subject", type=int, default=3)
    parser.add_argument("--min-anchors-per-subject", type=int, default=1)
    parser.add_argument("--anchor-candidates-per-track", type=int, default=8)
    parser.add_argument("--max-gallery-tracks", type=int, default=5)
    parser.add_argument("--faces-per-track", type=int, default=8)
    parser.add_argument("--calibration-query-tracks", type=int, default=1)
    parser.add_argument("--evaluation-query-tracks", type=int, default=1)
    parser.add_argument("--calibration-imposter-gallery-tracks", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fmr-targets", type=parse_targets, default=(0.01, 0.05, 0.10))
    args = parser.parse_args()

    if args.repeats <= 0:
        parser.error("--repeats 必须大于 0")
    if args.calibration_query_tracks <= 0 or args.evaluation_query_tracks <= 0:
        parser.error("校准和评测 query 轨迹数必须大于 0")
    if args.calibration_imposter_gallery_tracks < 0:
        parser.error("--calibration-imposter-gallery-tracks 不能小于 0")

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        parser.error(f"未知 variant：{','.join(unknown)}；可选 {','.join(VARIANTS)}")
    selected_variants = {name: VARIANTS[name] for name in variants}
    extraction_variants = dict(selected_variants)
    extraction_variants.setdefault("FR0", VARIANTS["FR0"])
    quality_variant = "FR0"

    from app import face as face_mod
    from app import gait as gait_mod
    from app import body_gallery as gallery_mod
    from app import body_reid as reid_mod
    from app.core.config import settings

    settings.face_superres = (
        "gfpgan" if any(VARIANTS[name]["superres"] for name in variants) else "off"
    )
    if any(VARIANTS[name]["superres"] for name in variants):
        face_mod._ensure_superres()
        error = face_mod.superres_error()
        if error:
            raise RuntimeError(f"GFPGAN 启动检查失败：{error}")
    if any(VARIANTS[name]["backend"] == "adaface" for name in variants):
        probe = face_mod.embed_aligned_face(
            np.zeros((112, 112, 3), dtype=np.uint8),
            "adaface",
        )
        error = face_mod.adaface_error()
        if error or probe is None or probe.shape != (512,):
            raise RuntimeError(f"AdaFace 启动检查失败：{error or 'embedding 输出非法'}")

    data_dir = Path(args.data)
    tracklets = common.load_mevid(data_dir)
    grouped = common.group_tracklets(tracklets)
    required_queries = (
        args.calibration_query_tracks + args.evaluation_query_tracks
    )
    valid_pids = sorted(
        pid
        for pid, split in grouped.items()
        if split["gallery"] and len(split["query"]) >= required_queries
    )
    print(
        f"[*] B2 加载完成：轨迹={len(tracklets)}，满足 gallery + "
        f"{required_queries} 条 query 的身份={len(valid_pids)}"
    )

    anchor_manifest = {}
    anchor_paths_by_pid = {}
    anchor_scan_started = time.time()
    for index, pid in enumerate(valid_pids, start=1):
        manifest = common.select_diverse_gallery_face_anchors(
            grouped[pid]["gallery"][:args.max_gallery_tracks],
            face_mod,
            settings,
            args.anchor_candidates_per_track,
            args.anchors_per_subject,
        )
        anchor_manifest[pid] = [
            {
                **{key: value for key, value in item.items() if key not in {"path", "rank"}},
                "path": str(item["path"].relative_to(data_dir)),
            }
            for item in manifest
        ]
        anchor_paths_by_pid[pid] = [item["path"] for item in manifest]
        if index % 10 == 0 or index == len(valid_pids):
            print(f"    anchor scan {index}/{len(valid_pids)}")

    candidate_pids = [
        pid
        for pid in valid_pids
        if len(anchor_paths_by_pid[pid]) >= args.min_anchors_per_subject
    ]
    if len(candidate_pids) < args.enroll_subjects:
        raise RuntimeError(
            f"仅 {len(candidate_pids)} 个身份满足共同锚点要求，"
            f"不足以建档 {args.enroll_subjects} 人"
        )
    print(
        f"[*] 共同锚点可用身份={len(candidate_pids)}/{len(valid_pids)}，"
        f"扫描耗时={time.time() - anchor_scan_started:.0f}s"
    )

    anchor_records = {}
    for index, pid in enumerate(candidate_pids, start=1):
        paths = anchor_paths_by_pid[pid]
        pseudo_track = common.Tracklet(
            pid=pid,
            cam=-1,
            outfit=-1,
            track=-1,
            frames=paths,
            is_query=False,
        )
        anchor_records[pid] = common.extract_track_features(
            pseudo_track,
            face_mod,
            reid_mod,
            gait_mod,
            gallery_mod,
            settings,
            extraction_variants,
            len(paths),
            0,
            need_body=False,
            need_gait=False,
        )
        if index % 10 == 0 or index == len(candidate_pids):
            print(f"    anchor embed {index}/{len(candidate_pids)}")

    common_template_pids = []
    anchor_vectors = {name: {} for name in variants}
    for pid, record in anchor_records.items():
        vectors = {
            name: common.aggregate_face(record["face"][name])
            for name in variants
        }
        if all(vector is not None for vector in vectors.values()):
            common_template_pids.append(pid)
            for name, vector in vectors.items():
                anchor_vectors[name][pid] = vector
    if len(common_template_pids) < args.enroll_subjects:
        raise RuntimeError(
            f"仅 {len(common_template_pids)} 个身份在所有 variant 中均能建立模板，"
            f"不足以建档 {args.enroll_subjects} 人"
        )
    print(f"[*] 所有 variant 共同模板身份={len(common_template_pids)}")

    query_records_by_pid = defaultdict(list)
    query_work = [
        (pid, tracklet)
        for pid in valid_pids
        for tracklet in grouped[pid]["query"]
    ]
    feature_started = time.time()
    for index, (pid, tracklet) in enumerate(query_work, start=1):
        record = common.extract_track_features(
            tracklet,
            face_mod,
            reid_mod,
            gait_mod,
            gallery_mod,
            settings,
            extraction_variants,
            args.faces_per_track,
            0,
            need_body=False,
            need_gait=False,
        )
        query_records_by_pid[pid].append((pid, tracklet.track, record))
        if index % 20 == 0 or index == len(query_work):
            elapsed = time.time() - feature_started
            print(f"    query embed {index}/{len(query_work)}  {elapsed:.0f}s")

    imposter_gallery_records_by_pid = defaultdict(list)
    imposter_gallery_work = [
        (pid, tracklet)
        for pid in valid_pids
        for tracklet in grouped[pid]["gallery"][
            :args.calibration_imposter_gallery_tracks
        ]
    ]
    gallery_feature_started = time.time()
    for index, (pid, tracklet) in enumerate(imposter_gallery_work, start=1):
        record = common.extract_track_features(
            tracklet,
            face_mod,
            reid_mod,
            gait_mod,
            gallery_mod,
            settings,
            extraction_variants,
            args.faces_per_track,
            0,
            need_body=False,
            need_gait=False,
        )
        imposter_gallery_records_by_pid[pid].append(
            (pid, tracklet.track, record)
        )
        if index % 20 == 0 or index == len(imposter_gallery_work):
            elapsed = time.time() - gallery_feature_started
            print(
                f"    imposter gallery embed {index}/{len(imposter_gallery_work)}  "
                f"{elapsed:.0f}s"
            )

    if any(VARIANTS[name]["backend"] == "adaface" for name in variants):
        error = face_mod.adaface_error()
        if error:
            raise RuntimeError(f"AdaFace 加载/推理失败：{error}")
    if any(VARIANTS[name]["superres"] for name in variants):
        error = face_mod.superres_error()
        if error:
            raise RuntimeError(f"GFPGAN 加载/推理失败：{error}")

    repetitions = []
    for repeat in range(args.repeats):
        split_seed = args.seed + repeat
        enroll_pool = sorted(common_template_pids)
        random.Random(split_seed).shuffle(enroll_pool)
        enroll_pids = enroll_pool[:args.enroll_subjects]

        remaining = [pid for pid in valid_pids if pid not in set(enroll_pids)]
        random.Random(split_seed + 100_000).shuffle(remaining)
        requested_imposters = (
            len(remaining) if args.imposter_subjects < 0 else args.imposter_subjects
        )
        if requested_imposters <= 0 or requested_imposters > len(remaining):
            raise RuntimeError(
                f"repeat={repeat} 最多可使用 {len(remaining)} 个冒充身份，"
                f"收到 {requested_imposters}"
            )
        imposter_pids = remaining[:requested_imposters]

        calibration_genuine, evaluation_genuine = _split_query_records(
            query_records_by_pid,
            enroll_pids,
            args.calibration_query_tracks,
            args.evaluation_query_tracks,
            split_seed,
        )
        calibration_imposter, evaluation_imposter = _split_query_records(
            query_records_by_pid,
            imposter_pids,
            args.calibration_query_tracks,
            args.evaluation_query_tracks,
            split_seed,
        )
        calibration_imposter.extend(
            record
            for pid in imposter_pids
            for record in imposter_gallery_records_by_pid[pid]
        )
        calibration_records = calibration_genuine + calibration_imposter
        evaluation_records = evaluation_genuine + evaluation_imposter

        repeat_results = {}
        eval_rows_by_variant = {}
        for name in variants:
            templates = {
                pid: anchor_vectors[name][pid]
                for pid in enroll_pids
            }
            calibration_rows = _query_rows(
                calibration_records,
                templates,
                name,
                quality_variant,
            )
            calibration_curve = common.det_curve(calibration_rows)
            calibration_points = {
                f"{target:.3f}": common.select_operating_point(
                    calibration_curve,
                    target,
                )
                for target in args.fmr_targets
            }
            thresholds = {
                key: point["threshold"]
                for key, point in calibration_points.items()
            }
            evaluation_rows = _query_rows(
                evaluation_records,
                templates,
                name,
                quality_variant,
            )
            eval_rows_by_variant[name] = evaluation_rows
            repeat_results[name] = {
                "note": VARIANTS[name]["note"],
                "anchor_coverage": len(templates),
                "calibration": {
                    "genuine_tracks": sum(row["genuine"] for row in calibration_rows),
                    "imposter_tracks": sum(not row["genuine"] for row in calibration_rows),
                    "operating_points": calibration_points,
                    "det": calibration_curve,
                },
                "evaluation": common.summarize_with_thresholds(
                    evaluation_rows,
                    thresholds,
                ),
                "detected_probe_frames": sum(
                    row["detected_frames"] for row in evaluation_rows
                ),
                "enhanced_probe_frames": sum(
                    row["enhanced_frames"] for row in evaluation_rows
                ),
                "superres_attempted_probe_frames": sum(
                    row["superres_attempted_frames"] for row in evaluation_rows
                ),
                "rows": evaluation_rows,
            }

        paired = {}
        if "FR0" in eval_rows_by_variant and "FR2" in eval_rows_by_variant:
            paired["FR2_vs_FR0"] = _paired_superres_stats(
                eval_rows_by_variant["FR0"],
                eval_rows_by_variant["FR2"],
            )
        if "FR1" in eval_rows_by_variant and "FR3" in eval_rows_by_variant:
            paired["FR3_vs_FR1"] = _paired_superres_stats(
                eval_rows_by_variant["FR1"],
                eval_rows_by_variant["FR3"],
            )

        repetitions.append(
            {
                "repeat": repeat,
                "seed": split_seed,
                "enroll_pids": enroll_pids,
                "imposter_pids": imposter_pids,
                "results": repeat_results,
                "superres_paired": paired,
            }
        )
        print(f"[*] repeat {repeat + 1}/{args.repeats} 完成")

    aggregate = {}
    for name in variants:
        rank1_values = [
            item["results"][name]["evaluation"]["rank1_rate"]
            for item in repetitions
        ]
        operating = {}
        for target in args.fmr_targets:
            key = f"{target:.3f}"
            tpir_values = [
                item["results"][name]["evaluation"]["operating_points"][key]["tpir"]
                for item in repetitions
            ]
            fmr_values = [
                item["results"][name]["evaluation"]["operating_points"][key]["fmr"]
                for item in repetitions
            ]
            threshold_values = [
                item["results"][name]["calibration"]["operating_points"][key]["threshold"]
                for item in repetitions
                if item["results"][name]["calibration"]["operating_points"][key]["threshold"]
                is not None
            ]
            operating[key] = {
                "tpir": common.mean_ci95(tpir_values),
                "actual_fmr": common.mean_ci95(fmr_values),
                "calibrated_threshold": common.mean_ci95(
                    threshold_values,
                    clip=None,
                ),
            }
        cause_bins = {}
        for cause in common.CAUSE_TAG_ORDER:
            rank_values = [
                item["results"][name]["evaluation"]["by_degradation_tag"][cause]["rank1_rate"]
                for item in repetitions
                if item["results"][name]["evaluation"]["by_degradation_tag"][cause]["rank1_rate"]
                is not None
            ]
            cause_item = {
                "total_tracks": sum(
                    item["results"][name]["evaluation"]["by_degradation_tag"][cause]["total"]
                    for item in repetitions
                ),
                "rank1": common.mean_ci95(rank_values),
                "operating_points": {},
            }
            for target in args.fmr_targets:
                key = f"{target:.3f}"
                values = [
                    item["results"][name]["evaluation"]["by_degradation_tag"][cause][
                        f"tpir_at_{key}"
                    ]
                    for item in repetitions
                    if item["results"][name]["evaluation"]["by_degradation_tag"][cause][
                        f"tpir_at_{key}"
                    ]
                    is not None
                ]
                cause_item["operating_points"][key] = common.mean_ci95(values)
            cause_bins[cause] = cause_item
        aggregate[name] = {
            "note": VARIANTS[name]["note"],
            "rank1": common.mean_ci95(rank1_values),
            "operating_points": operating,
            "by_cause_bin": cause_bins,
            "enhanced_probe_frames_total": sum(
                item["results"][name]["enhanced_probe_frames"]
                for item in repetitions
            ),
            "superres_attempted_probe_frames_total": sum(
                item["results"][name]["superres_attempted_probe_frames"]
                for item in repetitions
            ),
        }

    print("\n================ MEVID Face B2 · repeated evaluation ================")
    header = (
        f"{'variant':<8}{'Rank-1 mean':>14}"
        + "".join(f"{'TPIR@'+str(target):>14}" for target in args.fmr_targets)
    )
    print(header)
    for name, result in aggregate.items():
        row = f"{name:<8}{100 * result['rank1']['mean']:>13.1f}%"
        for target in args.fmr_targets:
            row += (
                f"{100 * result['operating_points'][f'{target:.3f}']['tpir']['mean']:>13.1f}%"
            )
        print(row)
    print(
        "注：当前阈值使用同一批 test 身份的独立 query 轨迹校准；"
        "不存在轨迹泄漏，但绝对 TPIR@FMR 仍可能偏乐观，不能直接与独立身份校准的文献结果比较。"
    )

    payload = {
        "protocol": "MEVID-FACE-B2",
        "purpose": (
            "共同多样化 gallery 锚点；query 轨迹级校准/评测隔离；"
            "重复身份划分比较 ArcFace/AdaFace/GFPGAN"
        ),
        "limitations": (
            "当前未下载 bbox_train。genuine 阈值校准使用同一批 test 身份的独立 query 轨迹，"
            "imposter 校准使用库外身份的独立 query 与 gallery 轨迹；评测只使用未参与校准的 query。"
            "不存在图像/轨迹泄漏，但仍弱于使用独立 train 身份校准"
        ),
        "config": {
            "variants": variants,
            "enroll_subjects": args.enroll_subjects,
            "imposter_subjects": args.imposter_subjects,
            "anchors_per_subject": args.anchors_per_subject,
            "min_anchors_per_subject": args.min_anchors_per_subject,
            "anchor_candidates_per_track": args.anchor_candidates_per_track,
            "max_gallery_tracks": args.max_gallery_tracks,
            "faces_per_track": args.faces_per_track,
            "calibration_query_tracks": args.calibration_query_tracks,
            "evaluation_query_tracks": args.evaluation_query_tracks,
            "calibration_imposter_gallery_tracks": (
                args.calibration_imposter_gallery_tracks
            ),
            "repeats": args.repeats,
            "seed": args.seed,
            "fmr_targets": args.fmr_targets,
        },
        "valid_pids": valid_pids,
        "common_anchor_candidate_pids": candidate_pids,
        "common_template_pids": common_template_pids,
        "anchor_manifest": anchor_manifest,
        "repetitions": repetitions,
        "aggregate": aggregate,
    }
    path = common.save_result(
        "mevid_face_b2",
        payload,
        f"r{args.repeats}_e{args.enroll_subjects}",
    )
    print(f"\n[✓] 协议 B2 结果 → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
