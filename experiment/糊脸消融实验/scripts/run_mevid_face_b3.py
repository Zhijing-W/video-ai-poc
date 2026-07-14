"""MEVID 协议 B3：使用独立 train 身份校准阈值，仅在 test 身份报告最终结果。"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import mevid_eval_common as common
from run_mevid_face_b2 import (
    VARIANTS,
    _paired_superres_stats,
    _query_rows,
    parse_targets,
)


sys.path.insert(0, str(common.ROOT))


def _extract_anchor_records(
    pids,
    grouped,
    manifests,
    face_mod,
    reid_mod,
    gait_mod,
    gallery_mod,
    settings,
    variants,
    label,
):
    records = {}
    for index, pid in enumerate(pids, start=1):
        paths = [item["path"] for item in manifests[pid]]
        pseudo_track = common.Tracklet(
            pid=pid,
            cam=-1,
            outfit=-1,
            track=-1,
            frames=paths,
            is_query=False,
        )
        records[pid] = common.extract_track_features(
            pseudo_track,
            face_mod,
            reid_mod,
            gait_mod,
            gallery_mod,
            settings,
            variants,
            len(paths),
            0,
            need_body=False,
            need_gait=False,
        )
        if index % 10 == 0 or index == len(pids):
            print(f"    {label} anchor embed {index}/{len(pids)}")
    return records


def _scan_anchor_manifests(
    pids,
    grouped,
    face_mod,
    settings,
    candidates_per_track,
    anchors_per_subject,
    max_tracks,
    label,
):
    manifests = {}
    started = time.time()
    for index, pid in enumerate(pids, start=1):
        manifests[pid] = common.select_diverse_gallery_face_anchors(
            grouped[pid]["gallery"][:max_tracks],
            face_mod,
            settings,
            candidates_per_track,
            anchors_per_subject,
        )
        if index % 10 == 0 or index == len(pids):
            print(f"    {label} anchor scan {index}/{len(pids)}")
    print(f"[*] {label} anchor scan 耗时={time.time() - started:.0f}s")
    return manifests


def _common_anchor_vectors(records, variants):
    vectors = {name: {} for name in variants}
    common_pids = []
    for pid, record in records.items():
        per_variant = {
            name: common.aggregate_face(record["face"][name])
            for name in variants
        }
        if all(vector is not None for vector in per_variant.values()):
            common_pids.append(pid)
            for name, vector in per_variant.items():
                vectors[name][pid] = vector
    return common_pids, vectors


def _extract_track_records(
    work,
    face_mod,
    reid_mod,
    gait_mod,
    gallery_mod,
    settings,
    variants,
    faces_per_track,
    label,
):
    output = []
    started = time.time()
    for index, (pid, tracklet) in enumerate(work, start=1):
        record = common.extract_track_features(
            tracklet,
            face_mod,
            reid_mod,
            gait_mod,
            gallery_mod,
            settings,
            variants,
            faces_per_track,
            0,
            need_body=False,
            need_gait=False,
        )
        output.append((pid, tracklet.track, record))
        if index % 20 == 0 or index == len(work):
            print(
                f"    {label} {index}/{len(work)}  "
                f"{time.time() - started:.0f}s"
            )
    return output


def _aggregate_results(repetitions, variants, targets):
    aggregate = {}
    for name in variants:
        rank1_values = [
            repeat["results"][name]["evaluation"]["rank1_rate"]
            for repeat in repetitions
        ]
        operating = {}
        for target in targets:
            key = f"{target:.3f}"
            operating[key] = {
                "tpir": common.mean_ci95(
                    [
                        repeat["results"][name]["evaluation"]["operating_points"][key]["tpir"]
                        for repeat in repetitions
                    ]
                ),
                "actual_fmr": common.mean_ci95(
                    [
                        repeat["results"][name]["evaluation"]["operating_points"][key]["fmr"]
                        for repeat in repetitions
                    ]
                ),
            }
        tags = {}
        for tag in common.CAUSE_TAG_ORDER:
            rank_values = [
                repeat["results"][name]["evaluation"]["by_degradation_tag"][tag]["rank1_rate"]
                for repeat in repetitions
                if repeat["results"][name]["evaluation"]["by_degradation_tag"][tag]["rank1_rate"]
                is not None
            ]
            tag_result = {
                "mean_tracks_per_repeat": round(
                    sum(
                        repeat["results"][name]["evaluation"]["by_degradation_tag"][tag]["total"]
                        for repeat in repetitions
                    )
                    / len(repetitions),
                    2,
                ),
                "rank1": common.mean_ci95(rank_values),
                "operating_points": {},
            }
            for target in targets:
                key = f"{target:.3f}"
                values = [
                    repeat["results"][name]["evaluation"]["by_degradation_tag"][tag][
                        f"tpir_at_{key}"
                    ]
                    for repeat in repetitions
                    if repeat["results"][name]["evaluation"]["by_degradation_tag"][tag][
                        f"tpir_at_{key}"
                    ]
                    is not None
                ]
                tag_result["operating_points"][key] = common.mean_ci95(values)
            tags[tag] = tag_result
        aggregate[name] = {
            "note": VARIANTS[name]["note"],
            "rank1": common.mean_ci95(rank1_values),
            "operating_points": operating,
            "by_degradation_tag": tags,
            "superres_attempted_probe_frames_total": sum(
                repeat["results"][name]["superres_attempted_probe_frames"]
                for repeat in repetitions
            ),
            "enhanced_probe_frames_total": sum(
                repeat["results"][name]["enhanced_probe_frames"]
                for repeat in repetitions
            ),
        }
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MEVID B3：train身份独立校准阈值，test身份仅用于最终评测"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--variants", default="FR0,FR1,FR2,FR3")
    parser.add_argument("--anchors-per-subject", type=int, default=3)
    parser.add_argument("--anchor-candidates-per-track", type=int, default=8)
    parser.add_argument("--max-anchor-tracks", type=int, default=5)
    parser.add_argument("--faces-per-track", type=int, default=8)
    parser.add_argument("--train-known-subjects", type=int, default=50)
    parser.add_argument("--train-genuine-tracks", type=int, default=2)
    parser.add_argument("--train-imposter-tracks", type=int, default=4)
    parser.add_argument("--test-enroll-subjects", type=int, default=27)
    parser.add_argument("--test-imposter-subjects", type=int, default=-1)
    parser.add_argument("--test-query-tracks", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fmr-targets", type=parse_targets, default=(0.01, 0.05, 0.10))
    args = parser.parse_args()

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        parser.error(f"未知 variant：{','.join(unknown)}")
    extraction_variants = {name: VARIANTS[name] for name in variants}
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
    train_tracklets = common.load_mevid_train(data_dir)
    test_tracklets = common.load_mevid(data_dir)
    train_grouped = common.group_tracklets(train_tracklets)
    test_grouped = common.group_tracklets(test_tracklets)
    train_pids = sorted(train_grouped)
    test_pids = sorted(
        pid
        for pid, split in test_grouped.items()
        if split["gallery"] and split["query"]
    )
    overlap = sorted(set(train_pids) & set(test_pids))
    if overlap:
        raise RuntimeError(f"MEVID train/test 身份不独立：{overlap[:10]}")
    print(
        f"[*] MEVID train={len(train_pids)} identities/{len(train_tracklets)} tracks，"
        f"test={len(test_pids)} identities/{len(test_tracklets)} tracks"
    )

    train_manifests = _scan_anchor_manifests(
        train_pids,
        train_grouped,
        face_mod,
        settings,
        args.anchor_candidates_per_track,
        args.anchors_per_subject,
        args.max_anchor_tracks,
        "train",
    )
    train_anchor_candidates = [
        pid for pid in train_pids if train_manifests[pid]
    ]
    train_anchor_records = _extract_anchor_records(
        train_anchor_candidates,
        train_grouped,
        train_manifests,
        face_mod,
        reid_mod,
        gait_mod,
        gallery_mod,
        settings,
        extraction_variants,
        "train",
    )
    train_common_pids, train_anchor_vectors = _common_anchor_vectors(
        train_anchor_records,
        variants,
    )
    if len(train_common_pids) < args.train_known_subjects:
        raise RuntimeError(
            f"train共同模板身份仅{len(train_common_pids)}，"
            f"不足{args.train_known_subjects}"
        )

    train_known_candidates = []
    for pid in train_common_pids:
        anchor_tracks = {item["track"] for item in train_manifests[pid]}
        independent_tracks = [
            tracklet
            for tracklet in train_grouped[pid]["gallery"]
            if tracklet.track not in anchor_tracks
        ]
        if len(independent_tracks) >= args.train_genuine_tracks:
            train_known_candidates.append(pid)
    if len(train_known_candidates) < args.train_known_subjects:
        raise RuntimeError(
            f"仅{len(train_known_candidates)}个train身份同时满足共同模板和"
            f"{args.train_genuine_tracks}条独立校准轨迹，不足{args.train_known_subjects}"
        )

    shuffled_train = sorted(train_known_candidates)
    random.Random(args.seed).shuffle(shuffled_train)
    train_known_pids = shuffled_train[:args.train_known_subjects]
    train_imposter_pids = [
        pid for pid in train_pids if pid not in set(train_known_pids)
    ]
    print(
        f"[*] train校准：known={len(train_known_pids)}，"
        f"imposter={len(train_imposter_pids)}"
    )

    train_genuine_work = []
    for pid in train_known_pids:
        anchor_tracks = {item["track"] for item in train_manifests[pid]}
        candidates = [
            tracklet
            for tracklet in train_grouped[pid]["gallery"]
            if tracklet.track not in anchor_tracks
        ]
        if len(candidates) < args.train_genuine_tracks:
            raise RuntimeError(f"train身份{pid}缺少独立genuine校准轨迹")
        train_genuine_work.extend(
            (pid, tracklet)
            for tracklet in candidates[:args.train_genuine_tracks]
        )
    train_imposter_work = []
    for pid in train_imposter_pids:
        candidates = train_grouped[pid]["gallery"]
        if not candidates:
            continue
        train_imposter_work.extend(
            (pid, tracklet)
            for tracklet in candidates[:args.train_imposter_tracks]
        )

    train_genuine_records = _extract_track_records(
        train_genuine_work,
        face_mod,
        reid_mod,
        gait_mod,
        gallery_mod,
        settings,
        extraction_variants,
        args.faces_per_track,
        "train genuine",
    )
    train_imposter_records = _extract_track_records(
        train_imposter_work,
        face_mod,
        reid_mod,
        gait_mod,
        gallery_mod,
        settings,
        extraction_variants,
        args.faces_per_track,
        "train imposter",
    )

    calibrated_thresholds = {}
    train_calibration = {}
    for name in variants:
        templates = {
            pid: train_anchor_vectors[name][pid]
            for pid in train_known_pids
        }
        rows = _query_rows(
            train_genuine_records + train_imposter_records,
            templates,
            name,
            quality_variant,
        )
        scorable_rows = [row for row in rows if row["pred"] is not None]
        scorable_genuine = [row for row in scorable_rows if row["genuine"]]
        scorable_imposters = [row for row in scorable_rows if not row["genuine"]]
        if not scorable_genuine or not scorable_imposters:
            raise RuntimeError(
                f"{name} train校准缺少可评分样本："
                f"genuine={len(scorable_genuine)} imposter={len(scorable_imposters)}"
            )
        curve = common.det_curve(scorable_rows)
        points = {
            f"{target:.3f}": common.select_operating_point(curve, target)
            for target in args.fmr_targets
        }
        calibrated_thresholds[name] = {
            key: point["threshold"] for key, point in points.items()
        }
        train_calibration[name] = {
            "known_subjects": len(train_known_pids),
            "imposter_subjects": len(train_imposter_pids),
            "genuine_tracks": sum(row["genuine"] for row in rows),
            "imposter_tracks": sum(not row["genuine"] for row in rows),
            "scorable_genuine_tracks": sum(
                row["genuine"] for row in scorable_rows
            ),
            "scorable_imposter_tracks": sum(
                not row["genuine"] for row in scorable_rows
            ),
            "operating_points": points,
            "det": curve,
        }
        print(f"[*] {name} train阈值={calibrated_thresholds[name]}")

    test_manifests = _scan_anchor_manifests(
        test_pids,
        test_grouped,
        face_mod,
        settings,
        args.anchor_candidates_per_track,
        args.anchors_per_subject,
        args.max_anchor_tracks,
        "test",
    )
    test_anchor_candidates = [
        pid for pid in test_pids if test_manifests[pid]
    ]
    test_anchor_records = _extract_anchor_records(
        test_anchor_candidates,
        test_grouped,
        test_manifests,
        face_mod,
        reid_mod,
        gait_mod,
        gallery_mod,
        settings,
        extraction_variants,
        "test",
    )
    test_common_pids, test_anchor_vectors = _common_anchor_vectors(
        test_anchor_records,
        variants,
    )
    if len(test_common_pids) < args.test_enroll_subjects:
        raise RuntimeError(
            f"test共同模板身份仅{len(test_common_pids)}，"
            f"不足{args.test_enroll_subjects}"
        )
    print(f"[*] test共同模板身份={len(test_common_pids)}/{len(test_pids)}")

    test_query_work = [
        (pid, tracklet)
        for pid in test_pids
        for tracklet in test_grouped[pid]["query"][:args.test_query_tracks]
    ]
    test_query_records = _extract_track_records(
        test_query_work,
        face_mod,
        reid_mod,
        gait_mod,
        gallery_mod,
        settings,
        extraction_variants,
        args.faces_per_track,
        "test query",
    )
    test_records_by_pid = defaultdict(list)
    for record in test_query_records:
        test_records_by_pid[record[0]].append(record)

    repetitions = []
    for repeat in range(args.repeats):
        split_seed = args.seed + repeat
        enroll_pool = sorted(test_common_pids)
        random.Random(split_seed).shuffle(enroll_pool)
        enroll_pids = enroll_pool[:args.test_enroll_subjects]
        remaining = [pid for pid in test_pids if pid not in set(enroll_pids)]
        random.Random(split_seed + 100_000).shuffle(remaining)
        imposter_count = (
            len(remaining)
            if args.test_imposter_subjects < 0
            else args.test_imposter_subjects
        )
        imposter_pids = remaining[:imposter_count]
        eval_records = [
            record
            for pid in enroll_pids + imposter_pids
            for record in test_records_by_pid[pid]
        ]

        results = {}
        rows_by_variant = {}
        for name in variants:
            templates = {
                pid: test_anchor_vectors[name][pid]
                for pid in enroll_pids
            }
            rows = _query_rows(
                eval_records,
                templates,
                name,
                quality_variant,
            )
            rows_by_variant[name] = rows
            scorable_rows = [row for row in rows if row["pred"] is not None]
            results[name] = {
                "note": VARIANTS[name]["note"],
                "anchor_coverage": len(templates),
                "evaluation": common.summarize_with_thresholds(
                    scorable_rows,
                    calibrated_thresholds[name],
                ),
                "evaluation_all_tracks": common.summarize_with_thresholds(
                    rows,
                    calibrated_thresholds[name],
                ),
                "face_query_coverage": {
                    "genuine_total": sum(row["genuine"] for row in rows),
                    "genuine_scorable": sum(
                        row["genuine"] for row in scorable_rows
                    ),
                    "imposter_total": sum(not row["genuine"] for row in rows),
                    "imposter_scorable": sum(
                        not row["genuine"] for row in scorable_rows
                    ),
                },
                "superres_attempted_probe_frames": sum(
                    row["superres_attempted_frames"] for row in rows
                ),
                "enhanced_probe_frames": sum(
                    row["enhanced_frames"] for row in rows
                ),
                "rows": rows,
            }

        paired = {}
        if "FR0" in rows_by_variant and "FR2" in rows_by_variant:
            paired["FR2_vs_FR0"] = _paired_superres_stats(
                rows_by_variant["FR0"],
                rows_by_variant["FR2"],
            )
        if "FR1" in rows_by_variant and "FR3" in rows_by_variant:
            paired["FR3_vs_FR1"] = _paired_superres_stats(
                rows_by_variant["FR1"],
                rows_by_variant["FR3"],
            )
        repetitions.append(
            {
                "repeat": repeat,
                "seed": split_seed,
                "enroll_pids": enroll_pids,
                "imposter_pids": imposter_pids,
                "results": results,
                "superres_paired": paired,
            }
        )
        print(f"[*] test repeat {repeat + 1}/{args.repeats} 完成")

    aggregate = _aggregate_results(
        repetitions,
        variants,
        args.fmr_targets,
    )
    print("\n================ MEVID Face B3 · train-calibrated ================")
    header = (
        f"{'variant':<8}{'Rank-1 mean':>14}"
        + "".join(f"{'TPIR@'+str(target):>14}" for target in args.fmr_targets)
    )
    print(header)
    for name, result in aggregate.items():
        rank_mean = result["rank1"]["mean"]
        rank_text = "-" if rank_mean is None else f"{100 * rank_mean:.1f}%"
        row = f"{name:<8}{rank_text:>14}"
        for target in args.fmr_targets:
            mean = result["operating_points"][f"{target:.3f}"]["tpir"]["mean"]
            text = "-" if mean is None else f"{100 * mean:.1f}%"
            row += f"{text:>14}"
        print(row)

    def serialize_manifests(manifests):
        return {
            pid: [
                {
                    **{key: value for key, value in item.items() if key not in {"path", "rank"}},
                    "path": str(item["path"].relative_to(data_dir)),
                }
                for item in items
            ]
            for pid, items in manifests.items()
        }

    payload = {
        "protocol": "MEVID-FACE-B3",
        "purpose": "独立train身份校准模型阈值；test身份只用于最终评测",
        "config": vars(args),
        "train": {
            "identities": len(train_pids),
            "tracklets": len(train_tracklets),
            "known_pids": train_known_pids,
            "imposter_pids": train_imposter_pids,
            "common_template_pids": train_common_pids,
            "anchor_manifest": serialize_manifests(train_manifests),
            "calibration": train_calibration,
            "thresholds": calibrated_thresholds,
        },
        "test": {
            "identities": len(test_pids),
            "tracklets": len(test_tracklets),
            "common_template_pids": test_common_pids,
            "anchor_manifest": serialize_manifests(test_manifests),
            "repetitions": repetitions,
            "aggregate": aggregate,
        },
    }
    path = common.save_result(
        "mevid_face_b3",
        payload,
        f"train{len(train_known_pids)}_r{args.repeats}_test{args.test_enroll_subjects}",
    )
    print(f"\n[✓] 协议 B3 结果 → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
