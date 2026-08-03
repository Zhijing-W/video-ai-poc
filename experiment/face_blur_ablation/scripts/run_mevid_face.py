"""MEVID 协议 B：固定官方 gallery 锚点，公平比较人脸识别与超分方案。"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import mevid_eval_common as common
import numpy as np


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MEVID 人脸模型实验：固定相同官方 gallery 最佳脸锚点，仅比较人脸后端和超分"
    )
    parser.add_argument("--data", required=True, help="MEVID 根目录")
    parser.add_argument("--variants", default="FR0,FR1,FR2,FR3")
    parser.add_argument("--enroll-subjects", type=int, default=27)
    parser.add_argument(
        "--imposter-subjects",
        type=int,
        default=-1,
        help="-1 表示使用建档身份之外的全部可用身份",
    )
    parser.add_argument("--anchors-per-subject", type=int, default=3)
    parser.add_argument("--anchor-candidates-per-track", type=int, default=8)
    parser.add_argument("--max-gallery-tracks", type=int, default=3)
    parser.add_argument("--faces-per-track", type=int, default=8)
    parser.add_argument("--max-query-tracks", type=int, default=4)
    parser.add_argument("--fmr-targets", type=parse_targets, default=(0.01, 0.05, 0.10))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        parser.error(f"未知 variant：{','.join(unknown)}；可选 {','.join(VARIANTS)}")
    selected_variants = {name: VARIANTS[name] for name in variants}

    from app import face as face_mod
    from app import gait as gait_mod
    from app import body_gallery as gallery_mod
    from app import body_reid as reid_mod
    from app.core.config import settings

    settings.face_superres = "gfpgan" if any(VARIANTS[name]["superres"] for name in variants) else "off"
    if any(VARIANTS[name]["backend"] == "adaface" for name in variants):
        probe = face_mod.embed_aligned_face(np.zeros((112, 112, 3), dtype=np.uint8), "adaface")
        error = face_mod.adaface_error()
        if error or probe is None or probe.shape != (512,):
            raise RuntimeError(f"AdaFace 启动检查失败：{error or 'embedding 输出非法'}")

    data_dir = Path(args.data)
    tracklets = common.load_mevid(data_dir)
    grouped = common.group_tracklets(tracklets)
    valid_pids = sorted(
        pid
        for pid, split in grouped.items()
        if split["gallery"] and split["query"]
    )
    enroll_pids, imposter_pids = common.split_subjects(
        valid_pids,
        args.enroll_subjects,
        args.imposter_subjects,
        args.seed,
    )
    print(
        f"[*] 协议 B / Face：可用身份={len(valid_pids)} "
        f"建档={len(enroll_pids)} 冒充={len(imposter_pids)}"
    )

    anchor_records = {}
    anchor_manifest = {}
    started = time.time()
    for index, pid in enumerate(enroll_pids, start=1):
        anchor_paths = common.select_gallery_face_anchors(
            grouped[pid]["gallery"][:args.max_gallery_tracks],
            face_mod,
            settings,
            args.anchor_candidates_per_track,
            args.anchors_per_subject,
        )
        anchor_manifest[pid] = [str(path.relative_to(data_dir)) for path in anchor_paths]
        pseudo_track = common.Tracklet(
            pid=pid,
            cam=-1,
            outfit=-1,
            track=-1,
            frames=anchor_paths,
            is_query=False,
        )
        anchor_records[pid] = common.extract_track_features(
            pseudo_track,
            face_mod,
            reid_mod,
            gait_mod,
            gallery_mod,
            settings,
            selected_variants,
            max(1, len(anchor_paths)),
            0,
            need_body=False,
            need_gait=False,
        )
        if index % 5 == 0 or index == len(enroll_pids):
            print(f"    anchor {index}/{len(enroll_pids)}")

    query_work = []
    for pid in enroll_pids:
        for tracklet in grouped[pid]["query"][:args.max_query_tracks]:
            query_work.append((True, pid, tracklet))
    for pid in imposter_pids:
        for tracklet in grouped[pid]["query"][:args.max_query_tracks]:
            query_work.append((False, pid, tracklet))

    query_records = []
    for index, (genuine, pid, tracklet) in enumerate(query_work, start=1):
        record = common.extract_track_features(
            tracklet,
            face_mod,
            reid_mod,
            gait_mod,
            gallery_mod,
            settings,
            selected_variants,
            args.faces_per_track,
            0,
            need_body=False,
            need_gait=False,
        )
        query_records.append((genuine, pid, record))
        if index % 10 == 0 or index == len(query_work):
            elapsed = time.time() - started
            print(f"    query {index}/{len(query_work)}  {elapsed:.0f}s")

    if any(VARIANTS[name]["backend"] == "adaface" for name in variants):
        error = face_mod.adaface_error()
        if error:
            raise RuntimeError(f"AdaFace 加载/推理失败：{error}")
    if any(VARIANTS[name]["superres"] for name in variants):
        error = face_mod.superres_error()
        if error:
            raise RuntimeError(f"GFPGAN 加载/推理失败：{error}")

    results = {}
    for name in variants:
        template_vectors = defaultdict(list)
        for pid, record in anchor_records.items():
            vector = common.aggregate_face(record["face"][name])
            if vector is not None:
                template_vectors[pid].append(vector)
        templates = common.build_mean_templates(template_vectors)

        rows = []
        enhanced_frames = 0
        detected_frames = 0
        for genuine, pid, record in query_records:
            slot = record["face"][name]
            vector = common.aggregate_face(slot)
            pred, score = common.top1(vector, templates)
            enhanced_frames += slot["enhanced"]
            detected_frames += sum(1 for frame in slot["frames"] if frame["emb"] is not None)
            rows.append(
                {
                    "gt": pid,
                    "genuine": genuine,
                    "quality_bin": slot["best_cat"],
                    "pred": pred,
                    "confidence": round(float(score), 6),
                    "enhanced_frames": slot["enhanced"],
                }
            )
        results[name] = {
            "note": VARIANTS[name]["note"],
            "coverage": {"face": len(templates), "body": 0, "gait": 0},
            "anchor_coverage": len(templates),
            "detected_probe_frames": detected_frames,
            "enhanced_probe_frames": enhanced_frames,
            "summary": common.summarize_open_set(rows, args.fmr_targets),
            "rows": rows,
        }

    common.print_operating_table(results, args.fmr_targets)
    payload = {
        "protocol": "MEVID-FACE",
        "purpose": "固定相同官方 gallery 最佳脸输入，公平比较 ArcFace/AdaFace/GFPGAN",
        "anchor_policy": (
            "严格排除 query；每个身份在前 max_gallery_tracks 条官方 gallery 轨迹中均匀抽取候选帧，"
            "按原图产品人脸质量排序后选择固定锚点"
        ),
        "config": {
            "variants": variants,
            "enroll_subjects": len(enroll_pids),
            "imposter_subjects": len(imposter_pids),
            "anchors_per_subject": args.anchors_per_subject,
            "anchor_candidates_per_track": args.anchor_candidates_per_track,
            "max_gallery_tracks": args.max_gallery_tracks,
            "faces_per_track": args.faces_per_track,
            "max_query_tracks": args.max_query_tracks,
            "fmr_targets": args.fmr_targets,
            "seed": args.seed,
        },
        "enroll_pids": enroll_pids,
        "imposter_pids": imposter_pids,
        "anchor_manifest": anchor_manifest,
        "results": results,
    }
    path = common.save_result(
        "mevid_face",
        payload,
        f"e{len(enroll_pids)}_i{len(imposter_pids)}",
    )
    print(f"\n[✓] 协议 B 结果 → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
