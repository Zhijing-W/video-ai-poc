"""MEVID 协议 A：严格产品门控下的人脸/人形/步态端到端消融。"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import mevid_eval_common as common


sys.path.insert(0, str(common.ROOT))

ARMS = {
    "F": ("face",),
    "B": ("body",),
    "G": ("gait",),
    "FB": ("face", "body"),
    "BG": ("body", "gait"),
    "FBG": ("face", "body", "gait"),
}


def parse_targets(value: str) -> tuple[float, ...]:
    targets = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not targets or any(item < 0 or item > 1 for item in targets):
        raise argparse.ArgumentTypeError("FMR targets 必须是 0..1 的逗号分隔小数")
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MEVID 产品逻辑对齐实验：严格 monitor 建档门控，评估模态救回与误识"
    )
    parser.add_argument("--data", required=True, help="MEVID 根目录")
    parser.add_argument("--arms", default="F,B,G,FB,BG,FBG")
    parser.add_argument("--enroll-subjects", type=int, default=27)
    parser.add_argument(
        "--imposter-subjects",
        type=int,
        default=-1,
        help="-1 表示使用建档身份之外的全部可用身份",
    )
    parser.add_argument("--faces-per-track", type=int, default=8)
    parser.add_argument("--gait-frames", type=int, default=24)
    parser.add_argument("--max-gallery-tracks", type=int, default=3)
    parser.add_argument("--max-query-tracks", type=int, default=4)
    parser.add_argument("--fmr-targets", type=parse_targets, default=(0.01, 0.05, 0.10))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        parser.error(f"未知 arm：{','.join(unknown)}；可选 {','.join(ARMS)}")

    from app import face as face_mod
    from app import gait as gait_mod
    from app import body_gallery as gallery_mod
    from app import body_reid as reid_mod
    from app.core.config import settings

    settings.face_rec_backend = "arcface"
    settings.face_superres = "off"

    data_dir = Path(args.data)
    tracklets = common.load_mevid(data_dir)
    grouped = common.group_tracklets(tracklets)
    valid_pids = sorted(
        pid for pid, split in grouped.items() if split["gallery"] and split["query"]
    )
    enroll_pids, imposter_pids = common.split_subjects(
        valid_pids,
        args.enroll_subjects,
        args.imposter_subjects,
        args.seed,
    )
    print(
        f"[*] 协议 A / E2E：轨迹={len(tracklets)} 可用身份={len(valid_pids)} "
        f"建档={len(enroll_pids)} 冒充={len(imposter_pids)}"
    )

    work = []
    for pid in enroll_pids:
        for tracklet in grouped[pid]["gallery"][:args.max_gallery_tracks]:
            work.append(("enroll_gallery", pid, tracklet))
        for tracklet in grouped[pid]["query"][:args.max_query_tracks]:
            work.append(("genuine_query", pid, tracklet))
    for pid in imposter_pids:
        for tracklet in grouped[pid]["query"][:args.max_query_tracks]:
            work.append(("imposter_query", pid, tracklet))

    need_body = any("body" in ARMS[arm] for arm in arms)
    need_gait = any("gait" in ARMS[arm] for arm in arms)
    face_variants = {"arcface": {"backend": "arcface", "superres": False}}
    extracted = []
    started = time.time()
    for index, (role, pid, tracklet) in enumerate(work, start=1):
        record = common.extract_track_features(
            tracklet,
            face_mod,
            reid_mod,
            gait_mod,
            gallery_mod,
            settings,
            face_variants,
            args.faces_per_track,
            args.gait_frames,
            need_body=need_body,
            need_gait=need_gait,
        )
        extracted.append((role, pid, record))
        if index % 10 == 0 or index == len(work):
            elapsed = time.time() - started
            print(f"    {index}/{len(work)}  {elapsed:.0f}s  {index / max(elapsed, 1e-6):.2f} track/s")

    gallery_records = defaultdict(list)
    query_records = []
    for role, pid, record in extracted:
        if role == "enroll_gallery":
            gallery_records[pid].append(record)
        else:
            query_records.append((role == "genuine_query", pid, record))

    face_vectors = defaultdict(list)
    body_vectors = defaultdict(list)
    gait_vectors = defaultdict(list)
    for pid, records in gallery_records.items():
        for record in records:
            face_vector = common.aggregate_face(record["face"]["arcface"], {"clear"})
            if face_vector is not None:
                face_vectors[pid].append(face_vector)
            if record["body_enroll"] is not None:
                body_vectors[pid].append(record["body_enroll"])
            if record["gait"] is not None:
                gait_vectors[pid].append(record["gait"])

    templates = {
        "face": common.build_mean_templates(face_vectors),
        "body": common.build_mean_templates(body_vectors),
        "gait": common.build_mean_templates(gait_vectors),
    }
    full_coverage = {
        "face": len(templates["face"]),
        "body": len(templates["body"]),
        "gait": len(templates["gait"]),
        "total": len(enroll_pids),
    }
    print(
        "[*] 严格建档覆盖 face/body/gait="
        f"{full_coverage['face']}/{full_coverage['body']}/{full_coverage['gait']}"
    )

    weights = {
        "face": settings.identity_w_face,
        "body": settings.identity_w_body,
        "gait": settings.identity_w_gait,
    }
    results = {}
    for arm in arms:
        rows = []
        routes = ARMS[arm]
        for genuine, pid, record in query_records:
            face_slot = record["face"]["arcface"]
            face_vector = common.aggregate_face(face_slot)
            face_quality = float(face_slot["best_q"] or 0.0)
            face_weight = weights["face"] * (
                settings.identity_face_quality_floor
                + (1.0 - settings.identity_face_quality_floor) * face_quality
            )
            route_inputs = {}
            if "face" in routes:
                route_inputs["face"] = (face_vector, templates["face"], face_weight)
            if "body" in routes:
                route_inputs["body"] = (record["body"], templates["body"], weights["body"])
            if "gait" in routes:
                route_inputs["gait"] = (record["gait"], templates["gait"], weights["gait"])
            fused = common.fuse_route_votes(route_inputs, settings.identity_agree_bonus)
            rows.append(
                {
                    "gt": pid,
                    "genuine": genuine,
                    "quality_bin": face_slot["best_cat"],
                    **fused,
                }
            )

        coverage = {
            route: full_coverage[route] if route in routes else 0
            for route in ("face", "body", "gait")
        }
        results[arm] = {
            "routes": list(routes),
            "coverage": coverage,
            "summary": common.summarize_open_set(rows, args.fmr_targets),
            "rows": rows,
        }

    common.print_operating_table(results, args.fmr_targets)
    payload = {
        "protocol": "MEVID-E2E",
        "purpose": "严格 monitor 建档门控下，评估各模态的覆盖、救回和开放集误识",
        "identity_space": (
            "MEVID ground-truth person_id；face/body/gait 模板都以同一 canonical pid 为键，"
            "不是比较各模态 gallery 的本地自增 ID"
        ),
        "config": {
            "arms": arms,
            "enroll_subjects": len(enroll_pids),
            "imposter_subjects": len(imposter_pids),
            "faces_per_track": args.faces_per_track,
            "gait_frames": args.gait_frames,
            "max_gallery_tracks": args.max_gallery_tracks,
            "max_query_tracks": args.max_query_tracks,
            "fmr_targets": args.fmr_targets,
            "seed": args.seed,
        },
        "enroll_pids": enroll_pids,
        "imposter_pids": imposter_pids,
        "results": results,
    }
    path = common.save_result(
        "mevid_e2e",
        payload,
        f"e{len(enroll_pids)}_i{len(imposter_pids)}",
    )
    print(f"\n[✓] 协议 A 结果 → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
