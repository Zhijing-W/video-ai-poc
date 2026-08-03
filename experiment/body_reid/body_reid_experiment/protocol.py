from __future__ import annotations

import argparse
import datetime as dt
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .common import (
    EXPERIMENT_DIR,
    GALLERY_ROLES,
    SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_text,
    bytes_sha256,
    file_sha256,
    load_mevid_common,
    protocol_identity,
    serialize_samples,
    stable_hash,
)


@dataclass(frozen=True)
class FreezeConfig:
    seed: int = 0
    enroll_subjects: int = 27
    imposter_subjects: int = 25
    frames_per_track: int = 8
    max_gallery_tracks: int = 3
    max_query_tracks: int = 4
    calibration_enroll_subjects: int = 50
    calibration_imposter_subjects: int = -1
    calibration_gallery_tracks: int = 3
    calibration_query_tracks: int = 4
    expected_test_identities: int = 52

    def validate(self) -> None:
        positive = {
            "enroll_subjects": self.enroll_subjects,
            "imposter_subjects": self.imposter_subjects,
            "frames_per_track": self.frames_per_track,
            "max_gallery_tracks": self.max_gallery_tracks,
            "max_query_tracks": self.max_query_tracks,
            "calibration_enroll_subjects": self.calibration_enroll_subjects,
            "calibration_gallery_tracks": self.calibration_gallery_tracks,
            "calibration_query_tracks": self.calibration_query_tracks,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError("以下参数必须大于0：" + ", ".join(invalid))
        if self.calibration_imposter_subjects == 0:
            raise ValueError("calibration_imposter_subjects不能为0")
        if self.expected_test_identities < 0:
            raise ValueError("expected_test_identities不能小于0")


def _annotation_hashes(data_root: Path) -> dict[str, str]:
    annotation = data_root / "annotation" / "mevid-v1-annotation-data"
    names = (
        "test_name.txt",
        "track_test_info.txt",
        "query_IDX.txt",
        "train_name.txt",
        "track_train_info.txt",
    )
    missing = [name for name in names if not (annotation / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "MEVID标注文件缺失：" + ", ".join(str(annotation / name) for name in missing)
        )
    return {name: file_sha256(annotation / name) for name in names}


def _relative_source(path: Path, data_root: Path) -> str:
    source = path.resolve()
    root = data_root.resolve()
    try:
        return source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"MEVID样本不在数据根目录内：{source}") from exc


def _sample_tracklet(tracklet, count: int, mevid_common) -> list[tuple[int, Path]]:
    indexed = list(enumerate(tracklet.frames))
    return mevid_common.sample_evenly(indexed, count)


def _sample_row(
    *,
    split: str,
    role: str,
    tracklet,
    frame_position: int,
    sample_position: int,
    source: Path,
    data_root: Path,
    body_reid,
    body_gallery,
) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"冻结样本不存在：{source}")
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        image.load()
    quality = body_reid.assess_quality(image)
    if role in GALLERY_ROLES:
        gallery_allowed, rejection = body_gallery.quality_ok(quality)
    else:
        gallery_allowed, rejection = False, "not_gallery_role"
    sample_id = (
        f"{split}:t{int(tracklet.track):05d}:f{int(frame_position):05d}"
    )
    return {
        "split": split,
        "sample_id": sample_id,
        "person_id": tracklet.pid,
        "role": role,
        "official_track_index": str(int(tracklet.track)),
        "camera_id": str(int(tracklet.cam)),
        "outfit_id": str(int(tracklet.outfit)),
        "source_relpath": _relative_source(source, data_root),
        "source_sha256": file_sha256(source),
        "frame_position": str(int(frame_position)),
        "sample_position": str(int(sample_position)),
        "width": str(int(quality["width"])),
        "height": str(int(quality["height"])),
        "area": str(int(quality["area"])),
        "blur_var": f"{float(quality['blur_var']):.6f}",
        "aspect_ratio": f"{float(quality['aspect_ratio']):.6f}",
        "gallery_allowed": "true" if gallery_allowed else "false",
        "gallery_rejection_reason": rejection or "",
    }


def _append_tracklet_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    role: str,
    tracklet,
    frames_per_track: int,
    data_root: Path,
    mevid_common,
    body_reid,
    body_gallery,
) -> None:
    selected = _sample_tracklet(tracklet, frames_per_track, mevid_common)
    if not selected:
        raise ValueError(f"轨迹没有可冻结帧：split={split}, track={tracklet.track}")
    for sample_position, (frame_position, source) in enumerate(selected):
        rows.append(
            _sample_row(
                split=split,
                role=role,
                tracklet=tracklet,
                frame_position=frame_position,
                sample_position=sample_position,
                source=source,
                data_root=data_root,
                body_reid=body_reid,
                body_gallery=body_gallery,
            )
        )


def _select_test_tracklets(tracklets: list, config: FreezeConfig, mevid_common):
    grouped = mevid_common.group_tracklets(tracklets)
    candidates = sorted(
        pid
        for pid, split in grouped.items()
        if split["gallery"] and split["query"]
    )
    if (
        config.expected_test_identities
        and len(candidates) != config.expected_test_identities
    ):
        raise ValueError(
            "MEVID test可用身份数不符合冻结预期："
            f"expected={config.expected_test_identities}, actual={len(candidates)}"
        )
    enroll, imposters = mevid_common.split_subjects(
        candidates,
        config.enroll_subjects,
        config.imposter_subjects,
        config.seed,
    )
    selected: list[tuple[str, Any]] = []
    for pid in enroll:
        gallery = sorted(grouped[pid]["gallery"], key=lambda item: item.track)
        queries = sorted(grouped[pid]["query"], key=lambda item: item.track)
        selected.extend(
            ("enroll_gallery", item)
            for item in gallery[: config.max_gallery_tracks]
        )
        selected.extend(
            ("genuine_query", item)
            for item in queries[: config.max_query_tracks]
        )
    for pid in imposters:
        queries = sorted(grouped[pid]["query"], key=lambda item: item.track)
        selected.extend(
            ("imposter_query", item)
            for item in queries[: config.max_query_tracks]
        )
    return candidates, enroll, imposters, selected


def _select_calibration_tracklets(
    tracklets: list,
    config: FreezeConfig,
):
    grouped: dict[str, list] = defaultdict(list)
    for tracklet in tracklets:
        grouped[tracklet.pid].append(tracklet)
    eligible_enroll = sorted(
        pid for pid, items in grouped.items() if len(items) >= 2
    )
    random.Random(config.seed).shuffle(eligible_enroll)
    if config.calibration_enroll_subjects >= len(eligible_enroll):
        raise ValueError(
            "校准建档身份数必须小于至少有2条轨迹的train身份数："
            f"eligible={len(eligible_enroll)}, "
            f"requested={config.calibration_enroll_subjects}"
        )
    enroll = eligible_enroll[: config.calibration_enroll_subjects]
    imposter_candidates = sorted(set(grouped) - set(enroll))
    random.Random(config.seed + 1).shuffle(imposter_candidates)
    imposter_count = (
        len(imposter_candidates)
        if config.calibration_imposter_subjects < 0
        else config.calibration_imposter_subjects
    )
    if imposter_count <= 0 or imposter_count > len(imposter_candidates):
        raise ValueError(
            "校准陌生身份数非法："
            f"available={len(imposter_candidates)}, requested={imposter_count}"
        )
    imposters = imposter_candidates[:imposter_count]

    selected: list[tuple[str, Any]] = []
    for pid in enroll:
        items = sorted(grouped[pid], key=lambda item: item.track)
        gallery_count = min(config.calibration_gallery_tracks, len(items) - 1)
        gallery = items[:gallery_count]
        queries = items[
            gallery_count : gallery_count + config.calibration_query_tracks
        ]
        if not gallery or not queries:
            raise ValueError(f"校准身份{pid}无法形成独立Gallery/Query")
        selected.extend(("calibration_gallery", item) for item in gallery)
        selected.extend(
            ("calibration_genuine_query", item) for item in queries
        )
    for pid in imposters:
        items = sorted(grouped[pid], key=lambda item: item.track)
        selected.extend(
            ("calibration_imposter_query", item)
            for item in items[: config.calibration_query_tracks]
        )
    return enroll, imposters, selected


def _assert_no_leakage(selected: list[tuple[str, Any]]) -> None:
    gallery_paths = {
        path.resolve()
        for role, tracklet in selected
        if role in GALLERY_ROLES
        for path in tracklet.frames
    }
    query_paths = {
        path.resolve()
        for role, tracklet in selected
        if role not in GALLERY_ROLES
        for path in tracklet.frames
    }
    overlap = gallery_paths & query_paths
    if overlap:
        preview = ", ".join(str(path) for path in sorted(overlap)[:3])
        raise ValueError(f"Gallery与Query存在图片路径泄漏：{preview}")


def _diagnostic_subsets(
    rows: list[dict[str, Any]],
    selected_test: list[tuple[str, Any]],
) -> tuple[list[int], list[int]]:
    accepted_tracks = {
        int(row["official_track_index"])
        for row in rows
        if row["role"] == "enroll_gallery"
        and row["gallery_allowed"] == "true"
    }
    gallery_by_pid: dict[str, list] = defaultdict(list)
    genuine = []
    for role, tracklet in selected_test:
        if role == "enroll_gallery" and tracklet.track in accepted_tracks:
            gallery_by_pid[tracklet.pid].append(tracklet)
        elif role == "genuine_query":
            genuine.append(tracklet)

    p2 = []
    p3 = []
    for query in genuine:
        gallery = gallery_by_pid.get(query.pid, [])
        if not gallery:
            continue
        gallery_outfits = {item.outfit for item in gallery}
        if query.outfit not in gallery_outfits:
            p2.append(int(query.track))
            continue
        same_outfit_cameras = {
            item.cam for item in gallery if item.outfit == query.outfit
        }
        if query.cam not in same_outfit_cameras:
            p3.append(int(query.track))
    return sorted(p2), sorted(p3)


def freeze_protocol(
    data_root: Path,
    manifest_path: Path,
    protocol_path: Path,
    config: FreezeConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    config.validate()
    data_root = data_root.resolve()
    manifest_path = manifest_path.resolve()
    protocol_path = protocol_path.resolve()
    existing = [path for path in (manifest_path, protocol_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "冻结文件已存在，拒绝覆盖；确认重建协议时显式使用--force："
            + ", ".join(str(path) for path in existing)
        )

    from app import body_gallery, body_reid
    from app.core.config import settings

    mevid_common = load_mevid_common()
    annotation_hashes = _annotation_hashes(data_root)
    test_tracklets = mevid_common.load_mevid(data_root)
    train_tracklets = mevid_common.load_mevid_train(data_root)
    test_pids = {item.pid for item in test_tracklets}
    train_pids = {item.pid for item in train_tracklets}
    overlap = test_pids & train_pids
    if overlap:
        raise ValueError(
            "MEVID train/test身份不独立：" + ", ".join(sorted(overlap))
        )

    candidates, enroll, imposters, selected_test = _select_test_tracklets(
        test_tracklets,
        config,
        mevid_common,
    )
    calibration_enroll, calibration_imposters, selected_calibration = (
        _select_calibration_tracklets(train_tracklets, config)
    )
    selected = [*selected_test, *selected_calibration]
    _assert_no_leakage(selected)

    rows: list[dict[str, Any]] = []
    for split, selected_rows in (
        ("test", selected_test),
        ("train_calibration", selected_calibration),
    ):
        for role, tracklet in selected_rows:
            _append_tracklet_rows(
                rows,
                split=split,
                role=role,
                tracklet=tracklet,
                frames_per_track=config.frames_per_track,
                data_root=data_root,
                mevid_common=mevid_common,
                body_reid=body_reid,
                body_gallery=body_gallery,
            )

    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("冻结样本产生重复sample_id")
    p2_tracks, p3_tracks = _diagnostic_subsets(rows, selected_test)
    csv_text = serialize_samples(rows)
    csv_bytes = csv_text.encode("utf-8")
    manifest_hash = bytes_sha256(csv_bytes)
    quality_gate = {
        "implementation": "app.body_reid.assess_quality + app.body_gallery.quality_ok",
        "reid_min_area": settings.reid_min_area,
        "reid_min_blur_var": settings.reid_min_blur_var,
        "reid_min_aspect": settings.reid_min_aspect,
        "reid_max_aspect": settings.reid_max_aspect,
        "query_frames_are_never_dropped": True,
    }
    role_counts = {
        role: sum(row["role"] == role for row in rows)
        for role in sorted({row["role"] for row in rows})
    }
    gallery_covered = sorted(
        {
            row["person_id"]
            for row in rows
            if row["role"] == "enroll_gallery"
            and row["gallery_allowed"] == "true"
        }
    )
    calibration_gallery_covered = sorted(
        {
            row["person_id"]
            for row in rows
            if row["role"] == "calibration_gallery"
            and row["gallery_allowed"] == "true"
        }
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mevid_body_reid_protocol",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": {
            "name": "MEVID",
            "root_hint": data_root.name,
            "annotation_sha256": annotation_hashes,
            "train_test_person_id_overlap": [],
        },
        "config": asdict(config),
        "sampling": {
            "method": "existing MEVID common.sample_evenly floor(i * length / count)",
            "selection_uses_model_output": False,
        },
        "quality_gate": quality_gate,
        "sample_manifest": {
            "path": manifest_path.name,
            "sha256": manifest_hash,
            "rows": len(rows),
            "fields": list(rows[0]) if rows else [],
        },
        "test": {
            "candidate_pids": candidates,
            "enroll_pids": enroll,
            "imposter_pids": imposters,
            "gallery_covered_pids": gallery_covered,
            "p2_cross_outfit_query_tracks": p2_tracks,
            "p3_same_outfit_cross_camera_query_tracks": p3_tracks,
        },
        "calibration": {
            "source": "MEVID official train split only",
            "test_used_for_threshold_selection": False,
            "enroll_pids": calibration_enroll,
            "imposter_pids": calibration_imposters,
            "gallery_covered_pids": calibration_gallery_covered,
        },
        "counts": {
            "rows": len(rows),
            "roles": role_counts,
            "test_candidate_identities": len(candidates),
            "test_gallery_covered_identities": len(gallery_covered),
            "calibration_gallery_covered_identities": len(
                calibration_gallery_covered
            ),
            "p2_query_tracks": len(p2_tracks),
            "p3_query_tracks": len(p3_tracks),
        },
    }
    payload["protocol_id"] = stable_hash(protocol_identity(payload))
    atomic_write_text(manifest_path, csv_text)
    atomic_write_json(protocol_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="冻结MEVID人形ReID的test与train校准样本"
    )
    parser.add_argument("--data", required=True, help="MEVID根目录")
    parser.add_argument(
        "--manifest",
        default=str(EXPERIMENT_DIR / "manifests" / "frozen_body_samples.csv"),
    )
    parser.add_argument(
        "--protocol",
        default=str(EXPERIMENT_DIR / "manifests" / "frozen_protocol.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enroll-subjects", type=int, default=27)
    parser.add_argument("--imposter-subjects", type=int, default=25)
    parser.add_argument("--frames-per-track", type=int, default=8)
    parser.add_argument("--max-gallery-tracks", type=int, default=3)
    parser.add_argument("--max-query-tracks", type=int, default=4)
    parser.add_argument("--calibration-enroll-subjects", type=int, default=50)
    parser.add_argument("--calibration-imposter-subjects", type=int, default=-1)
    parser.add_argument("--calibration-gallery-tracks", type=int, default=3)
    parser.add_argument("--calibration-query-tracks", type=int, default=4)
    parser.add_argument("--expected-test-identities", type=int, default=52)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = FreezeConfig(
        seed=args.seed,
        enroll_subjects=args.enroll_subjects,
        imposter_subjects=args.imposter_subjects,
        frames_per_track=args.frames_per_track,
        max_gallery_tracks=args.max_gallery_tracks,
        max_query_tracks=args.max_query_tracks,
        calibration_enroll_subjects=args.calibration_enroll_subjects,
        calibration_imposter_subjects=args.calibration_imposter_subjects,
        calibration_gallery_tracks=args.calibration_gallery_tracks,
        calibration_query_tracks=args.calibration_query_tracks,
        expected_test_identities=args.expected_test_identities,
    )
    payload = freeze_protocol(
        Path(args.data),
        Path(args.manifest),
        Path(args.protocol),
        config,
        force=args.force,
    )
    print(f"[saved] protocol={payload['protocol_id']}")
    print(f"[saved] manifest={Path(args.manifest).resolve()}")
    print(f"[saved] protocol_json={Path(args.protocol).resolve()}")
    print(
        "[summary] "
        f"rows={payload['counts']['rows']} "
        f"test_ids={payload['counts']['test_candidate_identities']} "
        f"P2={payload['counts']['p2_query_tracks']} "
        f"P3={payload['counts']['p3_query_tracks']}"
    )
    return 0
