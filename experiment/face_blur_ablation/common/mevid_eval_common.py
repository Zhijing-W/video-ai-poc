"""MEVID 两类实验共享的数据加载、特征提取和开放集指标。"""
from __future__ import annotations

import datetime as _dt
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
ROOT = EXPERIMENT_DIR.parents[1]
OUT_DIR = EXPERIMENT_DIR / "results" / "runs"
BIN_ORDER = ("clear", "marginal", "poor", "none")
CAUSE_BIN_ORDER = ("clear", "marginal", "small-face", "blur", "pose", "other-poor", "none")
CAUSE_TAG_ORDER = ("clear", "marginal", "small-face", "blur", "pose", "other-poor", "none")
_CATEGORY_RANK = {"none": 0, "poor": 1, "marginal": 2, "clear": 3}


@dataclass
class Tracklet:
    pid: str
    cam: int
    outfit: int
    track: int
    frames: list[Path]
    is_query: bool = False


def l2norm(value) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def sample_evenly(items: list, count: int) -> list:
    if count <= 0 or len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def load_mevid(data_dir: Path) -> list[Tracklet]:
    """加载 MEVID 官方 test tracklet 和 gallery/query 划分。"""
    ann = data_dir / "annotation" / "mevid-v1-annotation-data"
    bbox = data_dir / "bbox_test"
    required = [
        ann / "test_name.txt",
        ann / "track_test_info.txt",
        ann / "query_IDX.txt",
        bbox,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("MEVID 文件缺失：" + ", ".join(missing))

    names = (ann / "test_name.txt").read_text(encoding="utf-8").split()
    info_lines = (ann / "track_test_info.txt").read_text(encoding="utf-8").strip().splitlines()
    query_idx = {
        int(float(value))
        for value in (ann / "query_IDX.txt").read_text(encoding="utf-8").split()
    }
    if query_idx and (min(query_idx) < 0 or max(query_idx) >= len(info_lines)):
        raise ValueError("query_IDX.txt 必须使用官方 0-based tracklet 行号")

    tracklets: list[Tracklet] = []
    for row, line in enumerate(info_lines):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"track_test_info.txt 第 {row + 1} 行列数不足：{line!r}")
        start, end = int(float(parts[0])), int(float(parts[1]))
        pid = int(float(parts[2]))
        outfit = int(float(parts[3])) if len(parts) > 3 else 0
        cam = int(float(parts[4])) if len(parts) > 4 else 0
        if start < 0 or end < start or end >= len(names):
            raise ValueError(f"track_test_info.txt 第 {row + 1} 行帧范围非法：{start}..{end}")
        pid_s = f"{pid:04d}"
        frames = [bbox / pid_s / names[index] for index in range(start, end + 1)]
        tracklets.append(
            Tracklet(
                pid=pid_s,
                cam=cam,
                outfit=outfit,
                track=row,
                frames=frames,
                is_query=row in query_idx,
            )
        )
    return tracklets


def load_mevid_train(data_dir: Path) -> list[Tracklet]:
    """加载 MEVID 官方 train tracklet；训练集没有 query/gallery 官方划分。"""
    ann = data_dir / "annotation" / "mevid-v1-annotation-data"
    bbox = data_dir / "bbox_train"
    required = [
        ann / "train_name.txt",
        ann / "track_train_info.txt",
        bbox,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("MEVID train 文件缺失：" + ", ".join(missing))

    names = (ann / "train_name.txt").read_text(encoding="utf-8").split()
    info_lines = (ann / "track_train_info.txt").read_text(encoding="utf-8").strip().splitlines()
    tracklets = []
    for row, line in enumerate(info_lines):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"track_train_info.txt 第 {row + 1} 行列数不足：{line!r}")
        start, end = int(float(parts[0])), int(float(parts[1]))
        pid = int(float(parts[2]))
        outfit = int(float(parts[3]))
        cam = int(float(parts[4]))
        if end < start:
            continue
        if start < 0 or end >= len(names):
            raise ValueError(f"track_train_info.txt 第 {row + 1} 行帧范围非法：{start}..{end}")
        pid_s = f"{pid:04d}"
        frames = [bbox / pid_s / names[index] for index in range(start, end + 1)]
        tracklets.append(
            Tracklet(
                pid=pid_s,
                cam=cam,
                outfit=outfit,
                track=row,
                frames=frames,
                is_query=False,
            )
        )
    return tracklets


def group_tracklets(tracklets: list[Tracklet]):
    grouped = defaultdict(lambda: {"gallery": [], "query": []})
    for tracklet in tracklets:
        grouped[tracklet.pid]["query" if tracklet.is_query else "gallery"].append(tracklet)
    return grouped


def split_subjects(
    candidate_pids: list[str],
    enroll_subjects: int,
    imposter_subjects: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """稳定划分建档/冒充身份；imposter_subjects<0 表示使用全部剩余身份。"""
    pids = sorted(candidate_pids)
    random.Random(seed).shuffle(pids)
    if enroll_subjects <= 0 or enroll_subjects >= len(pids):
        raise ValueError(f"建档身份数必须在 1..{len(pids) - 1}，收到 {enroll_subjects}")
    remaining = len(pids) - enroll_subjects
    requested_imposters = remaining if imposter_subjects < 0 else imposter_subjects
    if requested_imposters <= 0 or requested_imposters > remaining:
        raise ValueError(
            f"当前协议共有 {len(pids)} 个可用身份；建档 {enroll_subjects} 后最多 "
            f"{remaining} 个冒充身份，收到 {requested_imposters}"
        )
    return pids[:enroll_subjects], pids[enroll_subjects:enroll_subjects + requested_imposters]


def load_checkin_anchors(data_dir: Path) -> dict[str, list[Path]]:
    """仅解析登记照文件名前缀；官方公开资料未证明其与 test person_id 的映射。"""
    base = data_dir / "actor_checkin"
    if not base.exists():
        raise FileNotFoundError(f"MEVID actor_checkin 不存在：{base}")
    by_pid: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((*base.rglob("*.jpg"), *base.rglob("*.png"))):
        match = re.match(r"^(\d+)-", path.stem)
        if not match or path.stem.endswith("-b"):
            continue
        by_pid[f"{int(match.group(1)):04d}"].append(path)
    return dict(by_pid)


def select_gallery_face_anchors(
    tracklets: list[Tracklet],
    face_mod,
    settings,
    candidates_per_track: int,
    anchors_per_subject: int,
) -> list[Path]:
    """只从官方 gallery 中按原图人脸质量确定性选择固定锚点，不读取 query。"""
    manifest = select_diverse_gallery_face_anchors(
        tracklets,
        face_mod,
        settings,
        candidates_per_track,
        anchors_per_subject,
    )
    return [item["path"] for item in manifest]


def select_diverse_gallery_face_anchors(
    tracklets: list[Tracklet],
    face_mod,
    settings,
    candidates_per_track: int,
    anchors_per_subject: int,
) -> list[dict]:
    """从不同 gallery 轨迹/相机/套装中贪心选择共同人脸锚点。"""
    from PIL import Image

    settings.face_rec_backend = "arcface"
    candidates = []
    seen = set()
    for tracklet in sorted(tracklets, key=lambda item: item.track):
        if tracklet.is_query:
            raise ValueError("人脸锚点选择收到 query tracklet，存在数据泄漏")
        for path in sample_evenly(tracklet.frames, candidates_per_track):
            if path in seen:
                continue
            seen.add(path)
            pil = Image.open(path).convert("RGB")
            faces = face_mod.detect(pil, with_quality=True, enhance_blurry=False)
            if not faces:
                continue
            best = max(faces, key=lambda item: float(item.get("det_score", 0.0)))
            quality = best.get("quality") or {}
            rank = (
                _CATEGORY_RANK.get(quality.get("category", "poor"), 0),
                float(quality.get("quality") or 0.0),
                float(best.get("det_score") or 0.0),
                str(path),
            )
            candidates.append(
                {
                    "path": path,
                    "track": tracklet.track,
                    "cam": tracklet.cam,
                    "outfit": tracklet.outfit,
                    "rank": rank,
                    "category": quality.get("category", "poor"),
                    "quality": float(quality.get("quality") or 0.0),
                    "reason": quality.get("reason"),
                }
            )

    selected = []
    used_tracks = set()
    used_cams = set()
    used_outfits = set()
    remaining = list(candidates)
    while remaining and len(selected) < anchors_per_subject:
        best = max(
            remaining,
            key=lambda item: (
                item["track"] not in used_tracks,
                item["cam"] not in used_cams,
                item["outfit"] not in used_outfits,
                item["rank"],
            ),
        )
        selected.append(best)
        used_tracks.add(best["track"])
        used_cams.add(best["cam"])
        used_outfits.add(best["outfit"])
        remaining = [
            item
            for item in remaining
            if item["path"] != best["path"] and item["track"] != best["track"]
        ]
    return selected


def new_face_slot() -> dict:
    return {
        "frames": [],
        "best_emb": None,
        "best_cat": "none",
        "best_q": None,
        "best_reason": None,
        "best_quality": {},
        "best_tags": [],
        "enhanced": 0,
        "superres_attempted": 0,
    }


def _update_face_slot(slot: dict, frame: dict) -> None:
    slot["frames"].append(frame)
    if frame["enhanced"]:
        slot["enhanced"] += 1
    if frame.get("superres_attempted"):
        slot["superres_attempted"] += 1
    if frame["emb"] is None:
        return
    current = (_CATEGORY_RANK.get(slot["best_cat"], 0), float(slot["best_q"] or 0.0))
    candidate = (_CATEGORY_RANK.get(frame["cat"], 0), float(frame["q"] or 0.0))
    if slot["best_emb"] is None or candidate > current:
        slot["best_emb"] = frame["emb"]
        slot["best_cat"] = frame["cat"]
        slot["best_q"] = frame["q"]
        slot["best_reason"] = frame.get("reason")
        slot["best_quality"] = frame.get("quality") or {}
        slot["best_tags"] = list(frame.get("tags") or [])


def quality_cause_bucket(slot: dict) -> str:
    """把产品质量类别进一步拆成主要退化原因。"""
    if slot.get("best_emb") is None:
        return "none"
    category = slot.get("best_cat", "poor")
    if category in {"clear", "marginal"}:
        return category
    tags = slot.get("best_tags") or []
    for candidate in ("small-face", "blur", "pose"):
        if candidate in tags:
            return candidate
    return "other-poor"


def quality_degradation_tags(slot: dict) -> list[str]:
    if slot.get("best_emb") is None:
        return ["none"]
    tags = list(slot.get("best_tags") or [])
    if tags:
        return tags
    category = slot.get("best_cat", "poor")
    return [category if category in {"clear", "marginal"} else "other-poor"]


def aggregate_face(slot: dict, allowed_categories: set[str] | None = None) -> np.ndarray | None:
    """按产品连续质量分对轨迹内多帧人脸 embedding 加权聚合。"""
    vectors = []
    weights = []
    for frame in slot["frames"]:
        if frame["emb"] is None:
            continue
        if allowed_categories is not None and frame["cat"] not in allowed_categories:
            continue
        vectors.append(frame["emb"])
        weights.append(max(0.05, float(frame["q"] or 0.0)))
    if not vectors:
        return None
    return l2norm(np.average(np.stack(vectors), axis=0, weights=np.asarray(weights)))


def extract_track_features(
    tracklet: Tracklet,
    face_mod,
    reid_mod,
    gait_mod,
    gallery_mod,
    settings,
    face_variants: dict[str, dict],
    faces_per_track: int,
    gait_frames: int,
    *,
    need_body: bool,
    need_gait: bool,
) -> dict:
    """一次遍历采样帧，提取所有人脸 variant、产品门控后人形模板及步态向量。"""
    from PIL import Image
    from insightface.utils import face_align

    record = {
        "pid": tracklet.pid,
        "face": {name: new_face_slot() for name in face_variants},
        "body": None,
        "body_enroll": None,
        "body_quality": {"accepted": 0, "total": 0},
        "gait": None,
    }
    body_vectors = []
    accepted_body_vectors = []

    for path in sample_evenly(tracklet.frames, faces_per_track):
        pil = Image.open(path).convert("RGB")
        settings.face_rec_backend = "arcface"
        detected = face_mod.detect(pil, with_quality=True, enhance_blurry=False)
        best = max(detected, key=lambda item: float(item.get("det_score", 0.0))) if detected else None

        aligned_bgr = None
        enhanced_bgr = None
        quality = {}
        quality_tags = []
        if best is not None:
            quality = best.get("quality") or {}
            if quality.get("reason") == "too_small":
                quality_tags.append("small-face")
            blur_var = quality.get("blur_var")
            if blur_var is not None and blur_var < settings.face_blur_clear_var:
                quality_tags.append("blur")
            yaw = abs(float(quality.get("yaw") or 0.0))
            pitch = abs(float(quality.get("pitch") or 0.0))
            if yaw > settings.face_yaw_clear or pitch > settings.face_pitch_clear:
                quality_tags.append("pose")
            if not quality_tags:
                category = quality.get("category", "poor")
                quality_tags.append(
                    category if category in {"clear", "marginal"} else "other-poor"
                )
            if best.get("kps") is not None:
                bgr = np.asarray(pil)[:, :, ::-1].copy()
                aligned_bgr = face_align.norm_crop(
                    bgr,
                    np.asarray(best["kps"], dtype=np.float32),
                    image_size=112,
                )

        for name, variant in face_variants.items():
            emb = None
            enhanced = False
            superres_attempted = False
            if best is not None:
                backend = variant["backend"]
                use_superres = bool(variant.get("superres"))
                source_bgr = aligned_bgr
                if use_superres and aligned_bgr is not None:
                    blur_var = quality.get("blur_var")
                    is_blurry = blur_var is not None and blur_var < settings.face_blur_clear_var
                    angle_ok = abs(float(quality.get("yaw") or 0.0)) < settings.face_yaw_max
                    if is_blurry and angle_ok:
                        superres_attempted = True
                        if enhanced_bgr is None:
                            aligned_rgb = Image.fromarray(aligned_bgr[:, :, ::-1])
                            restored = face_mod.enhance(aligned_rgb, aligned=True)
                            if restored is not aligned_rgb:
                                enhanced_bgr = np.asarray(restored.convert("RGB"))[:, :, ::-1].copy()
                        if enhanced_bgr is not None:
                            source_bgr = enhanced_bgr
                            enhanced = True
                if backend == "arcface" and not enhanced:
                    emb = best.get("embedding")
                elif source_bgr is not None and quality.get("can_match", True):
                    emb = face_mod.embed_aligned_face(source_bgr, backend)

            _update_face_slot(
                record["face"][name],
                {
                    "emb": l2norm(emb) if emb is not None else None,
                    "cat": quality.get("category", "none") if best is not None else "none",
                    "q": quality.get("quality") if best is not None else None,
                    "det": float(best.get("det_score", 0.0)) if best is not None else 0.0,
                    "enhanced": enhanced,
                    "superres_attempted": superres_attempted,
                    "reason": quality.get("reason") if best is not None else None,
                    "quality": dict(quality) if best is not None else {},
                    "tags": list(quality_tags),
                    "path": str(path),
                },
            )

        if need_body:
            body_quality = reid_mod.assess_quality(pil)
            accepted, _ = gallery_mod.quality_ok(body_quality)
            vector = l2norm(reid_mod.embed(pil))
            body_vectors.append(vector)
            record["body_quality"]["total"] += 1
            if accepted:
                accepted_body_vectors.append(vector)
                record["body_quality"]["accepted"] += 1

    if body_vectors:
        record["body"] = l2norm(np.mean(body_vectors, axis=0))
    if accepted_body_vectors:
        record["body_enroll"] = l2norm(np.mean(accepted_body_vectors, axis=0))
    if need_gait:
        record["gait"] = extract_gait(tracklet, gait_mod, gait_frames)
    return record


def extract_gait(tracklet: Tracklet, gait_mod, gait_frames: int) -> np.ndarray | None:
    import cv2
    from PIL import Image

    pose_sequence = []
    silhouette_sequence = []
    for path in sample_evenly(tracklet.frames, gait_frames):
        bgr = cv2.cvtColor(np.asarray(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        persons = gait_mod.extract_persons(bgr)
        if not persons:
            continue
        best = max(
            persons,
            key=lambda item: (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1]),
        )
        pose_sequence.append(best["kpts"])
        silhouette_sequence.append(best["mask"])
    if not pose_sequence:
        return None
    embedding = gait_mod.embed_track(pose_sequence, silhouette_sequence)
    return l2norm(embedding) if embedding is not None else None


def build_mean_templates(vectors_by_pid: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        pid: l2norm(np.mean(vectors, axis=0))
        for pid, vectors in vectors_by_pid.items()
        if vectors
    }


def top1(vector: np.ndarray | None, templates: dict[str, np.ndarray]) -> tuple[str | None, float]:
    if vector is None or not templates:
        return None, 0.0
    scores = {pid: float(template @ vector) for pid, template in templates.items()}
    pid = max(scores, key=scores.get)
    return pid, scores[pid]


def fuse_route_votes(
    route_inputs: dict[str, tuple[np.ndarray | None, dict[str, np.ndarray], float]],
    agree_bonus: float,
) -> dict:
    """各模态先独立选 top-1，再在 canonical identity 空间投票。

    本实验中每个模态模板都由 MEVID 真值 person_id 建立，因此 top-1 返回的不是各 gallery
    自增的本地 ID，而是同一套 MEVID canonical pid。产品运行时必须先通过跨模态 subject
    registry 把 face/body/gait 的本地 gallery ID 映射到 canonical subject，才能复用此逻辑。
    """
    routes = {}
    total_weight = 0.0
    contributions: dict[str, float] = defaultdict(float)
    agreeing: dict[str, list[str]] = defaultdict(list)
    for route, (vector, templates, weight) in route_inputs.items():
        pred, raw_score = top1(vector, templates)
        if pred is None or weight <= 0:
            continue
        strength = max(0.0, min(1.0, raw_score))
        if strength <= 0:
            continue
        routes[route] = {"pred": pred, "score": round(raw_score, 6), "weight": round(weight, 6)}
        total_weight += weight
        contributions[pred] += weight * strength
        agreeing[pred].append(route)
    if not contributions or total_weight <= 0:
        return {"pred": None, "confidence": 0.0, "agreed_routes": [], "routes": routes}
    pred = max(contributions, key=contributions.get)
    confidence = contributions[pred] / total_weight
    agreed_routes = agreeing[pred]
    if len(agreed_routes) >= 2:
        confidence = min(1.0, confidence + agree_bonus)
    return {
        "pred": pred,
        "confidence": round(float(confidence), 6),
        "agreed_routes": sorted(agreed_routes),
        "routes": routes,
    }


def det_curve(rows: list[dict]) -> list[dict]:
    genuine = [row for row in rows if row["genuine"]]
    imposters = [row for row in rows if not row["genuine"]]
    scores = sorted({float(row["confidence"]) for row in rows}, reverse=True)
    thresholds = [1.000001, *scores]
    points = []
    for threshold in thresholds:
        correct = sum(
            1
            for row in genuine
            if row["confidence"] >= threshold and row["pred"] == row["gt"]
        )
        false_matches = sum(
            1
            for row in imposters
            if row["pred"] is not None and row["confidence"] >= threshold
        )
        points.append(
            {
                "threshold": round(float(threshold), 6),
                "tpir": round(correct / len(genuine), 6) if genuine else 0.0,
                "fmr": round(false_matches / len(imposters), 6) if imposters else 0.0,
                "fnir": round(1.0 - (correct / len(genuine)), 6) if genuine else 1.0,
            }
        )
    return points


def select_operating_point(points: list[dict], target_fmr: float) -> dict:
    eligible = [point for point in points if point["fmr"] <= target_fmr]
    if not eligible:
        return {"target_fmr": target_fmr, "threshold": None, "tpir": 0.0, "fmr": None}
    best = max(
        eligible,
        key=lambda point: (point["tpir"], -point["fmr"], -point["threshold"]),
    )
    return {"target_fmr": target_fmr, **best}


def summarize_open_set(rows: list[dict], targets: tuple[float, ...]) -> dict:
    genuine = [row for row in rows if row["genuine"]]
    imposters = [row for row in rows if not row["genuine"]]
    curve = det_curve(rows)
    operating = {
        f"{target:.3f}": select_operating_point(curve, target)
        for target in targets
    }
    by_bin = {}
    for category in BIN_ORDER:
        selected = [row for row in genuine if row["quality_bin"] == category]
        by_bin[category] = {
            "total": len(selected),
            "rank1_correct": sum(1 for row in selected if row["pred"] == row["gt"]),
        }
        by_bin[category]["rank1_rate"] = (
            round(by_bin[category]["rank1_correct"] / len(selected), 6)
            if selected
            else None
        )
        for key, point in operating.items():
            threshold = point["threshold"]
            accepted = (
                sum(
                    1
                    for row in selected
                    if threshold is not None
                    and row["confidence"] >= threshold
                    and row["pred"] == row["gt"]
                )
                if selected
                else 0
            )
            by_bin[category][f"tpir_at_fmr_{key}"] = (
                round(accepted / len(selected), 6) if selected else None
            )

    rank1_correct = sum(1 for row in genuine if row["pred"] == row["gt"])
    return {
        "genuine_tracks": len(genuine),
        "imposter_tracks": len(imposters),
        "rank1_correct": rank1_correct,
        "rank1_rate": round(rank1_correct / len(genuine), 6) if genuine else None,
        "operating_points": operating,
        "by_quality_bin": by_bin,
        "det": curve,
    }


def summarize_with_thresholds(
    rows: list[dict],
    thresholds: dict[str, float | None],
) -> dict:
    """使用校准集冻结的阈值评测，禁止在当前 rows 上重新选择 operating point。"""
    genuine = [row for row in rows if row["genuine"]]
    imposters = [row for row in rows if not row["genuine"]]
    operating = {}
    for key, threshold in thresholds.items():
        correct = sum(
            1
            for row in genuine
            if threshold is not None
            and row["pred"] == row["gt"]
            and row["confidence"] >= threshold
        )
        false_matches = sum(
            1
            for row in imposters
            if threshold is not None
            and row["pred"] is not None
            and row["confidence"] >= threshold
        )
        tpir = correct / len(genuine) if genuine else 0.0
        fmr = false_matches / len(imposters) if imposters else 0.0
        operating[key] = {
            "threshold": threshold,
            "tpir": round(tpir, 6),
            "fmr": round(fmr, 6),
            "fnir": round(1.0 - tpir, 6),
            "correct": correct,
            "false_matches": false_matches,
        }

    def summarize_bins(field: str, order: tuple[str, ...]) -> dict:
        result = {}
        for category in order:
            selected = [row for row in genuine if row.get(field) == category]
            rank1_correct = sum(1 for row in selected if row["pred"] == row["gt"])
            item = {
                "total": len(selected),
                "rank1_correct": rank1_correct,
                "rank1_rate": round(rank1_correct / len(selected), 6) if selected else None,
            }
            for key, threshold in thresholds.items():
                correct = sum(
                    1
                    for row in selected
                    if threshold is not None
                    and row["pred"] == row["gt"]
                    and row["confidence"] >= threshold
                )
                item[f"tpir_at_{key}"] = round(correct / len(selected), 6) if selected else None
            result[category] = item
        return result

    def summarize_tags() -> dict:
        result = {}
        for tag in CAUSE_TAG_ORDER:
            selected = [
                row
                for row in genuine
                if tag in (row.get("degradation_tags") or [])
            ]
            rank1_correct = sum(1 for row in selected if row["pred"] == row["gt"])
            item = {
                "total": len(selected),
                "rank1_correct": rank1_correct,
                "rank1_rate": round(rank1_correct / len(selected), 6) if selected else None,
            }
            for key, threshold in thresholds.items():
                correct = sum(
                    1
                    for row in selected
                    if threshold is not None
                    and row["pred"] == row["gt"]
                    and row["confidence"] >= threshold
                )
                item[f"tpir_at_{key}"] = round(correct / len(selected), 6) if selected else None
            result[tag] = item
        return result

    rank1_correct = sum(1 for row in genuine if row["pred"] == row["gt"])
    return {
        "genuine_tracks": len(genuine),
        "imposter_tracks": len(imposters),
        "rank1_correct": rank1_correct,
        "rank1_rate": round(rank1_correct / len(genuine), 6) if genuine else None,
        "operating_points": operating,
        "by_quality_bin": summarize_bins("quality_bin", BIN_ORDER),
        "by_cause_bin": summarize_bins("cause_bin", CAUSE_BIN_ORDER),
        "by_degradation_tag": summarize_tags(),
        "det": det_curve(rows),
    }


def mean_ci95(
    values: list[float],
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> dict:
    """返回均值、样本标准差和小样本 t 分布 95% CI。"""
    values = [value for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    t975 = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    critical = t975.get(len(array) - 1, 1.96)
    half = critical * std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    low = mean - half
    high = mean + half
    if clip is not None:
        low = max(clip[0], low)
        high = min(clip[1], high)
    return {
        "n": len(values),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "ci95_low": round(low, 6),
        "ci95_high": round(high, 6),
    }


def save_result(kind: str, payload: dict, stem: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"{kind}_{stem}_{timestamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def print_operating_table(results: dict, targets: tuple[float, ...]) -> None:
    headers = ["arm", "rank1", *[f"TPIR@FMR<={target:g}" for target in targets], "coverage"]
    print("  ".join(f"{header:>16}" for header in headers))
    for arm, result in results.items():
        summary = result["summary"]
        cells = [arm, _pct(summary["rank1_rate"])]
        for target in targets:
            point = summary["operating_points"][f"{target:.3f}"]
            cells.append(_pct(point["tpir"]))
        coverage = result.get("coverage") or {}
        cells.append("/".join(str(coverage.get(key, 0)) for key in ("face", "body", "gait")))
        print("  ".join(f"{cell:>16}" for cell in cells))


def _pct(value) -> str:
    return "-" if value is None else f"{100.0 * float(value):.1f}%"


ProgressCallback = Callable[[int, int], None]
