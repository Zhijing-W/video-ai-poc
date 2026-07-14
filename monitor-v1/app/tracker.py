"""多目标跟踪层（Phase 3 · Step 11 / Phase 4 BoT-SORT+ReID 升级）。

定位：YOLO 每帧"独立"检测，**两帧之间不知道"这个框和上一帧哪个框是同一个目标"**。
本层在 Phase 2 的逐帧检测之上补这条"上下帧关系"——用 MOT（Multi-Object Tracking）
给每个目标分配一个**跨帧稳定的 `track_id`**，把跨帧的检测框关联成"轨迹（track）"。
有了稳定身份，后续就能"识别一次、整条轨迹复用"（Phase 3 最大省钱杠杆）。

实现：复用 **ultralytics 内置 tracker**，支持 `bytetrack` / `botsort` / `botsort_reid`。
检测仍走 `detector._predict`（共享单例权重，不重复推理），本层只负责"把这帧的框关联到已有轨迹/新建轨迹"。

有状态：MOT 是**有状态**的——必须按帧时序、用同一个 tracker 连续喂入。本模块按
`session_id` 隔离 tracker 实例，避免多视频/多会话串味；每个 session 配独立锁，保证
同一会话的帧串行更新（不同会话可并行）。换视频/重新开始时调用 `reset_tracker`。

ByteTrack 工作机制（一句话）：用卡尔曼滤波预测每条轨迹下一帧位置，再用 IoU 做匹配。
BoT-SORT 在此基础上加入全局运动补偿；`botsort_reid` 再接入本项目 `app.reid` 的外观特征，
用于多人交叉/遮挡时降低 ID switch。
"""
from __future__ import annotations

import threading
import time
import types

import numpy as np
from PIL import Image

from .config import settings
from .detector import _predict
from . import reid as reid_mod

# session_id -> {"tracker": tracker instance, "backend": str, "lock": Lock}
_trackers: dict[str, dict] = {}
_registry_lock = threading.Lock()


def _normalize_backend(value: str | None = None) -> str:
    backend = (value or settings.track_backend or "botsort_reid").strip().lower().replace("-", "_")
    aliases = {
        "byte": "bytetrack",
        "byte_track": "bytetrack",
        "bot": "botsort",
        "bot_sort": "botsort",
        "bot_sort_reid": "botsort_reid",
        "botsort+reid": "botsort_reid",
        "botsort_reid": "botsort_reid",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"bytetrack", "botsort", "botsort_reid"}:
        raise ValueError(f"未知 TRACK_BACKEND：{value!r}，可选 bytetrack / botsort / botsort_reid")
    return backend


def _build_args(backend: str) -> types.SimpleNamespace:
    """把 settings 里的 MOT 阈值打包成 Ultralytics tracker 需要的 args 命名空间。"""
    return types.SimpleNamespace(
        tracker_type="bytetrack" if backend == "bytetrack" else "botsort",
        track_high_thresh=settings.track_high_thresh,
        track_low_thresh=settings.track_low_thresh,
        new_track_thresh=settings.new_track_thresh,
        track_buffer=settings.track_buffer,
        match_thresh=settings.track_match_thresh,
        fuse_score=settings.track_fuse_score,
        gmc_method=settings.track_gmc_method,
        proximity_thresh=settings.track_proximity_thresh,
        appearance_thresh=settings.track_appearance_thresh,
        with_reid=backend == "botsort_reid",
        model="auto",
    )


class _AppReIDEncoder:
    """Adapter: BoT-SORT expects encoder(img, xywh_dets); reuse this project's person ReID embedding."""

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        arr = np.asarray(dets)
        dim = reid_mod.embed_dim()
        if arr.size == 0:
            return []
        feats: list[np.ndarray] = []
        h, w = img.shape[:2]
        for det in arr:
            cx, cy, bw, bh = [float(v) for v in det[:4]]
            x1 = max(0, int(round(cx - bw / 2)))
            y1 = max(0, int(round(cy - bh / 2)))
            x2 = min(w, int(round(cx + bw / 2)))
            y2 = min(h, int(round(cy + bh / 2)))
            if x2 <= x1 or y2 <= y1:
                feats.append(np.zeros(dim, dtype=np.float32))
                continue
            crop = Image.fromarray(img[y1:y2, x1:x2][:, :, ::-1])
            feats.append(np.asarray(reid_mod.embed(crop), dtype=np.float32).reshape(-1))
        return feats


def _build_tracker(backend: str):
    if backend == "bytetrack":
        from ultralytics.trackers.byte_tracker import BYTETracker

        return BYTETracker(_build_args(backend))

    from ultralytics.trackers.bot_sort import BOTSORT

    tracker = BOTSORT(_build_args(backend))
    if backend == "botsort_reid":
        tracker.encoder = _AppReIDEncoder()
    return tracker


def _get_entry(session_id: str) -> dict:
    """懒加载：按 session 取（或新建）一个 tracker 实例及其专属锁。"""
    backend = _normalize_backend()
    with _registry_lock:
        entry = _trackers.get(session_id)
        if entry is None or entry.get("backend") != backend:
            entry = {"tracker": _build_tracker(backend), "backend": backend, "lock": threading.Lock()}
            _trackers[session_id] = entry
        return entry


def reset_tracker(session_id: str = "default") -> bool:
    """清空某 session 的跟踪状态（换视频 / 重新开始监控时调用）。

    Returns:
        bool: True 表示该 session 之前存在并已被清除，False 表示本来就没有。
    """
    with _registry_lock:
        return _trackers.pop(session_id, None) is not None


def reset_all_trackers() -> int:
    """清空所有 session 的跟踪状态，返回被清除的 session 数量。"""
    with _registry_lock:
        count = len(_trackers)
        _trackers.clear()
        return count


def active_sessions() -> list[str]:
    """返回当前持有跟踪状态的 session_id 列表（便于排查/运维）。"""
    with _registry_lock:
        return list(_trackers)


def active_backend() -> str:
    """返回当前配置的 tracker backend（不要求已有活跃 session）。"""
    return _normalize_backend()


def track_objects(
    image: str | bytes, session_id: str = "default", conf: float | None = None
) -> dict:
    """对一帧做"检测 + 多目标跟踪"，给每个目标补上跨帧稳定的 track_id。

    必须按视频帧时序、用同一 session_id 连续调用，track_id 才有意义。

    Args:
        image: data URI / 纯 base64 / 原始字节。
        session_id: 跟踪会话标识（不同视频/摄像头用不同 id，互不串味）。
        conf: 喂给跟踪器的检测阈值（None 则用 settings.track_conf，故意取较低值，
              让 ByteTrack 能用低分框做二段关联、抗短遮挡）。

    Returns:
        dict（形状与 detect_objects 对齐，便于复用下游门控/着色逻辑）：
          model, session_id, infer_ms, track_ms, img_w, img_h,
          detections: [{label, confidence, box:[x1,y1,x2,y2], track_id}]
          counts: {label: 数量}
          active_tracks: 当前活跃轨迹数
    """
    entry = _get_entry(session_id)
    r, img_w, img_h, infer_ms = _predict(
        image, conf=settings.track_conf if conf is None else conf
    )

    # ultralytics 官方跟踪管线即如此喂入：把这帧的框转成 numpy，交给 tracker 关联。
    det = r.boxes.cpu().numpy()
    names = r.names

    t0 = time.perf_counter()
    with entry["lock"]:
        # 返回 (N, 8) 数组：[x1, y1, x2, y2, track_id, score, cls, idx]；无活跃轨迹时为空。
        tracks = entry["tracker"].update(det, r.orig_img)
    track_ms = round((time.perf_counter() - t0) * 1000, 1)

    detections: list[dict] = []
    counts: dict[str, int] = {}
    for row in tracks:
        track_id = int(row[4])
        score = round(float(row[5]), 3)
        cls = int(row[6])
        label = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
        box = [round(float(row[i]), 1) for i in range(4)]
        detections.append(
            {"label": label, "confidence": score, "box": box, "track_id": track_id}
        )
        counts[label] = counts.get(label, 0) + 1

    return {
        "model": settings.yolo_model,
        "tracker_backend": entry["backend"],
        "session_id": session_id,
        "infer_ms": infer_ms,
        "track_ms": track_ms,
        "img_w": img_w,
        "img_h": img_h,
        "detections": detections,
        "counts": counts,
        "active_tracks": len(detections),
    }
