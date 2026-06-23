"""主体记忆集成层（Phase 3 · "连"）—— 把 Step 14(认人) + Step 15(融合) 接进实时流程。

定位：Step 14 `/identify`、Step 15 `/fusion` 此前是独立端点；本模块把它们**接进 Step 12
三时钟的实时 `/analyze-frame` 流程**——每帧跟踪后，对画面里的人逐个查主体记忆库、累积多帧
证据做融合，得到"这条 track 是谁(subject_id)"，并把身份回填到检测框 + 给前端一份身份摘要。

职责边界：**只做"认人 + 汇总"，不改门控判定**。是否调 gpt-4o 仍由 `track_gate`(3.2) 决定；
本层是叠加在其上的"认人 + 可视化"维度。任何失败都应被上层 try/except 兜住，绝不拖垮主流程。

每帧产出：
  - 把 person 检测就地加上 `subject_id` / `subject_decision` / `subject_reused`（供前端画框显示）。
  - 返回身份摘要：backend、记忆库主体数、回头客命中数(cross-track)、本帧各 track 的身份。

有状态 & 会话隔离：按 session_id 维护帧计数 + "subject→出现过的 track 集合" + 回头客计数；
`reset_identity` 一并清空底层 gallery / fusion / 本层状态（换视频/重新开始时调用）。
"""
from __future__ import annotations

import threading

from .. import reid
from .fusion_service import add_observation, reset_fusion, resolve_track
from .gallery_service import gallery_stats, identify_detections, reset_gallery

# session_id -> {"frame": int, "subject_tracks": {subject_id: set(track_id)},
#                "counted_cross": set((subject_id, track_id)), "cross_hits": int}
_state: dict[str, dict] = {}
_lock = threading.Lock()


def _get(session_id: str) -> dict:
    st = _state.get(session_id)
    if st is None:
        st = {"frame": 0, "subject_tracks": {}, "counted_cross": set(), "cross_hits": 0}
        _state[session_id] = st
    return st


def _summary(session_id: str, per_track: list[dict]) -> dict:
    with _lock:
        st = _get(session_id)
        cross_hits = st["cross_hits"]
    try:
        known = gallery_stats(session_id).get("subjects", 0)
    except Exception:
        known = 0
    return {
        "backend": reid.active_backend(),
        "known_subjects": known,
        "cross_track_hits": cross_hits,
        "per_track": per_track,
    }


def enrich_with_identity(session_id: str, image: str, yolo: dict) -> dict:
    """对一帧做"认人"：查主体记忆 + 融合，回填 subject 到 person 检测，返回身份摘要。

    只处理带 track_id 的 person（ReID 以人形为主）；其余检测不动。
    """
    dets = yolo.get("detections") or []
    persons = [d for d in dets if d.get("label") == "person" and d.get("track_id") is not None]
    if not persons:
        return _summary(session_id, [])

    id_input = [{"box": d.get("box"), "track_id": d.get("track_id"), "label": "person"} for d in persons]
    res = identify_detections(image, id_input, session_id=session_id, auto_enroll=True)
    results = res.get("results", [])
    det_by_tid = {d.get("track_id"): d for d in persons}

    with _lock:
        st = _get(session_id)
        frame_idx = st["frame"]
        st["frame"] += 1

    per_track: list[dict] = []
    for r in results:
        tid = r.get("track_id")
        det = det_by_tid.get(tid)
        color = det.get("color") if det else None

        # 累积进融合，并拿到 track 级裁决
        add_observation(
            session_id, tid, frame_idx=frame_idx, box=r.get("box"),
            quality=r.get("quality"), reid_subject=r.get("subject_id"),
            reid_decision=r.get("decision"), reid_score=r.get("score", 0.0), color=color,
        )
        fused = resolve_track(session_id, tid)
        decision = fused.get("decision")
        sid = fused.get("subject_id") if decision == "resolved" else r.get("subject_id")

        reused = False
        if sid is not None:
            with _lock:
                st = _get(session_id)
                tracks = st["subject_tracks"].setdefault(sid, set())
                is_new_track = tid not in tracks
                if is_new_track and tracks:  # 该主体此前出现在别的 track 上 → 回头客
                    key = (sid, tid)
                    if key not in st["counted_cross"]:
                        st["counted_cross"].add(key)
                        st["cross_hits"] += 1
                tracks.add(tid)
                reused = len(tracks) > 1

        if det is not None:
            det["subject_id"] = sid
            det["subject_decision"] = decision
            det["subject_reused"] = reused

        per_track.append({
            "track_id": tid, "subject_id": sid, "decision": decision,
            "reused": reused, "score": r.get("score"),
        })

    return _summary(session_id, per_track)


def reset_identity(session_id: str = "default") -> bool:
    """清空某 session 的主体记忆集成状态（含底层 gallery / fusion）。"""
    reset_gallery(session_id)
    reset_fusion(session_id)
    with _lock:
        return _state.pop(session_id, None) is not None


__all__ = ["enrich_with_identity", "reset_identity"]
