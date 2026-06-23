"""多线索融合 + 最佳帧投票（Phase 3 · Step 15 / 设计文档 3.5）—— 别赌单帧。

定位：`gallery.identify`（Step 14）是**逐帧**给出"这一帧 crop 像谁"。但单帧可能糊、可能
背身、可能恰好被遮挡——只信一帧会随机出错。本模块在一条轨迹（同 `track_id`，由 Step 11
跟踪保证身份连续）上**攒多帧证据再下结论**：

  1. **最佳帧选择**：一个 track 的若干帧里，挑最清晰/最大的那帧（清晰度=拉普拉斯方差，
     再按面积/长宽比加权）——识别就用它，别用糊帧。
  2. **多帧投票**：对该 track 各帧的识别结果做**加权投票**（权重=帧质量×ReID 分×运动连续×
     颜色一致），压住个别帧的随机误判。
  3. **多线索融合**：最终身份置信度 = 时序连续性（最强先验，黏住上次结论防抖）+ ReID 分 +
     颜色一致 + 位置/运动连续 的加权综合，而非只信任一个线索。人脸线索留好插槽（Step 17）。

与 gallery 解耦：本模块**只做聚合裁决**，不查库、不调 LLM。上层（三时钟编排 Step 12）每帧
把"这一帧某 track 的观测"喂进来（`add_observation`），需要给整条轨迹定身份时调 `resolve`。

有状态 & 会话隔离：与 `tracker.py` / `gallery.py` 一致，按 `session_id` 隔离；每条 track 维护
一个定长观测环形缓冲（`fusion_buffer_size`）。换视频调用 `reset_fusion`。
"""
from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field

from .config import settings


@dataclass
class Observation:
    """同一 track 在某一帧的一次观测（喂给融合器的最小单位）。"""

    frame_idx: int
    box: list[float] | None = None
    quality: dict = field(default_factory=dict)   # reid.assess_quality 的输出
    q_score: float = 0.0                           # 由 quality 派生的"清晰度×大小"标量（最佳帧用）
    reid_subject: int | None = None                # gallery 这一帧判出的 subject_id（可能 None）
    reid_decision: str | None = None               # hit / new / grey / ...
    reid_score: float = 0.0                         # gallery 余弦分
    color: str | None = None                        # 主色（颜色一致线索）


def _quality_score(quality: dict) -> float:
    """把质量字典折成一个"越大越好"的标量：清晰度为主，面积/长宽比为辅。"""
    blur = float(quality.get("blur_var", 0.0))          # 拉普拉斯方差：越大越清晰
    area = float(quality.get("area", 0.0))
    size = min(1.0, area / float(settings.fusion_ref_area)) if settings.fusion_ref_area > 0 else 1.0
    ar = quality.get("aspect_ratio")
    ar_ok = 1.0 if (ar is None or settings.reid_min_aspect <= ar <= settings.reid_max_aspect) else 0.5
    return blur * size * ar_ok


class TrackFusion:
    """单条 track 的多帧证据缓冲 + 投票/融合裁决。"""

    def __init__(self) -> None:
        self.obs: deque[Observation] = deque(maxlen=settings.fusion_buffer_size)
        self.last_resolved: int | None = None   # 上次定下的 subject_id（时序黏滞先验）

    # ---- 观测累积 ----
    def add(self, ob: Observation) -> None:
        ob.q_score = _quality_score(ob.quality)
        self.obs.append(ob)

    def _modal_color(self) -> str | None:
        counts: dict[str, int] = {}
        for o in self.obs:
            if o.color:
                counts[o.color] = counts.get(o.color, 0) + 1
        return max(counts, key=counts.get) if counts else None

    def best(self) -> Observation | None:
        """最清晰/最大的那一帧（最佳帧选择）。"""
        return max(self.obs, key=lambda o: o.q_score) if self.obs else None

    def _motion_weight(self, ob: Observation, prev: Observation | None) -> float:
        """位置/运动连续性：相邻帧中心跳变越大，越像 ID 切换，权重越低（高斯衰减）。"""
        if prev is None or not ob.box or not prev.box:
            return 1.0
        cx = (ob.box[0] + ob.box[2]) / 2.0
        cy = (ob.box[1] + ob.box[3]) / 2.0
        pcx = (prev.box[0] + prev.box[2]) / 2.0
        pcy = (prev.box[1] + prev.box[3]) / 2.0
        diag = math.hypot(ob.box[2] - ob.box[0], ob.box[3] - ob.box[1]) or 1.0
        dist = math.hypot(cx - pcx, cy - pcy) / diag
        sigma = max(1e-3, settings.fusion_motion_sigma)
        return math.exp(-(dist / sigma) ** 2)

    # ---- 融合裁决 ----
    def resolve(self) -> dict:
        """对当前 track 的所有观测做投票 + 多线索融合，给整条轨迹定一个身份。

        decision ∈ {resolved, uncertain, unresolved}：
          - resolved   : 融合置信度 ≥ fusion_resolve_thresh → 采信该 subject，可整条复用、不调 LLM。
          - uncertain  : 有候选但置信度不够 → 灰区，交上层升级（细粒度/人脸/LLM）。
          - unresolved : 还没有任何帧识别出 subject（全是新/灰）→ 信息不足。
        """
        if not self.obs:
            return {"decision": "unresolved", "subject_id": None, "fused_score": 0.0,
                    "support": 0, "frames": 0}

        modal_color = self._modal_color()
        prev: Observation | None = None
        tally: dict[int, float] = {}
        winner_frames: dict[int, list[Observation]] = {}
        motion_weights: list[float] = []
        color_consistent = 0

        for o in self.obs:
            mw = self._motion_weight(o, prev)
            motion_weights.append(mw)
            cw = 1.0
            if modal_color is not None and o.color is not None and o.color != modal_color:
                cw = settings.fusion_color_penalty
            if modal_color is not None and o.color == modal_color:
                color_consistent += 1
            prev = o
            if o.reid_subject is None:
                continue
            w = o.q_score * max(o.reid_score, 0.0) * mw * cw
            if w <= 0:
                w = 1e-6  # 给极糊但有判定的帧一点点票，不至于完全失声
            tally[o.reid_subject] = tally.get(o.reid_subject, 0.0) + w
            winner_frames.setdefault(o.reid_subject, []).append(o)

        if not tally:
            return {"decision": "unresolved", "subject_id": None, "fused_score": 0.0,
                    "support": 0, "frames": len(self.obs),
                    "best_frame_idx": self.best().frame_idx if self.best() else None}

        # 时序连续性：黏住上次结论（防身份抖动），仅在它仍获得选票时加成。
        total = sum(tally.values())
        if self.last_resolved in tally and total > 0:
            tally[self.last_resolved] += settings.fusion_continuity_bonus * total

        total = sum(tally.values())
        winner = max(tally, key=tally.get)
        vote_conf = tally[winner] / total if total > 0 else 0.0

        wf = winner_frames.get(winner, [])
        reid_cue = (sum(min(max(o.reid_score, 0.0), 1.0) for o in wf) / len(wf)) if wf else 0.0
        color_cue = color_consistent / len(self.obs) if modal_color is not None else 0.0
        motion_cue = sum(motion_weights) / len(motion_weights) if motion_weights else 1.0
        face_cue = 0.0  # 占位：Step 17 人脸接入后填入

        w_vote = settings.fusion_w_vote
        w_reid = settings.fusion_w_reid
        w_color = settings.fusion_w_color
        w_motion = settings.fusion_w_motion
        w_face = settings.fusion_w_face
        wsum = w_vote + w_reid + w_color + w_motion + w_face
        fused = (
            w_vote * vote_conf + w_reid * reid_cue + w_color * color_cue
            + w_motion * motion_cue + w_face * face_cue
        ) / wsum if wsum > 0 else 0.0

        if fused >= settings.fusion_resolve_thresh:
            decision = "resolved"
            self.last_resolved = winner
        else:
            decision = "uncertain"

        best = self.best()
        return {
            "decision": decision,
            "subject_id": winner,
            "fused_score": round(fused, 4),
            "vote_confidence": round(vote_conf, 4),
            "support": len(wf),                 # 投给胜者的帧数
            "frames": len(self.obs),
            "best_frame_idx": best.frame_idx if best else None,
            "modal_color": modal_color,
            "cues": {
                "vote": round(vote_conf, 4),
                "reid": round(reid_cue, 4),
                "color": round(color_cue, 4),
                "motion": round(motion_cue, 4),
                "face": face_cue,
            },
        }


# ---- 按 session 隔离的融合器注册表（与 tracker.py / gallery.py 同形）----
_sessions: dict[str, dict] = {}   # session_id -> {"tracks": {track_id: TrackFusion}, "lock": Lock}
_registry_lock = threading.Lock()


def _session(session_id: str) -> dict:
    with _registry_lock:
        entry = _sessions.get(session_id)
        if entry is None:
            entry = {"tracks": {}, "lock": threading.Lock()}
            _sessions[session_id] = entry
        return entry


def add_observation(
    session_id: str,
    track_id: int,
    *,
    frame_idx: int,
    box: list[float] | None = None,
    quality: dict | None = None,
    reid_subject: int | None = None,
    reid_decision: str | None = None,
    reid_score: float = 0.0,
    color: str | None = None,
) -> None:
    """把"某 track 在某帧的一次识别观测"累积进融合缓冲。"""
    entry = _session(session_id)
    with entry["lock"]:
        tf = entry["tracks"].get(track_id)
        if tf is None:
            tf = TrackFusion()
            entry["tracks"][track_id] = tf
        tf.add(Observation(
            frame_idx=frame_idx, box=box, quality=quality or {},
            reid_subject=reid_subject, reid_decision=reid_decision,
            reid_score=reid_score, color=color,
        ))


def resolve_track(session_id: str, track_id: int) -> dict:
    """对单条 track 做融合裁决（投票 + 多线索）。"""
    entry = _session(session_id)
    with entry["lock"]:
        tf = entry["tracks"].get(track_id)
        if tf is None:
            return {"decision": "unresolved", "subject_id": None, "fused_score": 0.0,
                    "support": 0, "frames": 0, "track_id": track_id}
        out = tf.resolve()
        out["track_id"] = track_id
        return out


def resolve_session(session_id: str) -> dict:
    """对该 session 所有活跃 track 做融合裁决（便于一次拿到全局身份视图）。"""
    entry = _session(session_id)
    with entry["lock"]:
        results = []
        for track_id, tf in entry["tracks"].items():
            out = tf.resolve()
            out["track_id"] = track_id
            results.append(out)
    return {"session_id": session_id, "tracks": results, "active_tracks": len(results)}


def reset_fusion(session_id: str = "default") -> bool:
    with _registry_lock:
        return _sessions.pop(session_id, None) is not None


def reset_all_fusion() -> int:
    with _registry_lock:
        count = len(_sessions)
        _sessions.clear()
        return count


def active_fusion_sessions() -> list[str]:
    with _registry_lock:
        return list(_sessions)
