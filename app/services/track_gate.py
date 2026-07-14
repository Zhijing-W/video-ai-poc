"""Track 级事件门控 + 结论缓存（Phase 3 · Step 12「三时钟解耦」之理解时钟）。

定位：Phase 2 的事件门控（gate_service）是**逐帧/逐计数**的——靠"物体数量是否变化 /
像素签名是否变化"决定要不要调 gpt-4o。Phase 3 有了跨帧稳定的 `track_id` 后，可以把
"理解时钟"升级成**逐轨迹（per-track）**：

> 只要"活跃轨迹集合"没变（还是那几条 track），就**复用上次 gpt-4o 结论**，一次 LLM 都不调；
> 只有**新轨迹出生**（新主体进入画面）才触发 gpt-4o——"认过一次就记住，整条轨迹共享结论"。

这比 Phase 2 的计数门控更稳、更省：同一个人站着不动会一直是同一个 track_id，
计数门控可能因像素抖动反复触发，而 track 门控认得出"还是这条轨迹"，直接复用。

状态：按 `session_id` 维护（活跃/已知轨迹集合、上次 LLM 时间、缓存结论、省调用计数）。
换视频/重新开始监控时调用 `reset_track_gate`（与 tracker 的 reset 配对）。
"""
from __future__ import annotations

import threading
import time

from ..core import settings

# session_id -> 状态字典
_state: dict[str, dict] = {}
_lock = threading.Lock()


def _new_state() -> dict:
    return {
        "known": set(),         # 累计见过的所有 track_id（用于判断"新主体进入"）
        "last_active": set(),   # 上一帧的活跃 track_id（用于判断"主体离开"）
        "last_llm_ts": None,    # 上次真正调用 gpt-4o 的时间戳（ms）；None 表示从未调用
        "conclusion": None,     # 上次 gpt-4o 的结论（供轨迹未变时复用）
        "llm_calls": 0,         # 本会话真正调用 gpt-4o 的次数
        "reuse_calls": 0,       # 本会话因轨迹未变而省下的调用次数
    }


def _get(session_id: str) -> dict:
    st = _state.get(session_id)
    if st is None:
        st = _new_state()
        _state[session_id] = st
    return st


def reset_track_gate(session_id: str = "default") -> bool:
    """清空某 session 的 track 门控状态（换视频/重新开始监控时）。"""
    with _lock:
        return _state.pop(session_id, None) is not None


def decide_track_gate(
    session_id: str,
    detections: list[dict],
    comparing: bool = False,
    now_ms: float | None = None,
) -> dict:
    """根据"活跃轨迹集合"决定本帧是否需要调 gpt-4o。

    Args:
        session_id: 跟踪会话标识（与 tracker 同一个 id）。
        detections: track_objects 产出的检测列表（每个带 track_id）。
        comparing: 比对模式（为真则每帧都过，保证比对准确，不复用）。

    Returns:
        dict:
          verdict: "pass"（需调 LLM） / "reuse"（复用缓存结论） / "skip"（冷却跳过，无缓存可复用）
          priority: high / medium / low / skip
          reason: 人类可读的判定理由
          signals: {active_tracks, new_tracks, left_tracks, new_key_classes}
          since_last_llm_ms: 距上次 LLM 的毫秒数（无则 None）
          conclusion: verdict=="reuse" 时返回缓存结论，否则 None
    """
    if now_ms is None:
        now_ms = time.time() * 1000
    key_classes = settings.gate_key_class_set()

    active: dict[int, str | None] = {}
    for d in detections:
        tid = d.get("track_id")
        if tid is not None:
            active[int(tid)] = d.get("label")
    active_ids = set(active)

    with _lock:
        st = _get(session_id)
        new_ids = active_ids - st["known"]
        left_ids = st["last_active"] - active_ids
        new_key = sorted({active[i] for i in new_ids if active.get(i) in key_classes})

        last_ts = st["last_llm_ts"]
        since = (now_ms - last_ts) if last_ts is not None else None
        cooling = since is not None and since < settings.gate_cooldown_ms
        heartbeat = since is None or since >= settings.gate_heartbeat_ms

        signals = {
            "active_tracks": sorted(active_ids),
            "new_tracks": sorted(new_ids),
            "left_tracks": sorted(left_ids),
            "new_key_classes": new_key,
        }

        if comparing and active_ids:
            verdict, priority, reason = "pass", "high", f"比对模式 · 画面有 {len(active_ids)} 条轨迹（每帧裁决）"
        elif new_key:
            verdict, priority, reason = "pass", "high", f"新主体进入（关键类别）：{', '.join(new_key)}"
        elif new_ids:
            verdict, priority, reason = "pass", "medium", f"新轨迹进入：track {sorted(new_ids)}"
        elif cooling:
            verdict, priority, reason = "skip", "skip", f"冷却中（距上次 gpt-4o {int(since)}ms）"
        elif heartbeat:
            verdict, priority, reason = "pass", "low", "心跳巡检（轨迹集合长时间未变，定期复核）"
        else:
            verdict, priority, reason = "reuse", "low", "轨迹集合未变 · 复用上次 gpt-4o 结论（省一次调用）"

        # 无缓存可复用时（首帧之前从没调过 LLM），把 reuse/skip 都视为 skip（走 YOLO 合成）。
        conclusion = st["conclusion"] if verdict == "reuse" else None
        if verdict == "reuse" and conclusion is None:
            verdict = "skip"

        # 更新轨迹集合状态（"new" 优先级高于 "cooling"，故 known 总是可安全并入）。
        st["last_active"] = active_ids
        st["known"] |= active_ids

        return {
            "verdict": verdict,
            "priority": priority,
            "reason": reason,
            "signals": signals,
            "since_last_llm_ms": int(since) if since is not None else None,
            "conclusion": conclusion,
        }


def record_llm_conclusion(session_id: str, conclusion: dict, now_ms: float | None = None) -> None:
    """记下一次真正的 gpt-4o 结论，供后续轨迹未变时复用。"""
    with _lock:
        st = _get(session_id)
        st["conclusion"] = conclusion
        st["last_llm_ts"] = now_ms if now_ms is not None else time.time() * 1000
        st["llm_calls"] += 1


def record_reuse(session_id: str) -> None:
    """记一次"因轨迹未变而省下的 gpt-4o 调用"。"""
    with _lock:
        _get(session_id)["reuse_calls"] += 1


def gate_stats(session_id: str = "default") -> dict:
    """返回本会话省钱统计：真正调用数 / 复用省下数。"""
    with _lock:
        st = _get(session_id)
        return {"llm_calls": st["llm_calls"], "reuse_calls": st["reuse_calls"]}
