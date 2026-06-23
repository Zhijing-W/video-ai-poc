"""事件门控 Event Gate（Phase 2 · Step 6）—— Phase 2 省钱的灵魂。

定位：消费 YOLO（廉价守门员）的检测结果 + 轻量帧间状态，用**纯规则**判断
"这一帧值不值得花钱调 gpt-4o"。只有命中关键事件才放行 LLM，其余帧只记 YOLO 标签。

要点：
  - 不是机器学习模型，**不需要数据集、不需要训练**，全是可解释的 if-else + 阈值。
  - 所有阈值/关键类别在 config.py 可配（.env 可覆盖），换业务场景只改配置不改代码。
  - 帧间状态（上一帧 counts、距上次 LLM 时长）由调用方（前端）传入，后端保持无状态。

决策优先级（短路求值，命中即放行，不再下判）：
  1. 关键类别命中 / 比对模式有目标   → 放行（high）   —— 省钱主力
  2. 冷却中（距上次 LLM 太近）        → 跳过           —— 防烧钱兜底
  3. 物体数量变化（有人/物进出画面）  → 放行（medium） —— 抓进出场事件
  4. 心跳（距上次 LLM 太久）          → 放行（low）    —— 定期巡检安全网
  5. 否则                            → 跳过           —— 复用上次结论，只记 YOLO 标签
"""
from __future__ import annotations

from .config import settings


def _fmt_counts(counts: dict) -> str:
    if not counts:
        return "无目标"
    return "、".join(f"{k}×{v}" for k, v in counts.items())


def decide(
    counts: dict[str, int],
    prev_counts: dict[str, int] | None,
    since_last_llm_ms: int | None,
    comparing: bool = False,
) -> dict:
    """门控判定：是否放行给 gpt-4o。

    Args:
        counts: 本帧 YOLO 各类别数量，如 {"person": 2, "car": 1}。
        prev_counts: 上一帧的 counts（前端传入，首帧为 None）。
        since_last_llm_ms: 距上次真正调用 LLM 的毫秒数（首次为 None）。
        comparing: 是否处于「开始比对」模式（前端传入）。

    Returns:
        dict: {pass(bool), reason(str), priority(high|medium|low|skip), signals(dict)}
    """
    key_classes = settings.gate_key_class_set()
    hit_key = sorted(c for c in counts if c in key_classes)
    count_changed = prev_counts is not None and counts != prev_counts
    cooling = since_last_llm_ms is not None and since_last_llm_ms < settings.gate_cooldown_ms
    silent_too_long = (
        since_last_llm_ms is None or since_last_llm_ms >= settings.gate_heartbeat_ms
    )

    signals = {
        "hit_key_classes": hit_key,
        "count_changed": count_changed,
        "cooling": cooling,
        "comparing": comparing,
    }

    # 1) 强信号：命中关键类别，或比对模式下画面里有任何可比对的目标
    if hit_key:
        return _pass(f"命中关键类别：{', '.join(hit_key)}", "high", signals)
    if comparing and counts:
        return _pass(f"比对模式 · 画面有目标（{_fmt_counts(counts)}）", "high", signals)

    # 2) 冷却中：弱信号一律按下，避免烧钱
    if cooling:
        return _skip(f"冷却中（距上次分析 {since_last_llm_ms}ms < {settings.gate_cooldown_ms}ms）", signals)

    # 3) 变化信号：物体进出 / 数量变化
    if count_changed:
        return _pass(f"物体数量变化（{_fmt_counts(prev_counts or {})} → {_fmt_counts(counts)}）", "medium", signals)

    # 4) 心跳巡检：太久没调 LLM，强制看一眼
    if silent_too_long:
        return _pass("心跳巡检（长时间无显著事件，定期复核）", "low", signals)

    # 5) 跳过：无显著事件，复用上次结论
    return _skip("无显著事件（YOLO 标签未变 / 未命中关键类别）", signals)


def _pass(reason: str, priority: str, signals: dict) -> dict:
    return {"pass": True, "reason": reason, "priority": priority, "signals": signals}


def _skip(reason: str, signals: dict) -> dict:
    return {"pass": False, "reason": reason, "priority": "skip", "signals": signals}
