"""事件门控服务与 YOLO 结果复用逻辑。"""
from __future__ import annotations

from typing import Any

from ..core import settings
from ..models import GateDecision


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "无目标"
    return "、".join(f"{label}×{count}" for label, count in counts.items())


def _gate_response(passed: bool, reason: str, priority: str, signals: dict[str, Any]) -> dict:
    return GateDecision(
        passed=passed,
        reason=reason,
        priority=priority,
        signals=signals,
    ).model_dump(by_alias=True)


def decide_gate(
    counts: dict[str, int],
    prev_counts: dict[str, int] | None,
    since_last_llm_ms: int | None,
    comparing: bool = False,
) -> dict:
    key_classes = settings.gate_key_class_set()
    hit_key = sorted(label for label in counts if label in key_classes)
    count_changed = prev_counts is not None and counts != prev_counts
    cooling = (
        since_last_llm_ms is not None
        and since_last_llm_ms < settings.gate_cooldown_ms
    )
    silent_too_long = (
        since_last_llm_ms is None
        or since_last_llm_ms >= settings.gate_heartbeat_ms
    )

    signals = {
        "hit_key_classes": hit_key,
        "count_changed": count_changed,
        "cooling": cooling,
        "comparing": comparing,
    }

    if hit_key:
        return _gate_response(True, f"命中关键类别：{', '.join(hit_key)}", "high", signals)
    if comparing and counts:
        return _gate_response(
            True,
            f"比对模式 · 画面有目标（{_format_counts(counts)}）",
            "high",
            signals,
        )
    if cooling:
        return _gate_response(
            False,
            f"冷却中（距上次分析 {since_last_llm_ms}ms < {settings.gate_cooldown_ms}ms）",
            "skip",
            signals,
        )
    if count_changed:
        return _gate_response(
            True,
            f"物体数量变化（{_format_counts(prev_counts or {})} → {_format_counts(counts)}）",
            "medium",
            signals,
        )
    if silent_too_long:
        return _gate_response(True, "心跳巡检（长时间无显著事件，定期复核）", "low", signals)
    return _gate_response(False, "无显著事件（YOLO 标签未变 / 未命中关键类别）", "skip", signals)


def yolo_signature(yolo: dict) -> str:
    items = [
        f"{detection.get('label')}|{detection.get('color') or '-'}"
        for detection in yolo.get("detections", [])
    ]
    items.sort()
    return ";".join(items)


def synthesize_result_from_yolo(yolo: dict, gate_reason: str) -> dict:
    counts = yolo.get("counts", {})
    labels = list(counts.keys())
    scene = (
        "YOLO 检出：" + "、".join(f"{label}×{count}" for label, count in counts.items())
        if counts
        else "画面无显著目标"
    )
    return {
        "scene": scene,
        "detected_objects": labels,
        "match": {"is_match": None, "confidence": None, "target": "", "reason": ""},
        "alert_level": "normal",
        "notification": f"门控跳过 · 未调用 gpt-4o（{gate_reason}）",
    }
