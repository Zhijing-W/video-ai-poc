"""选帧②：事件驱动的多帧关键帧选择（Phase 4 · Step 25 / 设计 3.3）。

定位：选帧①（定时均匀 ≈4fps）会产出几百帧；直接喂多模态 LLM 既超 token 又超图片数。本模块
在喂 LLM **之前**把帧数砍到几十张，但**砍的依据是"事件"而非"像素变化"**。

⚠️ 关键原则：**"图像变化大" ≠ "发生了事件"**。像素变化可能只是光照/风/抖动。所以事件由
**语义信号**定义（检测/跟踪/身份/provider 的输出），**不是** ffmpeg scene-change / 像素差。
本模块**只接收上游算好的每帧事件标注**，自己不算场景变化、不碰 ffmpeg——事件定义权在检测/跟踪层。

选帧规则：
  1. 有事件的帧 → 必留（新轨迹进入 / 轨迹离开 / 身份命中 / provider 报警 / 计数变化）；
  2. 每条 track 的最佳帧 → 留一张（最清晰，给认人与叙述）；
  3. 事件前后各留 N 帧上下文（LLM 才看得懂"发生了什么"）；
  4. 其余"无事件、与已选帧太像"的 → 去重丢掉（相邻相似度**只用于去冗余，不定义事件**）；
  5. 全程**保时序**；若仍超过上限，按"事件优先、再按时间均匀"降采样。

输入（每帧一条 FrameMeta，字段可缺省；由上游检测/跟踪/身份填好）：
    {
      "index": 0, "timestamp": 0.0,
      "signature": <可选, 灰度指纹/embedding，用于去重>,
      "active_tracks": [7, 9],           # 本帧活跃 track 集合
      "track_quality": {7: 0.8, 9: 0.2}, # 各 track 在本帧的质量（挑最佳帧用）
      "events": ["new_track:9", "identity_hit:A123"],  # 语义事件（非空即"事件帧"）
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import settings


@dataclass
class FrameMeta:
    """一帧的语义元数据（不含像素；选帧只依据这些上游算好的信号）。"""

    index: int
    timestamp: float = 0.0
    signature: object | None = None             # 去重用：灰度指纹/embedding（可选）
    active_tracks: list[int] = field(default_factory=list)
    track_quality: dict[int, float] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "FrameMeta":
        return cls(
            index=int(d["index"]),
            timestamp=float(d.get("timestamp", 0.0)),
            signature=d.get("signature"),
            active_tracks=[int(t) for t in (d.get("active_tracks") or [])],
            track_quality={int(k): float(v) for k, v in (d.get("track_quality") or {}).items()},
            events=list(d.get("events") or []),
        )

    @property
    def has_event(self) -> bool:
        return bool(self.events)


def _signature_diff(a, b) -> float:
    """两帧签名差异（越大越不一样）。仅用于"去重"，不用于定义事件。

    支持 numpy 向量（灰度指纹/embedding）：用 1 - 余弦 或 归一化 L1；拿不到则返回 1（视作不同，不去重）。
    """
    if a is None or b is None:
        return 1.0
    try:
        import numpy as np

        va = np.asarray(a, dtype=np.float32).reshape(-1)
        vb = np.asarray(b, dtype=np.float32).reshape(-1)
        if va.shape != vb.shape or va.size == 0:
            return 1.0
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na > 0 and nb > 0:
            cos = float(np.dot(va, vb) / (na * nb))
            return max(0.0, 1.0 - cos)
        return float(np.abs(va - vb).mean() / 255.0)  # 未归一化指纹兜底
    except Exception:
        return 1.0


def select_keyframes(
    frames: list[dict] | list[FrameMeta],
    max_frames: int | None = None,
) -> list[int]:
    """事件驱动选关键帧，返回**保持时序**的帧 index 列表。

    Args:
        frames: 每帧的语义元数据（FrameMeta 或等价 dict），按时间顺序。
        max_frames: 喂 LLM 的上限（默认 settings.keyframe_max）。

    Returns:
        选中的帧 index 列表（升序）。
    """
    metas = [f if isinstance(f, FrameMeta) else FrameMeta.from_dict(f) for f in frames]
    if not metas:
        return []
    metas.sort(key=lambda m: m.index)
    cap = max_frames or settings.keyframe_max
    ctx = settings.keyframe_context          # 事件前后各留几帧上下文
    dedup_thresh = settings.keyframe_dedup_diff  # 低于此差异视为"太像"

    keep: set[int] = set()
    n = len(metas)
    pos_by_index = {m.index: i for i, m in enumerate(metas)}

    # 1) 事件帧必留 + 事件前后上下文
    for i, m in enumerate(metas):
        if m.has_event:
            keep.add(m.index)
            for j in range(max(0, i - ctx), min(n, i + ctx + 1)):
                keep.add(metas[j].index)

    # 2) 每条 track 的最佳帧（最清晰）各留一张
    best_pos_for_track: dict[int, int] = {}
    best_q_for_track: dict[int, float] = {}
    for i, m in enumerate(metas):
        for t in m.active_tracks:
            q = m.track_quality.get(t, 0.0)
            if t not in best_q_for_track or q > best_q_for_track[t]:
                best_q_for_track[t] = q
                best_pos_for_track[t] = i
    for i in best_pos_for_track.values():
        keep.add(metas[i].index)

    # 3) 去重：在"无事件"的帧里，丢掉与上一张已保留帧太像的（仅去冗余）
    ordered = [m.index for m in metas]
    pruned: set[int] = set(keep)
    last_kept_sig = None
    last_kept_idx = None
    for m in metas:
        if m.index in keep:
            last_kept_sig = m.signature
            last_kept_idx = m.index
            continue
        # 非事件帧：与上一张保留帧太像则丢
        if last_kept_sig is not None and m.signature is not None:
            if _signature_diff(last_kept_sig, m.signature) < dedup_thresh:
                continue  # 太像，丢
        # 不太像（画面确实推进了）→ 作为候选保留
        pruned.add(m.index)
        last_kept_sig = m.signature
        last_kept_idx = m.index

    selected = sorted(pruned)

    # 4) 仍超上限 → 事件帧优先，其余按时间均匀降采样
    if len(selected) > cap:
        event_idx = {m.index for m in metas if m.has_event}
        must = [ix for ix in selected if ix in event_idx]
        rest = [ix for ix in selected if ix not in event_idx]
        room = max(0, cap - len(must))
        if room <= 0:
            # 事件帧本身就超额 → 在事件帧里按时间均匀取 cap 张
            keep_must = _even_pick(sorted(event_idx & set(selected)), cap)
            return sorted(keep_must)
        rest_keep = _even_pick(rest, room)
        selected = sorted(set(must) | set(rest_keep))

    return selected


def _even_pick(items: list[int], k: int) -> list[int]:
    """从有序 items 里在位置上均匀挑 k 个（k>=len 时全取）。"""
    n = len(items)
    if k >= n:
        return list(items)
    if k <= 0:
        return []
    if k == 1:
        return [items[n // 2]]
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


__all__ = ["FrameMeta", "select_keyframes"]
