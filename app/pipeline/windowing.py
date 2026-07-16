from __future__ import annotations

from ..keyframe import FrameMeta


def split_windows(
    metas: list[FrameMeta], quiet_frames: int, max_window_frames: int | None = None
) -> list[list[int]]:
    """把帧序列按"活动段 + 时长上限"切窗，返回每个窗的帧 index 列表。

    关窗条件二选一：
      (a) 活动结束：连续 ≥ quiet_frames 帧无人（允许桥接 < quiet_frames 的短暂无人）；
      (b) 时长封顶：窗内帧数达到 max_window_frames（= 时长上限 × fps）→ 冲刷并立刻开新窗。

    (b) 是为"长连续事件"准备的：否则一个人在画面里连续待很久会被压成单窗、只调一次 LLM、
    关键帧被严重欠采样。封顶后长事件会被切成多个窗、各调一次，细节不丢（跨窗的"谁"靠 ReID
    主体记忆保持一致）。
    """
    windows: list[list[int]] = []
    cur: list[int] = []
    gap = 0
    for m in metas:
        if m.active_tracks:
            cur.append(m.index)
            gap = 0
            if max_window_frames and len(cur) >= max_window_frames:
                windows.append(cur)          # 时长封顶 → 冲刷
                cur = []
                gap = 0
        elif cur:
            gap += 1
            if gap >= quiet_frames:
                windows.append(cur)          # 活动结束 → 冲刷
                cur = []
                gap = 0
            else:
                cur.append(m.index)          # 短暂无人，桥接进当前窗
    if cur:
        windows.append(cur)
    return windows


_split_windows = split_windows

__all__ = ["split_windows"]
