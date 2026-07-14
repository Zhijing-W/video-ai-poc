"""逐帧跟踪诊断（临时脚本，不改生产代码）。

复用 out/event-monitor 下最近一次运行已抽好的帧，按时序喂 tracker.track_objects，
逐帧打印每个 person 的 track_id / conf / box，并计算与"上一帧同 track"的 IoU，
标出新建/丢失的 track，定位到底哪一帧、因为什么 track 断开（如 11s→12s）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import tracker as tracker_mod  # noqa: E402
from app.core.config import settings  # noqa: E402


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    runs_dir = ROOT / "out" / "event-monitor"
    latest_run = max((p for p in runs_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    frames_dir = latest_run / "frames"
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        print(f"没有帧：{frames_dir}")
        return
    fps = 2.0  # 与前端那次一致（16s / 32帧）
    sess = "diag"
    tracker_mod.reset_tracker(sess)

    print(f"backend={tracker_mod.active_backend()}  "
          f"match_thresh={settings.track_match_thresh} "
          f"appearance_thresh={settings.track_appearance_thresh} "
          f"proximity={settings.track_proximity_thresh} buffer={settings.track_buffer} "
          f"conf={settings.track_conf} high={settings.track_high_thresh} low={settings.track_low_thresh}")
    print(f"帧数={len(frames)}\n")

    prev_boxes: dict[int, list[float]] = {}   # track_id -> 上一帧 box
    prev_ids: set[int] = set()
    all_ids: set[int] = set()
    switch_frames: list[int] = []
    for i, fp in enumerate(frames):
        raw = fp.read_bytes()
        res = tracker_mod.track_objects(raw, session_id=sess)
        persons = [d for d in res["detections"]
                   if d.get("label") == "person" and d.get("track_id") is not None]
        ts = i / fps
        ids = {int(d["track_id"]) for d in persons}
        new = ids - prev_ids
        lost = prev_ids - ids
        if i > 0 and new:  # 第0帧的新建是正常起始，不算切换
            switch_frames.append(i)
        all_ids |= ids
        parts = []
        for d in sorted(persons, key=lambda x: x["track_id"]):
            tid = int(d["track_id"])
            box = [round(float(v), 1) for v in d["box"]]
            conf = float(d["confidence"])
            io = iou(d["box"], prev_boxes.get(tid, [0, 0, 0, 0])) if tid in prev_boxes else -1.0
            flag = " NEW" if tid in new else ""
            parts.append(f"[t{tid} conf={conf:.2f} iou_prev={io:.2f}{flag} box={box}]")
        flags = []
        if new:
            flags.append(f"新建={sorted(new)}")
        if lost:
            flags.append(f"丢失={sorted(lost)}")
        marker = "  <<<< " + " ".join(flags) if flags else ""
        print(f"帧{i:02d} @ {ts:4.1f}s  人数={len(persons)}  " + " ".join(parts) + marker)
        prev_boxes = {int(d["track_id"]): d["box"] for d in persons}
        prev_ids = ids

    print(f"\n==== 汇总：不同 track_id 总数={len(all_ids)} ({sorted(all_ids)})  "
          f"中途 ID 切换帧={switch_frames} ====")


if __name__ == "__main__":
    main()
