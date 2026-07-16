from __future__ import annotations

from ..keyframe import FrameMeta


def norm_box(box: list[float], img_w: int, img_h: int) -> list[float]:
    if not box or img_w <= 0 or img_h <= 0:
        return []
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    vals = [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]
    return [round(max(0.0, min(1.0, v)), 4) for v in vals]

def center_from_box(box: list[float], img_w: int, img_h: int) -> list[float]:
    if not box or img_w <= 0 or img_h <= 0:
        return []
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return [round(((x1 + x2) / 2.0) / img_w, 4), round(((y1 + y2) / 2.0) / img_h, 4)]

def direction(points: list[list[float]]) -> str:
    if len(points) < 2:
        return "unknown"
    sx, sy = points[0]
    ex, ey = points[-1]
    dx, dy = ex - sx, ey - sy
    parts: list[str] = []
    if abs(dx) >= 0.05:
        parts.append("right" if dx > 0 else "left")
    if abs(dy) >= 0.05:
        parts.append("down" if dy > 0 else "up")
    return "+".join(parts) if parts else "mostly_static"

def build_spatial_grounding(
    keyframe_indices: list[int],
    people: list[dict],
    tracks: dict[int, dict],
    identities: dict[int, dict],
    metas: list[FrameMeta],
    img_w: int,
    img_h: int,
) -> dict:
    """Build object-centric spatial evidence for LLM grounding: per-keyframe boxes + trajectory summary."""
    tid_to_subject = {tid: identities.get(tid, {}).get("subject_id") for tid in tracks}
    frames: list[dict] = []
    for idx in keyframe_indices:
        objects: list[dict] = []
        active = metas[idx].active_tracks if 0 <= idx < len(metas) else []
        for tid in active:
            t = tracks.get(tid)
            if not t:
                continue
            box = t["boxes"].get(idx)
            if not box:
                continue
            sid = tid_to_subject.get(tid)
            objects.append({
                "track_id": tid,
                "subject_id": sid,
                "label": f"subject#{sid}" if sid is not None else f"track#{tid}",
                "bbox": [round(float(v), 1) for v in box[:4]],
                "bbox_norm": norm_box(box, img_w, img_h),
                "center_norm": center_from_box(box, img_w, img_h),
                "decision": identities.get(tid, {}).get("decision"),
            })
        frames.append({
            "frame_index": idx,
            "timestamp": metas[idx].timestamp if 0 <= idx < len(metas) else None,
            "objects": objects,
        })

    trajectories: list[dict] = []
    for p in people:
        pts = p.get("trajectory") or []
        if not pts:
            continue
        show = [pts[0], pts[len(pts) // 2], pts[-1]] if len(pts) >= 3 else pts
        trajectories.append({
            "subject_id": p.get("subject_id"),
            "track_id": p.get("track_id"),
            "track_ids": p.get("source_track_ids") or [p.get("track_id")],
            "label": f"subject#{p.get('subject_id')}" if p.get("subject_id") is not None else f"track#{p.get('track_id')}",
            "path_sample": [[round(float(x), 4) for x in pt] for pt in show],
            "direction": direction([[float(x) for x in pt] for pt in pts]),
            "points": len(pts),
        })
    return {
        "image_size": [img_w, img_h],
        "coord": "bbox=[x1,y1,x2,y2] pixels; bbox_norm/center_norm normalized to 0..1 from top-left",
        "frames": frames,
        "trajectories": trajectories,
    }

def format_spatial_grounding(grounding: dict) -> str:
    """Compact text block appended to identity_context so the LLM can bind subjects to frame coordinates."""
    frames = grounding.get("frames") or []
    trajectories = grounding.get("trajectories") or []
    if not frames and not trajectories:
        return ""
    lines = [
        "【关键帧空间 grounding（坐标辅助理解，不要求模型重新检测）】",
        f"坐标约定：{grounding.get('coord', '')}",
    ]
    for fr in frames:
        objs = fr.get("objects") or []
        lines.append(f"- frame#{fr.get('frame_index')} @ {fr.get('timestamp')}：{len(objs)} objects")
        for obj in objs[:24]:
            lines.append(
                "  "
                f"{obj.get('label')} track={obj.get('track_id')} "
                f"bbox_norm={obj.get('bbox_norm')} center={obj.get('center_norm')}"
            )
        if len(objs) > 24:
            lines.append(f"  ... {len(objs) - 24} more objects omitted")
    if trajectories:
        lines.append("【跨帧轨迹摘要】")
        for tr in trajectories[:40]:
            lines.append(
                f"- {tr.get('label')} tracks={tr.get('track_ids')} "
                f"direction={tr.get('direction')} path={tr.get('path_sample')}"
            )
        if len(trajectories) > 40:
            lines.append(f"... {len(trajectories) - 40} more trajectories omitted")
    return "\n".join(lines)

__all__ = ["norm_box", "center_from_box", "direction", "build_spatial_grounding", "format_spatial_grounding"]
