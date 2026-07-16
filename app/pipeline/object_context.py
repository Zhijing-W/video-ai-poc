from __future__ import annotations

from ..core.config import settings
from .spatial_context import direction


# COCO 标签 → 中文（仅常见的 LANE D 物体类，便于 LLM/人读；缺省回退英文原名）
_OBJ_LABEL_CN = {
    "backpack": "背包", "handbag": "手提包", "suitcase": "行李箱/箱子",
    "car": "汽车", "truck": "卡车", "bus": "公交车",
    "motorcycle": "摩托车", "bicycle": "自行车",
}

def build_object_context(
    object_tracks: dict[int, dict], win_idx: list[int], metas: list[FrameMeta],
    img_w: int, img_h: int,
) -> list[dict]:
    """汇总落在本窗内的非人物体轨迹：label、起止 frame#@ts、运动方向（场景级，非身份）。"""
    win_set = set(win_idx)
    objs: list[dict] = []
    for otid, ot in object_tracks.items():
        idxs = sorted(i for i in ot["boxes"] if i in win_set)
        if len(idxs) < settings.object_min_frames:  # 出现帧数太少 → 多为误检/ID 跳变，丢弃
            continue
        pts = [list(ot["centers"][i]) for i in idxs if i in ot["centers"]]
        first_i, last_i = idxs[0], idxs[-1]
        objs.append({
            "track_id": otid,
            "label": ot["label"],
            "label_cn": _OBJ_LABEL_CN.get(ot["label"], ot["label"]),
            "first_ts": metas[first_i].timestamp if 0 <= first_i < len(metas) else None,
            "last_ts": metas[last_i].timestamp if 0 <= last_i < len(metas) else None,
            "first_frame": first_i,
            "last_frame": last_i,
            "direction": direction(pts) if len(pts) >= 2 else "unknown",
            "frames_present": len(idxs),
            "conf": round(float(ot.get("max_conf", 0.0)), 3),
        })
    objs.sort(key=lambda o: (-o["frames_present"], o["first_frame"]))
    return objs

def format_object_context(objs: list[dict]) -> str:
    """把窗内物体摘要格式化成注入 LLM 的 object_context（场景级；含包裹品牌/logo 提示）。"""
    if not objs:
        return ""
    lines = ["【画面中的物体（YOLO 检测，场景级，非人物身份）】"]
    for o in objs[:20]:
        direction = o["direction"]
        motion = ("基本静止" if direction in ("mostly_static", "unknown")
                  else f"移动({direction})，疑似被搬动/取走/放下")
        lines.append(
            f"- {o['label_cn']}({o['label']}) track#{o['track_id']}："
            f"frame#{o['first_frame']}@{o['first_ts']} → frame#{o['last_frame']}@{o['last_ts']}"
            f"（共{o['frames_present']}帧）；{motion}"
        )
    if len(objs) > 20:
        lines.append(f"  ... 其余 {len(objs) - 20} 个物体省略")
    lines.append(
        "说明：以上为画面中的**物体**（包裹/行李/车辆等场景线索，frame# 与人物 grounding 同坐标系），"
        "用于理解放下、取走、搬运、到达、离开等事件；**若疑似快递/包裹，请结合关键帧画面识别其品牌或 "
        "logo（如 Amazon / UPS / FedEx，OCR 读文字、你看图认 logo）**。请勿把物体当作人物身份。"
    )
    return "\n".join(lines)

__all__ = ["build_object_context", "format_object_context"]
