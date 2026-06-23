"""廉价属性校验（Phase 2 · LLM监工级联）。

定位：YOLO 只认"物体类别"（COCO 80 类，**没有颜色**）。当报警目标是"红色汽车"这类
"类别 + 属性"时，YOLO 给出 car 的框，由本模块在框内做**廉价 CV 颜色判断**补上 "red"。
纯 PIL/numpy，无 LLM、无训练，CPU 毫秒级 —— 让 YOLO 自动巡航期不必每帧调大模型。

注意：这是启发式（heuristic），对"主色明显"的目标够用；复杂属性（姿态/动作/持物）
YOLO+颜色无法判断，应在目标编译阶段标记 can_yolo_handle=false，回落到每帧 LLM。
"""
from __future__ import annotations

import colorsys
from collections import Counter

# 颜色名 -> 中文（仅用于展示）
COLOR_ZH = {
    "red": "红", "orange": "橙", "yellow": "黄", "green": "绿", "blue": "蓝",
    "purple": "紫", "pink": "粉", "white": "白", "gray": "灰", "black": "黑", "brown": "棕",
}

# 中性色（灰/白/黑）：背景多为中性，找"主体颜色"（如衣服）时应让位给有彩色。
_NEUTRAL = {"white", "gray", "black"}


def _classify_rgb(r: float, g: float, b: float) -> str:
    """把一个平均 RGB（0~255）分类成颜色名。基于 HSV 阈值。"""
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hue = h * 360.0
    if v < 0.2:
        return "black"
    if s < 0.15:
        return "white" if v > 0.8 else "gray"
    # 棕色：暖色调且偏暗
    if (hue < 45 or hue >= 345) and v < 0.5 and s > 0.3:
        return "brown"
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 45:
        return "orange"
    if hue < 65:
        return "yellow"
    if hue < 170:
        return "green"
    if hue < 255:
        return "blue"
    if hue < 290:
        return "purple"
    return "pink"


def _crop_region(img, box: list[float], region: str):
    """按 region 取框的子区域：
      whole  整框
      upper  上半（头/上身）
      lower  下半
      torso  躯干带（纵向 42%~82% + 横向各收 18%）—— person 取衣服色、避开人脸肤色与背景。
    """
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    if region == "upper":
        y2 = y1 + max(1, int((y2 - y1) * 0.5))
    elif region == "lower":
        y1 = y2 - max(1, int((y2 - y1) * 0.5))
    elif region == "torso":
        h, w = y2 - y1, x2 - x1
        y1, y2 = y1 + int(h * 0.42), y1 + int(h * 0.82)
        x1, x2 = x1 + int(w * 0.18), x2 - int(w * 0.18)
        if x2 <= x1 or y2 <= y1:
            return None
    return img.crop((x1, y1, x2, y2))


def dominant_color(img, box: list[float], region: str = "whole") -> str | None:
    """返回框内（指定区域）的主色名。

    关键改进（修「肤色/背景把主体颜色带偏」）：
      1. 不再对像素**求均值**再分类（均值会把"蓝衣+肤色+背景"混成脏色而误判），
         改为给**每个像素分类**再取**直方图众数**，对混合像素鲁棒得多。
      2. 对中性背景（灰/白）降权：只要有彩色像素占比≥25% 就采信它（衣服往往不是
         画面里像素最多的部分），否则才回落到整体众数（黑椅子/白墙等真·中性主体）。
      3. 配合 region="torso"（person）从躯干取色，避开人脸肤色。
    """
    crop = _crop_region(img, box, region)
    if crop is None:
        return None
    # 先收掉边缘 15%（削弱框边背景），再放大成 40×40 取样
    w, h = crop.size
    cx1, cy1, cx2, cy2 = int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)
    if cx2 > cx1 and cy2 > cy1:
        crop = crop.crop((cx1, cy1, cx2, cy2))
    crop = crop.resize((40, 40))
    pixels = list(crop.getdata())
    if not pixels:
        return None

    names = [_classify_rgb(p[0], p[1], p[2]) for p in pixels]
    cnt = Counter(names)
    total = len(names)
    chromatic = [(c, n) for c, n in cnt.items() if c not in _NEUTRAL]
    if chromatic:
        chromatic.sort(key=lambda t: t[1], reverse=True)
        c, n = chromatic[0]
        if n >= total * 0.25:           # 有彩色占比够高 → 采信主体颜色，压过灰白背景
            return c
    return cnt.most_common(1)[0][0]      # 否则用整体众数（中性主体或纯背景）


def color_matches(img, box: list[float], target_color: str, region: str = "whole") -> tuple[bool, str | None]:
    """框内主色是否匹配目标颜色。返回 (是否匹配, 实测主色名)。"""
    detected = dominant_color(img, box, region)
    if detected is None:
        return False, None
    target = (target_color or "").strip().lower()
    # 允许中文颜色输入
    zh2en = {v: k for k, v in COLOR_ZH.items()}
    target = zh2en.get(target, target)
    return detected == target, detected
