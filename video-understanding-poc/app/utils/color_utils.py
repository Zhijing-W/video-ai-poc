"""廉价 HSV 主色判断工具。"""
from __future__ import annotations

import colorsys
from collections import Counter

from .image_utils import crop_box_region

COLOR_ZH = {
    "red": "红",
    "orange": "橙",
    "yellow": "黄",
    "green": "绿",
    "blue": "蓝",
    "purple": "紫",
    "pink": "粉",
    "white": "白",
    "gray": "灰",
    "black": "黑",
    "brown": "棕",
}
_NEUTRAL = {"white", "gray", "black"}


def _classify_rgb(r: float, g: float, b: float) -> str:
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hue = h * 360.0
    if v < 0.2:
        return "black"
    if s < 0.15:
        return "white" if v > 0.8 else "gray"
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


def dominant_color(img, box: list[float], region: str = "whole") -> str | None:
    crop = crop_box_region(img, box, region)
    if crop is None:
        return None
    width, height = crop.size
    cx1, cy1 = int(width * 0.15), int(height * 0.15)
    cx2, cy2 = int(width * 0.85), int(height * 0.85)
    if cx2 > cx1 and cy2 > cy1:
        crop = crop.crop((cx1, cy1, cx2, cy2))
    crop = crop.resize((40, 40))
    pixels = list(crop.getdata())
    if not pixels:
        return None

    color_names = [_classify_rgb(pixel[0], pixel[1], pixel[2]) for pixel in pixels]
    counts = Counter(color_names)
    total = len(color_names)
    chromatic = [(color, count) for color, count in counts.items() if color not in _NEUTRAL]
    if chromatic:
        chromatic.sort(key=lambda item: item[1], reverse=True)
        color, count = chromatic[0]
        if count >= total * 0.25:
            return color
    return counts.most_common(1)[0][0]


def color_matches(
    img,
    box: list[float],
    target_color: str,
    region: str = "whole",
) -> tuple[bool, str | None]:
    detected = dominant_color(img, box, region)
    if detected is None:
        return False, None
    target = (target_color or "").strip().lower()
    zh_to_en = {zh: en for en, zh in COLOR_ZH.items()}
    target = zh_to_en.get(target, target)
    return detected == target, detected
