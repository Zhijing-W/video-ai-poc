"""图片处理工具：时间戳、base64 编解码与裁框。"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Sequence


def seconds_to_timestamp(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def image_to_data_uri(path: str | Path) -> str:
    path = Path(path)
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def decode_base64_image(image: str | bytes) -> bytes:
    if isinstance(image, bytes):
        return image
    payload = image.split(",", 1)[1] if image.startswith("data:") and "," in image else image
    return base64.b64decode(payload)


def save_data_uri_image(image: str, path: str | Path) -> bool:
    try:
        raw = decode_base64_image(image)
    except Exception:
        return False
    Path(path).write_bytes(raw)
    return True


def crop_box_region(img, box: Sequence[float], region: str = "whole"):
    x1, y1, x2, y2 = [int(value) for value in box[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    if region == "upper":
        y2 = y1 + max(1, int((y2 - y1) * 0.5))
    elif region == "lower":
        y1 = y2 - max(1, int((y2 - y1) * 0.5))
    elif region == "torso":
        height, width = y2 - y1, x2 - x1
        y1, y2 = y1 + int(height * 0.42), y1 + int(height * 0.82)
        x1, x2 = x1 + int(width * 0.18), x2 - int(width * 0.18)
        if x2 <= x1 or y2 <= y1:
            return None
    return img.crop((x1, y1, x2, y2))
