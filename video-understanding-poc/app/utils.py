"""通用小工具。"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


def seconds_to_timestamp(seconds: float) -> str:
    """3 → '00:00:03'。"""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def image_to_data_uri(path: str | Path) -> str:
    """把本地图片读成 data URI（base64），用于 OpenAI 多模态 image_url。"""
    path = Path(path)
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"
