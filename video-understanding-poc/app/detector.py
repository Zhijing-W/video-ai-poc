"""YOLO 物体检测层（Phase 2 · Step 5）。

定位：作为"廉价守门员"，对单帧画面做物体检测（bounding box + 类别 label + 置信度
confidence），为事件门控（Event Gate）提供判断依据 —— 避免无脑把每帧都丢给昂贵的
gpt-4o，只在命中关键物体/事件时才放行给 LLM。

模型：Ultralytics YOLO，默认 yolov8m（medium）。纯 CPU 可跑；首次调用自动下载权重。
模型名与置信度阈值可通过环境变量 YOLO_MODEL / YOLO_CONF 配置（见 config.py）。
"""
from __future__ import annotations

import base64
import io
import threading
import time

from .config import settings

_model = None
_model_lock = threading.Lock()


def _load_model():
    """懒加载 + 单例：进程内只加载一次权重（首次会自动下载）。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from ultralytics import YOLO

                _model = YOLO(settings.yolo_model)
    return _model


def class_names() -> list[str]:
    """返回当前模型的全部类别名（COCO 80 类），供目标编译时让 LLM 从合法类别里选。"""
    model = _load_model()
    names = model.names
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def _decode_image(image: str | bytes):
    """把 data URI / 纯 base64 字符串 / 原始字节解码成 PIL.Image。"""
    from PIL import Image

    if isinstance(image, str):
        if image.startswith("data:") and "," in image:
            image = image.split(",", 1)[1]
        raw = base64.b64decode(image)
    else:
        raw = image
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _predict(image: str | bytes, conf: float | None = None):
    """跑一次 YOLO 推理，返回原始 ultralytics 结果对象 + 元信息。

    抽出此函数是为了让"逐帧检测"（detect_objects）与"逐轨迹跟踪"（tracker.track_objects，
    Phase 3 · Step 11）共享同一次推理，避免对同一帧重复跑两遍 YOLO。

    Returns:
        (r, img_w, img_h, infer_ms)
          r: ultralytics Results（含 r.boxes / r.names / r.orig_img）
          infer_ms: 实测推理耗时（毫秒）
    """
    model = _load_model()
    img = _decode_image(image)
    img_w, img_h = img.size
    threshold = settings.yolo_conf if conf is None else conf

    t0 = time.perf_counter()
    results = model.predict(img, conf=threshold, verbose=False)
    infer_ms = round((time.perf_counter() - t0) * 1000, 1)
    return results[0], img_w, img_h, infer_ms


def _boxes_to_detections(r) -> tuple[list[dict], dict[str, int]]:
    """把 ultralytics Results.boxes 拍平成 [{label, confidence, box}] + counts。"""
    detections: list[dict] = []
    counts: dict[str, int] = {}
    names = r.names  # {类别下标: 类别名}
    for b in r.boxes:
        cls = int(b.cls[0])
        label = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
        confidence = round(float(b.conf[0]), 3)
        xyxy = [round(float(v), 1) for v in b.xyxy[0].tolist()]
        detections.append({"label": label, "confidence": confidence, "box": xyxy})
        counts[label] = counts.get(label, 0) + 1
    return detections, counts


def detect_objects(image: str | bytes, conf: float | None = None) -> dict:
    """对一张图做 YOLO 物体检测（无状态、逐帧）。

    Args:
        image: data URI（data:image/jpeg;base64,...）、纯 base64 字符串或原始字节。
        conf: 置信度阈值（None 则用 settings.yolo_conf）。

    Returns:
        dict:
          model: 使用的模型名
          infer_ms: 实测推理耗时（毫秒）—— 用来判断当前机型能不能扛 medium
          detections: [{label, confidence, box:[x1,y1,x2,y2]}]
          counts: {label: 数量}（给门控做"命中关键类别"判断用）
    """
    r, img_w, img_h, infer_ms = _predict(image, conf=conf)
    detections, counts = _boxes_to_detections(r)
    return {
        "model": settings.yolo_model,
        "infer_ms": infer_ms,
        "img_w": img_w,
        "img_h": img_h,
        "detections": detections,
        "counts": counts,
    }
