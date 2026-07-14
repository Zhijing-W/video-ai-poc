"""场景文字识别（OCR）—— LANE D 的核心子能力（Phase 4 · Step 29）。

定位：和"人物 provider（人脸/人形/步态→subject_id）"**平级**的**场景级 provider**。
它回答的不是"谁"，而是"画面里**写了什么字**"——监控时间戳叠层、车牌、包裹单号、门牌等。

关键边界（务必牢记）：OCR 读出的文字是**场景级信号**，**不指向某个人**，因此：
  - **绝不**碰 gallery / subject_id / 三路融合；
  - 输出汇成一段 `scene_context` 文本，**只在最后喂 LLM 时**与人物身份**并列**注入，
    由 LLM 自己决定"14:03 这个时间该配给哪个事件"。

引擎：默认 **RapidOCR**（onnxruntime，内置 PP-OCRv4 检测+识别模型，CPU 友好、Windows 安装干净，
与 PaddleOCR 同源模型，精度基本一致）。设计为**可插拔**（`OCR_BACKEND`）：上云可切 PaddleOCR
server 模型追更高精度，接口不变。

复用：与人物链路**同一批关键帧**（不重抽帧）。可选 ROI 只在固定角落（如时间戳区）跑，省算力。
"""
from __future__ import annotations

import threading

import numpy as np
from PIL import Image

from .config import settings

_lock = threading.Lock()
_state: dict = {"ready": False, "engine": None, "backend": None, "error": None}


# ---------------- 懒加载引擎 ----------------
def _ensure() -> bool:
    """线程安全地懒加载 OCR 引擎。失败则记录 error 并返回 False（不致命，跳过 OCR）。"""
    if _state["ready"]:
        return True
    if _state["error"] is not None:
        return False
    with _lock:
        if _state["ready"]:
            return True
        if _state["error"] is not None:
            return False
        backend = (settings.ocr_backend or "rapidocr").strip().lower()
        try:
            if backend in {"rapidocr", "rapid", "ppocr", "paddle_onnx"}:
                from rapidocr_onnxruntime import RapidOCR

                _state["engine"] = RapidOCR()
                _state["backend"] = "rapidocr"
            elif backend in {"paddleocr", "paddle"}:
                from paddleocr import PaddleOCR

                _state["engine"] = PaddleOCR(
                    use_angle_cls=True, lang=settings.ocr_lang, show_log=False
                )
                _state["backend"] = "paddleocr"
            else:
                raise ValueError(f"未知 OCR_BACKEND：{backend!r}（可选 rapidocr / paddleocr）")
            _state["ready"] = True
            return True
        except Exception as exc:  # noqa: BLE001
            _state["error"] = f"{type(exc).__name__}: {exc}"
            return False


def load_error() -> str | None:
    """返回引擎加载/运行错误（供上层回显"为什么没跑 OCR"）。"""
    return _state["error"]


def active_backend() -> str | None:
    return _state.get("backend")


# ---------------- 图像归一化 ----------------
def _to_bgr(image) -> np.ndarray | None:
    """把 路径 / PIL.Image / ndarray 统一成 OpenCV BGR ndarray。"""
    try:
        if isinstance(image, str):
            pil = Image.open(image).convert("RGB")
            arr = np.asarray(pil)
        elif isinstance(image, Image.Image):
            arr = np.asarray(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            arr = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
            return arr[:, :, ::-1].copy() if arr.shape[2] == 3 else None
        else:
            return None
        return arr[:, :, ::-1].copy()  # RGB → BGR
    except Exception:  # noqa: BLE001
        return None


def _apply_roi(bgr: np.ndarray, roi):
    """按归一化 ROI [x1,y1,x2,y2]（0~1）裁剪；返回 (裁剪图, (ox,oy) 偏移)。roi=None 则原图。"""
    if not roi:
        return bgr, (0, 0)
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = roi
    px1, py1 = max(0, int(x1 * w)), max(0, int(y1 * h))
    px2, py2 = min(w, int(x2 * w)), min(h, int(y2 * h))
    if px2 <= px1 or py2 <= py1:
        return bgr, (0, 0)
    return bgr[py1:py2, px1:px2].copy(), (px1, py1)


# ---------------- 单帧 OCR ----------------
def read_frame(image, roi=None, min_conf: float | None = None) -> list[dict]:
    """对一帧做 OCR，返回 [{text, box:[x1,y1,x2,y2], conf}]（按置信度降序）。

    Args:
        image: 路径 / PIL.Image / BGR-or-RGB ndarray。
        roi: 可选归一化裁剪区 [x1,y1,x2,y2]（0~1）；如时间戳常在角落，只 OCR 该区省算力、增准。
        min_conf: 置信度下限（默认取 settings.ocr_min_conf）。
    """
    if not _ensure():
        return []
    thr = settings.ocr_min_conf if min_conf is None else min_conf
    bgr = _to_bgr(image)
    if bgr is None:
        return []
    crop, (ox, oy) = _apply_roi(bgr, roi if roi is not None else settings.ocr_roi)

    out: list[dict] = []
    try:
        if _state["backend"] == "rapidocr":
            result, _ = _state["engine"](crop)
            for item in result or []:
                quad, text, score = item[0], item[1], float(item[2])
                if not text or score < thr:
                    continue
                out.append({"text": str(text).strip(), "box": _quad_to_box(quad, ox, oy),
                            "conf": round(score, 3)})
        else:  # paddleocr
            res = _state["engine"].ocr(crop, cls=True)
            for line in (res[0] if res else []) or []:
                quad, (text, score) = line[0], line[1]
                if not text or float(score) < thr:
                    continue
                out.append({"text": str(text).strip(), "box": _quad_to_box(quad, ox, oy),
                            "conf": round(float(score), 3)})
    except Exception as exc:  # noqa: BLE001 —— OCR 失败不致命
        _state.setdefault("frame_error", str(exc))
        return []
    out.sort(key=lambda d: d["conf"], reverse=True)
    return out


def _quad_to_box(quad, ox: int, oy: int) -> list[int]:
    """四点多边形 → 轴对齐外接框 [x1,y1,x2,y2]，并加回 ROI 偏移。"""
    xs = [float(p[0]) + ox for p in quad]
    ys = [float(p[1]) + oy for p in quad]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


# ---------------- 按窗聚合成 scene_context ----------------
def format_scene_context(per_frame: list[dict], max_frame_lines: int = 24) -> str:
    """把一个事件窗内若干关键帧的 OCR 结果聚成注入 LLM 的场景文字上下文。

    **与空间 grounding 对齐**：逐关键帧用同样的 `frame#N @ ts` 锚点（grounding 也是这套），
    让 LLM 能把"frame#N 这一刻谁在哪（grounding）"和"画面时钟/车牌读数（OCR）"逐帧对上。
    贯穿多帧不变的文字（如摄像头名 CAM2/地点）单列一次"固定文字"，省 token、降噪。

    Args:
        per_frame: [{"frame_index": int, "timestamp": 秒(float)/文本, "texts": read_frame 返回}]，按时间序。
    """
    def _norm(s: str) -> str:
        return "".join(s.split()).lower()

    # 逐帧去重收集；统计每条文字在多少帧出现（判"固定文字"）
    frame_texts: list[tuple] = []  # (frame_index, timestamp, [(key, display)])
    occur: dict[str, int] = {}
    n_text_frames = 0
    for fr in per_frame:
        texts: list[tuple] = []
        seen_in_frame: set[str] = set()
        for t in fr.get("texts", []):
            txt = (t.get("text") or "").strip()
            k = _norm(txt)
            if not txt or k in seen_in_frame:
                continue
            seen_in_frame.add(k)
            texts.append((k, txt))
        if texts:
            n_text_frames += 1
            for k, _ in texts:
                occur[k] = occur.get(k, 0) + 1
        frame_texts.append((fr.get("frame_index"), fr.get("timestamp"), texts))

    if n_text_frames == 0:
        return ""

    # 固定文字：在 ≥60%（且≥2）的有字帧里都出现 → 视为场景常驻（摄像头名/地点等）
    static_keys = {k for k, c in occur.items()
                   if n_text_frames >= 2 and c >= max(2, int(0.6 * n_text_frames))}
    static_display: dict[str, str] = {}
    for _, _, texts in frame_texts:
        for k, txt in texts:
            if k in static_keys and k not in static_display:
                static_display[k] = txt

    lines = ["【画面文字（OCR，逐关键帧；场景级，不代表任何人的身份）】"]
    if static_display:
        lines.append("固定文字（贯穿多帧，如摄像头名/地点）：" + " | ".join(static_display.values()))

    # 逐帧"变化文字"（多为时间戳），用 frame#N @ ts 锚点，和 grounding 同一坐标系
    changing_lines: list[str] = []
    for idx, ts, texts in frame_texts:
        changing = [txt for k, txt in texts if k not in static_keys]
        if not changing:
            continue
        anchor = f"frame#{idx} @ {ts}" if idx is not None else (str(ts) if ts is not None else "")
        changing_lines.append(f"- {anchor}：" + " | ".join(changing))
    if changing_lines:
        lines.append("逐帧文字（多为时间戳/动态信息，frame#与 grounding 一一对应）：")
        lines.extend(changing_lines[:max_frame_lines])
        if len(changing_lines) > max_frame_lines:
            lines.append(f"  ... 其余 {len(changing_lines) - max_frame_lines} 帧文字省略")

    lines.append(
        "说明：以上为画面中出现的文字（时间戳/车牌/单号/摄像头名等），可据此补全事件的**时间/物件**"
        "线索、并与同一 frame# 的人物位置对齐；请**不要**把这些文字当作人物身份，也不要据此推断『谁』。"
    )
    return "\n".join(lines)


__all__ = ["read_frame", "format_scene_context", "load_error", "active_backend"]
