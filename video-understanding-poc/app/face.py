"""人脸识别分支（Phase 4 · Step 20）。

定位：和 `reid.py`（人形指纹）并列的**人脸身份**线索——给一帧画面，检测人脸、提
512 维归一化 embedding（人脸指纹），并评估人脸质量。**与向量库 / 融合解耦**：本模块只回答
"这帧里有哪些脸、各自的指纹和质量是什么"，不关心库怎么存、怎么和人形/步态融合（那是集成步）。

为什么要它：监控痛点是**人脸模糊**。清晰正脸时人脸是最强身份信号；糊脸/背身时降权、退人形。
本模块负责"把人脸用好"——最佳脸选择、多帧脸融合、质量加权，正是攻人脸模糊的核心。

后端（`FACE_BACKEND`，默认 insightface）：
  - **insightface**：InsightFace buffalo_l（SCRFD 检测 + ArcFace w600k_r50 识别，预训练）。
    业界标杆、与客户 UniFace 同源；embedding **512 维、已 L2 归一化**，直接可进 FAISS 余弦库。
    只加载 detection+recognition 两个子模型（砍掉 3D 关键点 / 性别年龄）以提速、省内存。

性能：模型首次加载约 20~30s（进程内一次性，常驻）；人脸推理 CPU 上较慢，故**稀疏调用**——
只在每条 track 的最佳帧上跑一次（认出即复用），不逐帧、不逐人每帧跑。
"""
from __future__ import annotations

import threading

import numpy as np

from .config import settings

_lock = threading.Lock()
_state: dict = {"backend": None, "model": None}

FACE_DIM = 512  # ArcFace 输出维度


# ---------------- 后端：InsightFace ----------------
def _load_insightface():
    """懒加载 InsightFace，只启用 detection + recognition（提速、省内存）。"""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name=settings.face_model,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(settings.face_det_size, settings.face_det_size))
    return {"app": app}


def _ensure_backend() -> None:
    """线程安全地懒加载并选定 backend（仅初始化一次）。"""
    if _state["backend"] is not None:
        return
    with _lock:
        if _state["backend"] is not None:
            return
        want = (settings.face_backend or "insightface").strip().lower()
        if want == "insightface":
            _state["model"] = _load_insightface()
            _state["backend"] = "insightface"
        else:
            raise ValueError(f"未知 FACE_BACKEND：{want}")


def active_backend() -> str:
    _ensure_backend()
    return _state["backend"]


# ---------------- 图像归一化（项目内统一用 PIL/data URI，InsightFace 要 BGR np）----------------
def _to_bgr(image) -> np.ndarray:
    """把 data URI / 纯 base64 / 字节 / PIL.Image / np(RGB) 统一成 InsightFace 要的 BGR ndarray。"""
    from PIL import Image

    if isinstance(image, np.ndarray):
        arr = image
        # 约定传入 np 为 RGB；转 BGR
        return arr[:, :, ::-1].copy() if arr.ndim == 3 else arr
    if isinstance(image, Image.Image):
        pil = image
    else:
        from .detector import _decode_image  # 复用：支持 data URI / base64 / bytes → PIL RGB

        pil = _decode_image(image)
    rgb = np.asarray(pil.convert("RGB"))
    return rgb[:, :, ::-1].copy()  # RGB → BGR


# ---------------- 核心：检测 + 提 embedding + 质量 ----------------
def _frontalness(kps: np.ndarray) -> float:
    """用 5 点关键点估正脸度（0~1，越大越正）。

    kps 顺序：左眼、右眼、鼻、左嘴角、右嘴角。正脸时鼻子在两眼中线上、左右大致对称；
    侧脸时鼻子明显偏向一侧。用"鼻子到两眼水平中点的偏移 / 两眼间距"度量偏转，越小越正。
    """
    if kps is None or len(kps) < 3:
        return 0.0
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_dist = float(np.hypot(*(right_eye - left_eye))) or 1.0
    offset = abs(nose[0] - eye_mid_x) / eye_dist  # 0=正脸，越大越偏
    return float(max(0.0, 1.0 - offset))


def _blur_var(bgr: np.ndarray, bbox) -> float:
    """脸框内灰度拉普拉斯方差（清晰度代理，越大越清晰）。无 cv2 依赖，用 numpy。"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    crop = bgr[y1:y2, x1:x2].astype(np.float32).mean(axis=2)  # 灰度
    lap = (
        -4 * crop
        + np.roll(crop, 1, 0) + np.roll(crop, -1, 0)
        + np.roll(crop, 1, 1) + np.roll(crop, -1, 1)
    )
    return float(lap[1:-1, 1:-1].var()) if crop.size > 9 else 0.0


def assess_quality(face: dict, bgr: np.ndarray | None = None) -> dict:
    """评估一张脸的质量（供"糊脸降权"与最佳脸选择）。

    综合：检测分 det_score、脸框面积、正脸度、清晰度(blur_var)。返回各分量 + 一个 0~1 的 quality
    标量 + 是否过门控 quality_ok（太小/太侧/太糊则不可信）。
    """
    bbox = face["bbox"]
    w = max(0.0, bbox[2] - bbox[0])
    h = max(0.0, bbox[3] - bbox[1])
    area = w * h
    det = float(face.get("det_score", 0.0))
    front = _frontalness(np.asarray(face.get("kps"))) if face.get("kps") is not None else 0.0
    blur = _blur_var(bgr, bbox) if bgr is not None else None

    size_ok = min(w, h) >= settings.face_min_size
    det_ok = det >= settings.face_min_det_score
    front_ok = front >= settings.face_min_frontalness
    blur_ok = (blur is None) or (blur >= settings.face_min_blur_var)
    quality_ok = bool(size_ok and det_ok and front_ok and blur_ok)

    # 0~1 质量标量：检测分 × 正脸度 × 面积饱和度（清晰度作为额外乘子，若可得）
    size_term = min(1.0, area / float(settings.face_ref_area)) if settings.face_ref_area > 0 else 1.0
    quality = det * (0.5 + 0.5 * front) * (0.5 + 0.5 * size_term)
    if blur is not None and settings.face_min_blur_var > 0:
        quality *= min(1.0, blur / (settings.face_min_blur_var * 4))

    reason = None
    if not size_ok:
        reason = "too_small"
    elif not det_ok:
        reason = "low_det_score"
    elif not front_ok:
        reason = "too_profile"
    elif not blur_ok:
        reason = "too_blurry"

    return {
        "det_score": round(det, 3),
        "area": int(area),
        "frontalness": round(front, 3),
        "blur_var": round(blur, 2) if blur is not None else None,
        "quality": round(float(quality), 4),
        "quality_ok": quality_ok,
        "reason": reason,
    }


def detect(image, with_quality: bool = True) -> list[dict]:
    """检测一帧里的所有人脸，返回每张脸的 bbox / kps / det_score / 512维归一化 embedding / 质量。

    Args:
        image: data URI / base64 / bytes / PIL.Image / np(RGB)（整帧）。
        with_quality: 是否附带 assess_quality 结果。

    Returns:
        list[dict]，每项：
          {bbox:[x1,y1,x2,y2], kps:[[x,y]*5], det_score, embedding(np 512 normed), quality{...}}
    """
    _ensure_backend()
    bgr = _to_bgr(image)
    app = _state["model"]["app"]
    faces = app.get(bgr)

    out: list[dict] = []
    for f in faces:
        emb = np.asarray(f.normed_embedding, dtype=np.float32)  # 已 L2 归一化
        item = {
            "bbox": [float(v) for v in f.bbox],
            "kps": np.asarray(f.kps).tolist() if getattr(f, "kps", None) is not None else None,
            "det_score": float(f.det_score),
            "embedding": emb,
        }
        if with_quality:
            item["quality"] = assess_quality(item, bgr)
        out.append(item)
    return out


def best_face(faces: list[dict]) -> dict | None:
    """从一条 track 的若干帧人脸里挑质量最高的一张（最佳脸选择，攻人脸模糊）。"""
    cand = [f for f in faces if f.get("embedding") is not None]
    if not cand:
        return None
    return max(cand, key=lambda f: (f.get("quality", {}) or {}).get("quality", 0.0))


def fuse_embeddings(faces: list[dict]) -> np.ndarray | None:
    """多帧人脸 embedding 的质量加权融合 → 一个更稳的 512 维向量（再归一化）。

    对应"多帧脸融合"：几帧糊不要紧，按质量加权平均压住单帧噪声。质量全 0 时退化为等权平均。
    """
    embs, weights = [], []
    for f in faces:
        e = f.get("embedding")
        if e is None:
            continue
        embs.append(np.asarray(e, dtype=np.float32))
        weights.append(float((f.get("quality", {}) or {}).get("quality", 0.0)))
    if not embs:
        return None
    w = np.asarray(weights, dtype=np.float32)
    if w.sum() <= 0:
        w = np.ones(len(embs), dtype=np.float32)
    fused = (np.stack(embs) * w[:, None]).sum(axis=0)
    n = float(np.linalg.norm(fused))
    return (fused / n).astype(np.float32) if n > 0 else fused.astype(np.float32)


def _iou_contain(face_box, person_box) -> float:
    """人脸框相对 person 框的"被包含度"= 交集面积 / 人脸框面积（人脸应落在人体内）。"""
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box
    ix1, iy1 = max(fx1, px1), max(fy1, py1)
    ix2, iy2 = min(fx2, px2), min(fy2, py2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    face_area = max(1e-6, (fx2 - fx1) * (fy2 - fy1))
    return inter / face_area


def associate_to_persons(faces: list[dict], person_dets: list[dict]) -> dict[int, dict]:
    """把检测到的人脸对到 person 的 track_id（用"人脸被人体框包含"的程度）。

    Args:
        faces: detect() 的输出。
        person_dets: [{box:[x1,y1,x2,y2], track_id}]（来自 /track 的 person 检测）。

    Returns:
        {track_id: face}，每个 track 取被包含度最高的那张脸（> 阈值才算）。
    """
    persons = [d for d in person_dets if d.get("label", "person") == "person" and d.get("track_id") is not None]
    result: dict[int, dict] = {}
    best_score: dict[int, float] = {}
    for face in faces:
        fb = face["bbox"]
        for p in persons:
            tid = int(p["track_id"])
            score = _iou_contain(fb, p["box"])
            if score >= settings.face_assoc_min_contain and score > best_score.get(tid, 0.0):
                best_score[tid] = score
                result[tid] = face
    return result


__all__ = [
    "FACE_DIM",
    "active_backend",
    "assess_quality",
    "associate_to_persons",
    "best_face",
    "detect",
    "fuse_embeddings",
]
