"""人脸识别分支（Phase 4 · Step 20）。

定位：和 `body_reid.py`（人形指纹）并列的**人脸身份**线索——给一帧画面，检测人脸、提
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
from .identity.face.adaface import embed as _adaface_embed
from .identity.face.adaface import load_error as _adaface_error
from .identity.face.quality import assess_quality as _assess_quality_impl
from .identity.face.quality import deep_fiqa_score as _deep_fiqa_score
from .identity.face.super_resolution import _ensure_superres as _ensure_superres_impl
from .identity.face.super_resolution import enhance as _enhance_impl
from .identity.face.super_resolution import superres_error as _superres_error_impl

_lock = threading.Lock()
_state: dict = {"backend": None, "model": None}

FACE_DIM = 512  # ArcFace 输出维度


class _FaceRecord(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _resolve_cuda(device: str, ort: bool = False) -> bool:
    """把 auto/cuda/cpu 解析成"是否用 CUDA"。ort=True 时按 onnxruntime 的可用 provider 判断
    (InsightFace 走 onnxruntime)；否则按 torch.cuda 判断 (AdaFace 走 torch)。"""
    d = (device or "auto").strip().lower()
    if d == "cpu":
        return False
    if ort:
        try:
            import onnxruntime as _o
            return "CUDAExecutionProvider" in _o.get_available_providers()
        except Exception:
            return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def adaface_error() -> str | None:
    return _adaface_error()


# ---------------- 后端：InsightFace ----------------
def _load_insightface():
    """懒加载 InsightFace：detection + recognition，并按配置可选启用 3D-68 几何 cue。
    按 settings.face_device 选 CPU / CUDA(onnxruntime CUDAExecutionProvider)。"""
    from insightface.app import FaceAnalysis

    modules = ["detection", "recognition"]
    if settings.face_3d_cue:
        modules.insert(1, "landmark_3d_68")  # 武器①：3D 面部几何（糊脸兜底）
    use_cuda = _resolve_cuda(settings.face_device, ort=True)
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda
                 else ["CPUExecutionProvider"])
    app = FaceAnalysis(
        name=settings.face_model,
        allowed_modules=modules,
        providers=providers,
    )
    app.prepare(
        ctx_id=(0 if use_cuda else -1),
        det_thresh=settings.face_min_det_score,
        det_size=(settings.face_det_size, settings.face_det_size),
    )
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

def _reembed(bgr_face: np.ndarray) -> np.ndarray | None:
    """对一张已裁好的人脸 BGR 图，用 recognition 模型重提归一化 embedding（超分后重算用）。"""
    app = _state["model"]["app"]
    rec = None
    for m in app.models.values():
        if getattr(m, "taskname", "") == "recognition":
            rec = m
            break
    if rec is None:
        return None
    try:
        feat = rec.get_feat(bgr_face).reshape(-1).astype(np.float32)
    except Exception:
        return None
    n = float(np.linalg.norm(feat))
    return feat / n if n > 0 else feat


def embed_aligned_face(bgr_face: np.ndarray, backend: str) -> np.ndarray | None:
    """对已按 5 点关键点对齐的人脸提指定后端 embedding。"""
    _ensure_backend()
    name = backend.strip().lower()
    if name == "arcface":
        return _reembed(bgr_face)
    if name == "adaface":
        return _adaface_embed(bgr_face)
    raise ValueError(f"未知人脸识别后端：{backend}")


def _align_face(bgr: np.ndarray, kps: np.ndarray | None) -> np.ndarray | None:
    if kps is None:
        return None
    try:
        from insightface.utils import face_align

        return face_align.norm_crop(bgr, np.asarray(kps), image_size=112)
    except Exception:
        return None


def align_face(image, kps) -> np.ndarray | None:
    """使用产品相同的五点对齐逻辑，返回112×112 BGR人脸。"""
    return _align_face(_to_bgr(image), np.asarray(kps, dtype=np.float32) if kps is not None else None)


def _detect_face_candidates(app, bgr: np.ndarray) -> list[dict]:
    """仅调用SCRFD，避免质量门控前提前运行ArcFace识别模型。"""
    bboxes, kpss = app.det_model.detect(bgr, max_num=0, metric="default")
    candidates = []
    for index in range(bboxes.shape[0]):
        kps = np.asarray(kpss[index], dtype=np.float32) if kpss is not None else None
        candidates.append(
            {
                "bbox": [float(value) for value in bboxes[index, :4]],
                "kps": kps.tolist() if kps is not None else None,
                "det_score": float(bboxes[index, 4]),
                "_kps_array": kps,
            }
        )
    return candidates


def _attach_optional_geometry(app, bgr: np.ndarray, item: dict) -> None:
    if not settings.face_3d_cue:
        return
    try:
        face_obj = _FaceRecord(
            bbox=np.asarray(item["bbox"], dtype=np.float32),
            kps=item.get("_kps_array"),
            det_score=float(item["det_score"]),
        )
        for model in app.models.values():
            if getattr(model, "taskname", "") == "landmark_3d_68":
                model.get(bgr, face_obj)
                landmarks = getattr(face_obj, "landmark_3d_68", None)
                if landmarks is not None:
                    descriptor = geometry_descriptor(np.asarray(landmarks, dtype=np.float32))
                    if descriptor is not None:
                        item["geom3d"] = descriptor
                return
    except Exception as exc:
        item["geom3d_error"] = f"{type(exc).__name__}: {exc}"


def detect(
    image,
    with_quality: bool = True,
    enhance_blurry: bool | None = None,
    with_identity: bool = True,
    with_geometry: bool = True,
) -> list[dict]:
    """先检测和评估质量，仅对can_match=true的人脸提身份embedding。

    Args:
        image: data URI / base64 / bytes / PIL.Image / np(RGB)（整帧）。
        with_quality: 是否附带 assess_quality 结果。
        enhance_blurry: 是否对糊脸做超分(武器②)后重提 embedding。None 时取 settings（超分非 off 即开）。
        with_identity: 是否在质量评估后调用ArcFace/AdaFace。False用于只冻结检测与质量输入集。
        with_geometry: 是否计算可选3D-68几何描述；实验prepare可关闭以保持纯检测/质量链路。

    Returns:
        list[dict]，每项：
          {bbox, kps, det_score, quality, embedding?, geom3d?, enhanced?}
    """
    from PIL import Image

    _ensure_backend()
    bgr = _to_bgr(image)
    app = _state["model"]["app"]
    candidates = _detect_face_candidates(app, bgr)
    use_sr = (settings.face_superres not in {"off", "none", ""}) if enhance_blurry is None else enhance_blurry

    out: list[dict] = []
    for candidate in candidates:
        item = dict(candidate)
        aligned_bgr = _align_face(bgr, item.get("_kps_array"))
        quality = assess_quality(item, bgr, aligned_bgr=aligned_bgr)
        if with_geometry:
            _attach_optional_geometry(app, bgr, item)

        # 武器②：直接消费统一质量评估的can_superres，不再重复维护另一套角度/模糊门控。
        enhanced_aligned_bgr = None
        if use_sr and quality.get("can_superres"):
            x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(bgr.shape[1], x2), min(bgr.shape[0], y2)
            big_enough = min(x2 - x1, y2 - y1) >= 8 and aligned_bgr is not None
            if big_enough:
                item["superres_attempted"] = True
                aligned_rgb = Image.fromarray(aligned_bgr[:, :, ::-1])
                enh = enhance(aligned_rgb, aligned=True)
                if enh is not None and enh is not aligned_rgb:
                    enhanced_aligned_bgr = np.asarray(enh.convert("RGB"))[:, :, ::-1].copy()
                    fiqa_after = _deep_fiqa_score(enhanced_aligned_bgr)
                    if fiqa_after is not None:
                        quality["fiqa_after_superres"] = round(fiqa_after, 3)
                        before = quality.get("fiqa")
                        if before is not None:
                            quality["fiqa_delta_superres"] = round(fiqa_after - float(before), 3)
                    item["enhanced"] = True

        # 质量分类完成后才调用身份模型。CR-FIQA与产品ArcFace/AdaFace的embedding互不混用。
        identity_input = enhanced_aligned_bgr if enhanced_aligned_bgr is not None else aligned_bgr
        if with_identity and quality.get("can_match") and identity_input is not None:
            try:
                identity_embedding = embed_aligned_face(identity_input, settings.face_rec_backend)
                if identity_embedding is not None:
                    item["embedding"] = identity_embedding
                    item["rec_backend"] = settings.face_rec_backend
            except Exception as exc:
                item["identity_error"] = f"{type(exc).__name__}: {exc}"

        item.pop("_kps_array", None)
        if with_quality:
            item["quality"] = quality
        out.append(item)
    return out


def geometry_descriptor(landmarks_3d: np.ndarray) -> np.ndarray | None:
    """从 68 个 3D 关键点算一个**姿态/尺度不变**的面部几何描述子（L2 归一化）。

    思路：把 3D 点云中心化、按尺度归一化，再用关键点对之间的归一化距离（脸的"骨架结构"——
    颧骨宽、鼻梁高、下巴长等），这些在**纹理糊掉后依然稳定**，是攻人脸模糊的几何线索。
    与 ArcFace 外观向量互补：糊脸时几何撑住身份。
    """
    if landmarks_3d is None or landmarks_3d.shape[0] < 68:
        return None
    pts = landmarks_3d.astype(np.float32).copy()
    pts -= pts.mean(axis=0, keepdims=True)              # 中心化（平移不变）
    scale = float(np.sqrt((pts ** 2).sum(axis=1).mean()))
    if scale <= 1e-6:
        return None
    pts /= scale                                        # 尺度归一化
    # 选若干结构性关键点对（轮廓/眼/鼻/嘴/下巴），用点对距离刻画几何结构
    idx_pairs = [
        (36, 45),  # 两眼外角（脸宽）
        (39, 42),  # 两眼内角
        (31, 35),  # 鼻翼宽
        (27, 33),  # 鼻梁长
        (48, 54),  # 嘴角宽
        (51, 57),  # 上下唇
        (0, 16),   # 颧骨/脸颊最宽
        (8, 27),   # 下巴到鼻根（脸长）
        (17, 26),  # 两眉外端
        (21, 22),  # 两眉内端
        (3, 13),   # 下颌宽
        (30, 8),   # 鼻尖到下巴
    ]
    feats = []
    for a, b in idx_pairs:
        feats.append(float(np.linalg.norm(pts[a] - pts[b])))
    # 再补几个深度差（Z 轴，体现立体度：鼻梁凸起、眼窝深度）
    feats.append(float(pts[30, 2] - pts[27, 2]))        # 鼻尖 vs 鼻根 深度
    feats.append(float(pts[8, 2] - pts[30, 2]))         # 下巴 vs 鼻尖 深度
    feats.append(float(pts[0, 2] - pts[30, 2]))         # 脸颊 vs 鼻尖 深度（侧凸）
    vec = np.asarray(feats, dtype=np.float32)
    n = float(np.linalg.norm(vec))
    return (vec / n).astype(np.float32) if n > 0 else vec


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
        q = f.get("quality", {}) or {}
        weights.append(max(0.0, min(1.0, float(q.get("match_weight", q.get("quality", 0.0))))))
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



def superres_error() -> str | None:
    return _superres_error_impl()


def _ensure_superres():
    """兼容实验脚本：实际实现已迁到 identity.face.super_resolution。"""
    return _ensure_superres_impl()


def enhance(image, *, aligned: bool = False):
    return _enhance_impl(image, aligned=aligned)


def assess_quality(
    face: dict,
    bgr: np.ndarray | None = None,
    aligned_bgr: np.ndarray | None = None,
) -> dict:
    return _assess_quality_impl(face, bgr=bgr, aligned_bgr=aligned_bgr)


__all__ = [
    "FACE_DIM",
    "active_backend",
    "align_face",
    "adaface_error",
    "assess_quality",
    "associate_to_persons",
    "best_face",
    "detect",
    "embed_aligned_face",
    "enhance",
    "fuse_embeddings",
    "geometry_descriptor",
    "superres_error",
]
