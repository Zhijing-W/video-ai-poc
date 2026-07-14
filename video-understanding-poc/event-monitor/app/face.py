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
from pathlib import Path

import numpy as np

from .config import settings

_lock = threading.Lock()
_state: dict = {"backend": None, "model": None}
_sr_state: dict = {"ready": False, "model": None, "error": None}

FACE_DIM = 512  # ArcFace 输出维度


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


# ---------------- 武器②：人脸超分（GFP-GAN，识别前预处理把糊脸拉清）----------------
def _patch_basicsr() -> None:
    """兜底修 basicsr 引用已被新版 torchvision 删除的 functional_tensor（保证可移植，不靠手改 venv）。"""
    try:
        import torchvision.transforms.functional as _F
        import torchvision.transforms as _T

        if not hasattr(_T, "functional_tensor"):
            import types
            import sys as _sys

            mod = types.ModuleType("torchvision.transforms.functional_tensor")
            mod.rgb_to_grayscale = _F.rgb_to_grayscale
            _sys.modules["torchvision.transforms.functional_tensor"] = mod
    except Exception:
        pass


def _gfpgan_weights_path() -> str:
    """把远程 GFPGAN 权重缓存到 appuser 持久 HOME，避免写只读 site-packages。"""
    configured = (settings.face_gfpgan_weights or "").strip()
    if configured and not configured.startswith(("http://", "https://")):
        return configured

    url = configured or "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    cache_dir = Path.home() / ".cache" / "gfpgan"
    cache_dir.mkdir(parents=True, exist_ok=True)
    from basicsr.utils.download_util import load_file_from_url

    return load_file_from_url(url=url, model_dir=str(cache_dir), progress=True)


def _ensure_superres():
    """懒加载 GFP-GAN 人脸增强器（首次会下权重）。失败记录 error 并降级为 no-op。"""
    if _sr_state["ready"] or _sr_state["error"] is not None:
        return _sr_state["model"]
    with _lock:
        if _sr_state["ready"] or _sr_state["error"] is not None:
            return _sr_state["model"]
        try:
            _patch_basicsr()
            from gfpgan import GFPGANer

            weights = _gfpgan_weights_path()
            _sr_state["model"] = GFPGANer(
                model_path=weights, upscale=2, arch="clean", channel_multiplier=2, bg_upsampler=None
            )
            _sr_state["ready"] = True
        except Exception as exc:  # noqa: BLE001
            _sr_state["error"] = f"{type(exc).__name__}: {exc}"
        return _sr_state["model"]


def superres_error() -> str | None:
    return _sr_state.get("error")


# ---------------- 武器③：AdaFace 识别后端（质量自适应，低清脸更强）----------------
_ada_state: dict = {"ready": False, "model": None, "error": None}


def _ensure_adaface():
    """懒加载 AdaFace（IR-101 WebFace12M）到 CPU。失败记录 error 并降级回 ArcFace。"""
    if _ada_state["ready"] or _ada_state["error"] is not None:
        return _ada_state["model"]
    with _lock:
        if _ada_state["ready"] or _ada_state["error"] is not None:
            return _ada_state["model"]
        try:
            import sys

            import torch

            root = settings.face_adaface_root
            if root not in sys.path:
                sys.path.insert(0, root)
            import net as _adanet  # AdaFace 仓库的 net.py

            model = _adanet.build_model(settings.face_adaface_arch)
            sd = torch.load(settings.face_adaface_weights, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # 兼容两种前缀：CVLface 封装是 'net.'，原版 AdaFace 是 'model.'
            if any(k.startswith("net.") for k in sd):
                sd = {k[4:]: v for k, v in sd.items() if k.startswith("net.")}
            elif any(k.startswith("model.") for k in sd):
                sd = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
            model.load_state_dict(sd, strict=True)
            model.eval()
            dev = "cuda" if _resolve_cuda(settings.face_device) else "cpu"
            model = model.to(dev)
            _ada_state["torch"] = torch
            _ada_state["model"] = model
            _ada_state["device"] = dev
            _ada_state["ready"] = True
        except Exception as exc:  # noqa: BLE001
            _ada_state["error"] = f"{type(exc).__name__}: {exc}"
        return _ada_state["model"]


def adaface_error() -> str | None:
    return _ada_state.get("error")


def _adaface_embed(bgr_face: np.ndarray) -> np.ndarray | None:
    """AdaFace 对一张已对齐人脸 BGR 图提 512 维归一化 embedding。"""
    m = _ensure_adaface()
    if m is None:
        return None
    try:
        import cv2

        torch = _ada_state["torch"]
        bgr = cv2.resize(bgr_face, (112, 112)) if bgr_face.shape[:2] != (112, 112) else bgr_face
        x = ((bgr.astype(np.float32) / 255.0) - 0.5) / 0.5  # BGR, [-1,1]（AdaFace 约定）
        t = torch.from_numpy(x.transpose(2, 0, 1)[None]).float().to(_ada_state.get("device", "cpu"))
        with torch.no_grad():
            out = m(t)
        feat = (out[0] if isinstance(out, (tuple, list)) else out).reshape(-1).cpu().numpy().astype(np.float32)
        n = float(np.linalg.norm(feat))
        return feat / n if n > 0 else feat
    except Exception as exc:  # noqa: BLE001
        _ada_state.setdefault("embed_error", str(exc))
        return None


def enhance(image, *, aligned: bool = False):
    """把一张（糊）人脸图增强/拉清（PIL→PIL）。供识别前预处理；不可用时原样返回。

    纯增强函数：**是否该超分由调用方（detect）按「糊才超分」门控决定**，这里不再自带尺寸门。
    aligned=True 表示输入已经按 5 点关键点对齐，GFPGAN 不再重复检测和对齐。
    """
    from PIL import Image

    if settings.face_superres in {"off", "none", ""}:
        return image
    pil = image if isinstance(image, Image.Image) else None
    if pil is None:
        return image
    sr = _ensure_superres()
    if sr is None:
        return image
    try:
        bgr = np.asarray(pil.convert("RGB"))[:, :, ::-1]
        _, restored_faces, restored = sr.enhance(
            bgr,
            has_aligned=aligned,
            only_center_face=not aligned,
            paste_back=not aligned,
        )
        if restored is None and restored_faces:
            restored = restored_faces[0]
        if restored is None:
            return image
        rgb = np.asarray(restored)[:, :, ::-1]
        return Image.fromarray(rgb)
    except Exception as exc:  # noqa: BLE001
        _sr_state.setdefault("enhance_error", str(exc))
        return image


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
    app.prepare(ctx_id=(0 if use_cuda else -1),
                det_size=(settings.face_det_size, settings.face_det_size))
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


def _pose_angles(kps: np.ndarray) -> tuple[float, float]:
    """用 5 点关键点近似估计 (yaw, pitch)，单位度。

    kps 顺序：左眼、右眼、鼻、左嘴角、右嘴角。
    - yaw（左右转）：鼻子相对两眼水平中点的偏移 / 半眼距 → ×90。0=正脸，±90≈全侧脸。
    - pitch（上下，负=低头）：鼻子在「眼线→嘴线」纵向区间里的相对位置。正脸约 0.5；
      俯拍低头时鼻子上移(靠近眼)→比例<0.5→pitch 为负；抬头则相反。
    真值角需 3D/PnP，这里是可用的粗略代理（客户阈值 yaw~80/pitch~45 较宽松，足够门控/分桶）。
    """
    if kps is None or len(kps) < 5:
        return 0.0, 0.0
    le, re, nose, lm, rm = [np.asarray(p, dtype=np.float32) for p in kps[:5]]
    eye_mid = (le + re) / 2.0
    mouth_mid = (lm + rm) / 2.0
    eye_dist = float(np.hypot(*(re - le))) or 1.0
    yaw = float(np.clip((nose[0] - eye_mid[0]) / (eye_dist / 2.0) * 90.0, -90.0, 90.0))
    face_h = float(mouth_mid[1] - eye_mid[1])
    if abs(face_h) < 1e-3:
        return yaw, 0.0
    nose_rel = (nose[1] - eye_mid[1]) / face_h            # ~0.5 正脸
    pitch = float(np.clip((nose_rel - 0.5) * 200.0, -90.0, 90.0))  # <0=低头
    return yaw, pitch


def _deep_fiqa_score(bgr: np.ndarray | None, bbox) -> float | None:
    """模糊第②路：开源深度 FIQA 模型预测（0~1，越大越清晰）。可插拔，默认 off 返回 None。"""
    if settings.face_fiqa_backend in {"off", "none", ""} or bgr is None:
        return None
    # 占位：实现时按 settings.face_fiqa_backend 加载 OFIQ/CR-FIQA 等，对脸区打分并归一化。
    return None


def assess_quality(face: dict, bgr: np.ndarray | None = None) -> dict:
    """评估一张脸质量，产出**分级类别**（对齐客户「人脸过滤」：主看模糊+角度）。

    - 模糊：拉普拉斯方差 `blur_var` + 关键点/检测置信度 `det_score`（+ 可插拔深度 FIQA）。
    - 角度：`yaw` / `pitch`（俯拍低头比抬头更不容忍）。
    返回 category ∈ {clear, marginal, poor}（唯一真源，产品降权 & 实验分桶都用它），
    并保留 quality 标量 / quality_ok（= 非 poor，向后兼容旧消费方）/ 各分量。
    """
    bbox = face["bbox"]
    w = max(0.0, bbox[2] - bbox[0])
    h = max(0.0, bbox[3] - bbox[1])
    area = w * h
    det = float(face.get("det_score", 0.0))
    kps = np.asarray(face["kps"]) if face.get("kps") is not None else None
    front = _frontalness(kps) if kps is not None else 0.0
    yaw, pitch = _pose_angles(kps) if kps is not None else (0.0, 0.0)
    blur = _blur_var(bgr, bbox) if bgr is not None else None
    fiqa = _deep_fiqa_score(bgr, bbox)  # 可插拔，默认 None

    # ---- 角度判级 ----
    yaw_bad = abs(yaw) >= settings.face_yaw_max
    pitch_bad = (pitch <= -settings.face_pitch_down_max) or (pitch >= settings.face_pitch_up_max)
    angle_clear = abs(yaw) <= settings.face_yaw_clear and abs(pitch) <= settings.face_pitch_clear

    # ---- 模糊判级（拉普拉斯 + 检测置信度，深度 FIQA 若有则纳入）----
    blur_bad = (blur is not None and blur < settings.face_min_blur_var) or (det < settings.face_min_det_score)
    blur_clear = (blur is None or blur >= settings.face_blur_clear_var) and (det >= settings.face_min_det_score)
    if fiqa is not None:
        blur_bad = blur_bad or fiqa < 0.3
        blur_clear = blur_clear and fiqa >= 0.6

    size_bad = min(w, h) < settings.face_min_size

    # ---- 合成类别：任一维不合格 → poor；两维都清晰且够大 → clear；其余 marginal ----
    if yaw_bad or pitch_bad or blur_bad or size_bad:
        category = "poor"
    elif angle_clear and blur_clear:
        category = "clear"
    else:
        category = "marginal"

    reason = None
    if size_bad:
        reason = "too_small"
    elif yaw_bad:
        reason = "yaw_extreme"
    elif pitch_bad:
        reason = "pitch_down" if pitch < 0 else "pitch_up"
    elif blur_bad:
        reason = "too_blurry"

    # 0~1 质量标量（保留：最佳脸选择/加权用）：检测分 × 正脸度 × 面积饱和 × 清晰度
    size_term = min(1.0, area / float(settings.face_ref_area)) if settings.face_ref_area > 0 else 1.0
    quality = det * (0.5 + 0.5 * front) * (0.5 + 0.5 * size_term)
    if blur is not None and settings.face_min_blur_var > 0:
        quality *= min(1.0, blur / (settings.face_min_blur_var * 4))

    return {
        "det_score": round(det, 3),
        "area": int(area),
        "frontalness": round(front, 3),
        "yaw": round(yaw, 1),
        "pitch": round(pitch, 1),
        "blur_var": round(blur, 2) if blur is not None else None,
        "fiqa": round(fiqa, 3) if fiqa is not None else None,
        "quality": round(float(quality), 4),
        "category": category,
        "quality_ok": category != "poor",  # 向后兼容：非 poor 视为可用
        "reason": reason,
    }


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


def detect(image, with_quality: bool = True, enhance_blurry: bool | None = None) -> list[dict]:
    """检测一帧里的所有人脸，返回每张脸的 bbox / kps / det_score / 512维归一化 embedding / 质量。

    Args:
        image: data URI / base64 / bytes / PIL.Image / np(RGB)（整帧）。
        with_quality: 是否附带 assess_quality 结果。
        enhance_blurry: 是否对糊脸做超分(武器②)后重提 embedding。None 时取 settings（超分非 off 即开）。

    Returns:
        list[dict]，每项：
          {bbox:[x1,y1,x2,y2], kps, det_score, embedding(512 normed), geom3d?, enhanced?, quality{...}}
    """
    from PIL import Image

    _ensure_backend()
    bgr = _to_bgr(image)
    app = _state["model"]["app"]
    faces = app.get(bgr)
    use_sr = (settings.face_superres not in {"off", "none", ""}) if enhance_blurry is None else enhance_blurry

    out: list[dict] = []
    for f in faces:
        emb = np.asarray(f.normed_embedding, dtype=np.float32)  # 已 L2 归一化
        item = {
            "bbox": [float(v) for v in f.bbox],
            "kps": np.asarray(f.kps).tolist() if getattr(f, "kps", None) is not None else None,
            "det_score": float(f.det_score),
            "embedding": emb,
        }
        # 先算质量（模糊+角度分级）——超分门控要用它判「糊不糊」，故提前到超分之前
        quality = assess_quality(item, bgr)
        # 武器①：3D-68 几何描述子（糊脸时的额外身份线索；纹理糊但几何还在）
        l3 = getattr(f, "landmark_3d_68", None)
        if settings.face_3d_cue and l3 is not None:
            geom = geometry_descriptor(np.asarray(l3, dtype=np.float32))
            if geom is not None:
                item["geom3d"] = geom
        aligned_bgr = None
        if getattr(f, "kps", None) is not None:
            try:
                from insightface.utils import face_align

                aligned_bgr = face_align.norm_crop(bgr, np.asarray(f.kps), image_size=112)
            except Exception:
                aligned_bgr = None
        # 武器②：糊脸超分 —— 只对**糊脸**做（对齐客户人脸过滤：糊才超分），且跳过极端侧脸
        # （超分救的是清晰度，救不了大侧脸；清晰脸也无需超分，省算力/免伪影）。
        enhanced_aligned_bgr = None
        if use_sr:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(bgr.shape[1], x2), min(bgr.shape[0], y2)
            blur_var = quality.get("blur_var")
            is_blurry = blur_var is not None and blur_var < settings.face_blur_clear_var
            angle_ok = abs(quality.get("yaw", 0.0)) < settings.face_yaw_max
            big_enough = min(x2 - x1, y2 - y1) >= 8 and aligned_bgr is not None
            if is_blurry and angle_ok and big_enough:
                aligned_rgb = Image.fromarray(aligned_bgr[:, :, ::-1])
                enh = enhance(aligned_rgb, aligned=True)
                if enh is not None and enh is not aligned_rgb:
                    enhanced_aligned_bgr = np.asarray(enh.convert("RGB"))[:, :, ::-1].copy()
                    new_emb = _reembed(enhanced_aligned_bgr)
                    if new_emb is not None:
                        item["embedding"] = new_emb
                        item["enhanced"] = True
        # 武器③：AdaFace 后端 —— 用 5 点对齐脸提质量自适应 embedding，替换 ArcFace（低清脸更强）
        if settings.face_rec_backend == "adaface" and aligned_bgr is not None:
            try:
                ada = _adaface_embed(enhanced_aligned_bgr if enhanced_aligned_bgr is not None else aligned_bgr)
                if ada is not None:
                    item["embedding"] = ada
                    item["rec_backend"] = "adaface"
            except Exception:
                pass
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
