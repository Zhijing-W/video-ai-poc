"""步态识别分支（Phase 4 · Step 27）—— SkeletonGait++（OpenGait, GREW 权重）。

定位：和 `face.py`(人脸)、`reid.py`(人形) 并列的**第三路身份信号**。无脸/背身/远景时，
人脸和人形都弱，步态是兜底——靠"走路姿态"认人。客户三角(人脸+人形+步态)的最后一块。

模型：**SkeletonGait++**(AAAI'24，OpenGait)。双模态融合，每帧吃 3 通道 [3,64,44]：
  - 通道 0–1：姿态**热力图**(肢体图 + 关键点图，从 17 点 COCO 姿态渲染)
  - 通道 2：人体**剪影**(分割掩码)
选 **GREW**(野外大规模)权重 → 监控/跨域泛化最好(Rank-1≈87)。

⚠️ 本机纯 CPU：OpenGait 的 `BaseModel.__init__` 写死 CUDA+DDP，`np2var` 也强制 `.cuda()`，
无法整体直接用。故这里**只用它的 `SkeletonGaitPP` 网络类**在 CPU 建网 + 载权重，**复用它的
热图预处理**(`GenerateHeatmapTransform`)与标准剪影裁剪逻辑，**自己在 CPU 构造输入张量**喂
`forward`。效果与 GPU 完全一致(同权重同运算)，只是慢。上云改 `GAIT_DEVICE=cuda` 即可。

依赖：OpenGait 仓库 clone + 726MB GREW 权重(均在 git 仓库外，路径见 `GAIT_*` 配置)。
剪影分割用 ultralytics 的 `yolov8m-seg`，姿态用 `yolov8n-pose`(复用项目内权重)。
"""
from __future__ import annotations

import importlib
import sys
import threading

import numpy as np

from .config import settings

GAIT_DIM = None  # 实际维度在首次推理后确定（c*p）

_lock = threading.Lock()
_state: dict = {"ready": False, "model": None, "heat": None, "seg": None, "pose": None, "error": None}


# ---------------- 懒加载：建模型 + 热图变换 ----------------
def _ensure() -> bool:
    """线程安全地懒加载 SkeletonGait++(CPU) + OpenGait 热图变换。失败则记录 error 返回 False。"""
    if _state["ready"]:
        return True
    if _state["error"] is not None:
        return False
    with _lock:
        if _state["ready"]:
            return True
        if _state["error"] is not None:
            return False
        try:
            import torch
            import torch.nn as nn

            root = settings.gait_opengait_root
            og = root + "/opengait"
            for p in (root, og):
                if p not in sys.path:
                    sys.path.insert(0, p)

            # 模型文件名带 '++'，非法标识符 → 用 importlib 单独导入（避免触发 BigGait 等无关重依赖）
            mod = importlib.import_module("modeling.models.skeletongait++")
            SkeletonGaitPP = mod.SkeletonGaitPP

            # GREW SkeletonGait++ 的 model_cfg（取自 configs/skeletongait/skeletongait++_GREW.yaml）
            model_cfg = {
                "model": "SkeletonGaitPP",
                "Backbone": {"in_channels": 3, "blocks": [1, 4, 4, 1], "C": 2},
                "SeparateBNNecks": {"class_num": 20000},
            }
            # 绕过 BaseModel.__init__ 的 CUDA/DDP（CPU 上跑不起来、单序列推理也用不到），
            # 直接用它的 build_network 在 CPU 建网，再载权重。
            m = SkeletonGaitPP.__new__(SkeletonGaitPP)
            nn.Module.__init__(m)
            m.training = False
            m.build_network(model_cfg)
            m = m.to(settings.gait_device).eval()

            ckpt = torch.load(settings.gait_ckpt, map_location=settings.gait_device, weights_only=False)
            sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            missing, unexpected = m.load_state_dict(sd, strict=False)

            # 复用 OpenGait 的热图预处理（纯 numpy/CPU，可直接用）
            from datasets.pretreatment_heatmap import GenerateHeatmapTransform

            heat = GenerateHeatmapTransform(
                coco18tococo17_args={"transfer_to_coco17": False},
                padkeypoints_args={"pad_method": "knn", "use_conf": True},
                norm_args={"pose_format": "coco", "use_conf": True, "heatmap_image_height": 128},
                heatmap_generator_args={"sigma": 8.0, "use_score": True, "img_h": 128, "img_w": 128,
                                        "with_limb": None, "with_kp": None},
                align_args={"align": True, "final_img_size": 64, "offset": 0, "heatmap_image_size": 128},
            )

            _state["torch"] = torch
            _state["model"] = m
            _state["heat"] = heat
            _state["missing"] = len(missing)
            _state["unexpected"] = len(unexpected)
            _state["ready"] = True
            return True
        except Exception as exc:  # noqa: BLE001
            _state["error"] = f"{type(exc).__name__}: {exc}"
            return False


def available() -> bool:
    """步态分支是否可用（OpenGait + 权重就绪）。"""
    return _ensure()


def load_error() -> str | None:
    return _state.get("error")


# ---------------- 剪影标准化（复用 OpenGait imgs2pickle 的裁剪逻辑）----------------
def _cut_sil(mask: np.ndarray, img_size: int = 64) -> np.ndarray:
    """把一帧二值人体掩码裁成 gait 标准剪影 [64,64]：上下贴人、按身高缩放、按质心居中。"""
    import cv2

    img = mask
    if img.dtype != np.uint8:
        img = (img > 0).astype(np.uint8) * 255
    y_sum = img.sum(axis=1)
    nz = np.where(y_sum != 0)[0]
    if len(nz) == 0:
        return np.zeros((img_size, img_size), np.uint8)
    img = img[nz[0]: nz[-1] + 1, :]
    ratio = img.shape[1] / max(1, img.shape[0])
    img = cv2.resize(img, (max(1, int(img_size * ratio)), img_size), interpolation=cv2.INTER_CUBIC)
    total = float(img.sum())
    if total <= 0:
        return np.zeros((img_size, img_size), np.uint8)
    x_center = int(np.searchsorted(img.sum(axis=0).cumsum(), total / 2.0))
    half = img_size // 2
    left, right = x_center - half, x_center + half
    if left <= 0 or right >= img.shape[1]:
        pad = np.zeros((img.shape[0], half), img.dtype)
        img = np.concatenate([pad, img, pad], axis=1)
        left += half
        right += half
    out = img[:, left:right]
    if out.shape[1] != img_size:
        out = cv2.resize(out, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    return out.astype(np.uint8)


# ---------------- 核心：一条 track 的序列 → 步态向量 ----------------
def embed_track(pose_seq: list[np.ndarray], sil_seq: list[np.ndarray]) -> np.ndarray | None:
    """对一条 track 的(姿态序列 + 剪影序列)提步态向量（L2 归一化）。

    Args:
        pose_seq: 每帧 17×3 的 COCO17 关键点 [x, y, conf]（来自 YOLO-Pose），按时间序。
        sil_seq:  每帧人体二值掩码（HxW，0/255），与 pose_seq 一一对应。

    Returns:
        np.ndarray 步态向量（已 L2 归一化），或 None（不可用 / 帧太少 / 失败）。
    """
    global GAIT_DIM
    if not _ensure():
        return None
    n = min(len(pose_seq), len(sil_seq))
    if n < settings.gait_min_frames:
        return None
    try:
        torch = _state["torch"]
        pose = np.asarray(pose_seq[:n], dtype=np.float32)            # [T,17,3]
        heat = _state["heat"](pose)                                  # [T,2,64,64]
        sils = np.stack([_cut_sil(np.asarray(m)) for m in sil_seq[:n]])  # [T,64,64]
        cat = np.concatenate([np.asarray(heat, dtype=np.float32), sils[:, None].astype(np.float32)], axis=1)  # [T,3,64,64]
        cut = int(cat.shape[-1] // 64) * 10                          # 64→44
        if cut > 0:
            cat = cat[..., cut:-cut]
        cat = cat / 255.0                                            # 归一化（同 BaseSilCuttingTransform）
        ipt = torch.from_numpy(cat).unsqueeze(0).to(settings.gait_device).float()  # [1,T,3,64,44]
        seqL = torch.tensor([[n]], device=settings.gait_device)
        labs = torch.zeros(1, dtype=torch.long, device=settings.gait_device)
        with torch.no_grad():
            retval = _state["model"]((([ipt]), labs, None, None, seqL))
        emb = retval["inference_feat"]["embeddings"]                 # [1, c, p]
        vec = emb.reshape(-1).detach().cpu().numpy().astype(np.float32)
        nrm = float(np.linalg.norm(vec))
        vec = vec / nrm if nrm > 0 else vec
        GAIT_DIM = int(vec.shape[0])
        return vec
    except Exception as exc:  # noqa: BLE001
        _state["embed_error"] = f"{type(exc).__name__}: {exc}"
        return None


# ---------------- 逐帧提取：姿态 + 剪影（供集成层按 track 累积）----------------
def _seg_model():
    if _state["seg"] is None:
        from ultralytics import YOLO

        _state["seg"] = YOLO(settings.gait_seg_model)
    return _state["seg"]


def _pose_model():
    if _state["pose"] is None:
        from ultralytics import YOLO

        _state["pose"] = YOLO(settings.pose_model)
    return _state["pose"]


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def extract_persons(frame_bgr: np.ndarray) -> list[dict]:
    """对一帧跑 YOLO-Pose + YOLO-seg，返回每个人的 {box, kpts(17,3), mask(HxW 0/255)}。

    供集成层(event_analysis_pipeline)按 track 的 box 关联、逐帧累积成 pose_seq / sil_seq。
    """
    pose_res = _pose_model()(frame_bgr, verbose=False, conf=settings.pose_conf)[0]
    seg_res = _seg_model()(frame_bgr, verbose=False)[0]
    h, w = frame_bgr.shape[:2]

    # seg：取 person 类(0)的掩码 + 框
    seg_items = []
    if seg_res.masks is not None:
        for box, m, cls in zip(seg_res.boxes.xyxy.cpu().numpy(),
                               seg_res.masks.data.cpu().numpy(),
                               seg_res.boxes.cls.cpu().numpy()):
            if int(cls) != 0:
                continue
            mask = (m > 0.5).astype(np.uint8) * 255
            if mask.shape != (h, w):
                import cv2

                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            seg_items.append((box.tolist(), mask))

    out = []
    if pose_res.keypoints is None:
        return out
    kpts_all = pose_res.keypoints.data.cpu().numpy()  # [P,17,3]
    boxes_all = pose_res.boxes.xyxy.cpu().numpy() if pose_res.boxes is not None else []
    for i, kpts in enumerate(kpts_all):
        box = boxes_all[i].tolist() if i < len(boxes_all) else [0, 0, w, h]
        # 关联最佳 seg 掩码（box IoU）
        mask = None
        best = 0.0
        for sbox, smask in seg_items:
            iou = _iou(box, sbox)
            if iou > best:
                best, mask = iou, smask
        if mask is None:
            mask = np.zeros((h, w), np.uint8)
        out.append({"box": box, "kpts": kpts.astype(np.float32), "mask": mask})
    return out


__all__ = ["available", "load_error", "embed_track", "extract_persons", "GAIT_DIM"]
