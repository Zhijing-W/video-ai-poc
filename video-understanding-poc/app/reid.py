"""ReID 外观指纹提取（Phase 3 · Step 14）。

定位：给一个人/物的裁图（crop）提一个**归一化外观特征向量**（"指纹"），供 `gallery.py`
查库认人。**与向量库解耦**——本模块只回答"这张 crop 的指纹是什么"，不关心库怎么存。

可插拔后端（`REID_BACKEND`，默认 auto，按可用性自动择优 osnet → resnet50 → coarse）：
  - **osnet**   ：经 **boxmot** 加载的 OSNet 预训练（默认 `osnet_ain_x1_0_msmt17`，域泛化版 + 最难
                  训练集，512 维）。真·行人重识别、对跨摄像头/新场景鲁棒，**本机默认主力**。
                  （早期用 torchreid 直装在 Py3.12/numpy2 不可行，已改走 boxmot：pip 干净、权重自动管理。）
  - **resnet50**：torchvision ResNet50（ImageNet 预训练，取 avgpool 2048 维通用外观特征）。
                  通用骨干、非专用 ReID，仅作 osnet 不可用时的兜底。
  - **coarse**  ：零新依赖（PIL+numpy）的 HSV 颜色直方图（72 维），离线最末兜底。

设计文档对应：3.4「三档指纹（由粗到细）」。这里把"档位"做成可切换 backend，便于以后
按"先 coarse 粗筛、灰区再上 osnet"的成本梯度组合（留给 Step 15/三时钟编排去调度）。
"""
from __future__ import annotations

import threading

import numpy as np

from .config import settings

_lock = threading.Lock()
_state: dict = {"backend": None, "model": None, "dim": None}


# ---------------- 后端：coarse（零依赖） ----------------
_COARSE_DIM = 72  # 18 hue × 4 saturation 的联合直方图


def _embed_coarse(crop) -> np.ndarray:
    """色相×饱和度 (18×4=72) 联合直方图，按饱和度加权，L2 归一化。

    用"L1 颜色直方图档"的经典做法：**联合 H-S 直方图**而非分别的 H/S/V 边缘直方图——
    后者会让"红、蓝但同样亮/同样饱和"的两人因 S/V 分布相同而被判成一人。联合直方图里
    "红且高饱和"与"蓝且高饱和"落在不同 bin，颜色身份才真正可分；按 s 加权让灰背景近乎不计。
    """
    img = crop.convert("RGB").resize((64, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx, mn = arr.max(-1), arr.min(-1)
    diff = mx - mn
    s = np.where(mx > 0, diff / (mx + 1e-6), 0.0)
    mask = diff > 1e-6
    rc = np.where(mask, (mx - r) / (diff + 1e-6), 0.0)
    gc = np.where(mask, (mx - g) / (diff + 1e-6), 0.0)
    bc = np.where(mask, (mx - b) / (diff + 1e-6), 0.0)
    h = np.where(mx == r, bc - gc, np.where(mx == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    hist, _, _ = np.histogram2d(
        h.ravel(), s.ravel(), bins=[18, 4], range=[[0, 1], [0, 1]], weights=s.ravel()
    )
    vec = hist.astype(np.float32).ravel()
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


# ---------------- 后端：torchvision resnet50 ----------------
def _load_resnet50():
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2
    net = resnet50(weights=weights)
    net.fc = torch.nn.Identity()  # 去掉分类头，留 2048 维 avgpool 特征
    net.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return {"torch": torch, "net": net, "mean": mean, "std": std}


def _embed_resnet50(crop) -> np.ndarray:
    m = _state["model"]
    torch = m["torch"]
    img = crop.convert("RGB").resize((128, 256))  # (W,H) ReID 习惯 256×128
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,256,128)
    t = (t - m["mean"]) / m["std"]
    with torch.no_grad():
        feat = m["net"](t).squeeze(0).numpy().astype(np.float32)
    n = float(np.linalg.norm(feat))
    return feat / n if n > 0 else feat


# ---------------- 后端：OSNet（经 boxmot，预训练域泛化 ReID）----------------
def _load_osnet():
    """用 boxmot 的 ReID 运行时加载 OSNet 预训练权重（首次自动下载到 boxmot WEIGHTS 缓存）。

    选型 `osnet_ain_x1_0_msmt17`：OSNet 域泛化版（AIN 自适应实例归一化）+ MSMT17（最难最大
    ReID 训练集）→ 对"跨摄像头 / 没见过的新场景"鲁棒，契合"自找数据泛化到客户现场"。
    （早期用 torchreid 直装在 Py3.12/numpy2 上不可行，改走 boxmot——pip 干净、权重自动管理。）
    """
    from boxmot.reid import ReID
    from boxmot.utils import WEIGHTS

    reid = ReID(weights=WEIGHTS / settings.reid_osnet_weights, device="cpu", half=False)
    return {"reid": reid}


def _embed_osnet(crop) -> np.ndarray:
    """对一张人像 crop 提 OSNet 512 维 ReID 指纹（boxmot 已做 L2 归一化）。"""
    reid = _state["model"]["reid"]
    bgr = np.asarray(crop.convert("RGB"))[:, :, ::-1]  # PIL RGB → BGR（boxmot/cv2 约定）
    feats = reid([bgr])  # 返回 (N, dim) 已归一化
    feat = np.asarray(feats[0], dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(feat))
    return feat / n if n > 0 else feat


_BACKENDS = {
    "osnet": (_load_osnet, _embed_osnet, 512),
    "resnet50": (_load_resnet50, _embed_resnet50, 2048),
    "coarse": (None, _embed_coarse, _COARSE_DIM),
}
_AUTO_ORDER = ["osnet", "resnet50", "coarse"]


def _ensure_backend() -> None:
    """懒加载并按配置/可用性选定 backend（线程安全，仅初始化一次）。"""
    if _state["backend"] is not None:
        return
    with _lock:
        if _state["backend"] is not None:
            return
        want = (settings.reid_backend or "auto").strip().lower()
        order = _AUTO_ORDER if want == "auto" else [want]
        last_err = None
        for name in order:
            loader, _, dim = _BACKENDS[name]
            try:
                _state["model"] = loader() if loader else None
                _state["backend"] = name
                _state["dim"] = dim
                return
            except Exception as exc:  # 该后端不可用（缺依赖/下载失败）→ 尝试下一个
                last_err = exc
                continue
        # 理论上 coarse 永远可用；兜底再保险一次
        _state["backend"] = "coarse"
        _state["dim"] = _COARSE_DIM
        _state["model"] = None
        if last_err is not None:
            _state["load_warning"] = str(last_err)


def active_backend() -> str:
    _ensure_backend()
    return _state["backend"]


def reset_backend() -> None:
    """清空已选 backend，下次调用按当前 settings.reid_backend 重新择优加载。

    供"本次请求覆盖"的设置面板切 ReID 后端用（backend 是首次加载后缓存死的，切了需重置）。
    """
    with _lock:
        _state["backend"] = None
        _state["model"] = None
        _state["dim"] = None


def embed_dim() -> int:
    _ensure_backend()
    return int(_state["dim"])


def embed(crop) -> np.ndarray:
    """对一张 PIL 裁图提归一化外观指纹向量（维度由当前 backend 决定）。"""
    _ensure_backend()
    _, fn, _ = _BACKENDS[_state["backend"]]
    return fn(crop)


def assess_quality(crop) -> dict:
    """评估 crop 质量（供 gallery 质量门控）：面积 / 清晰度 / 长宽比。

    - area      : 像素面积，太小→远景小目标特征不可靠。
    - blur_var  : 拉普拉斯方差，越小越糊（运动模糊/失焦）。
    - aspect_ratio: 高/宽，人形通常 >1；异常多半是半个框或严重遮挡。
    """
    w, h = crop.size
    gray = np.asarray(crop.convert("L"), dtype=np.float32)
    # 拉普拉斯（4 邻域）方差，作为清晰度代理（无需 cv2）
    lap = (
        -4 * gray
        + np.roll(gray, 1, 0)
        + np.roll(gray, -1, 0)
        + np.roll(gray, 1, 1)
        + np.roll(gray, -1, 1)
    )
    blur_var = float(lap[1:-1, 1:-1].var()) if gray.size > 9 else 0.0
    return {
        "area": int(w * h),
        "width": int(w),
        "height": int(h),
        "blur_var": round(blur_var, 2),
        "aspect_ratio": round(h / w, 3) if w > 0 else None,
    }
