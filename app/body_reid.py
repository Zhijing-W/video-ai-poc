"""Pluggable body ReID feature extraction."""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import settings

BackendLoader = Callable[[], Any]
BackendEmbedder = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class ReIdBackend:
    name: str
    loader: BackendLoader | None
    embedder: BackendEmbedder
    dim: int | None
    label: str
    description: str
    requires_cuda: bool = False
    experimental: bool = False


@dataclass
class _BackendState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    ready: bool = False
    model: Any = None
    dim: int | None = None
    load_error: str | None = None


_registry_lock = threading.RLock()
_backends: dict[str, ReIdBackend] = {}
_states: dict[str, _BackendState] = {}
_active_lock = threading.RLock()
_active_backend: str | None = None
_ALIASES = {
    "clip_reid": "clipreid",
    "siglip_2": "siglip2",
    "siglip2_person_reid": "siglip2",
    "color": "coarse",
}
_AUTO_ORDER = ("osnet", "resnet50", "coarse")


def _reid_cuda() -> bool:
    """按 settings.reid_device (auto/cuda/cpu) 判断 ReID 是否用 CUDA。"""
    d = (getattr(settings, "reid_device", "auto") or "auto").strip().lower()
    if d == "cpu":
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


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
    path = Path(settings.reid_resnet50_weights).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"ResNet50权重不存在：{path}；请先运行 "
            "python scripts\\download_models.py --reid"
        )

    import torch
    from torchvision.models import resnet50

    net = resnet50(weights=None)
    net.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True),
        strict=True,
    )
    net.fc = torch.nn.Identity()  # 去掉分类头，留 2048 维 avgpool 特征
    net.eval()
    dev = "cuda" if _reid_cuda() else "cpu"
    net = net.to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
    return {"torch": torch, "net": net, "mean": mean, "std": std, "device": dev}


def _embed_resnet50(m, crop) -> np.ndarray:
    torch = m["torch"]
    img = crop.convert("RGB").resize((128, 256))  # (W,H) ReID 习惯 256×128
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(m["device"])  # (1,3,256,128)
    t = (t - m["mean"]) / m["std"]
    with torch.no_grad():
        feat = m["net"](t).squeeze(0).cpu().numpy().astype(np.float32)
    n = float(np.linalg.norm(feat))
    return feat / n if n > 0 else feat


# ---------------- 后端：OSNet（经 boxmot，预训练域泛化 ReID）----------------
def _load_osnet():
    """用 boxmot 的 ReID 运行时加载已准备的 OSNet 预训练权重。

    选型 `osnet_ain_x1_0_msmt17`：OSNet 域泛化版（AIN 自适应实例归一化）+ MSMT17（最难最大
    ReID 训练集）→ 对"跨摄像头 / 没见过的新场景"鲁棒，契合"自找数据泛化到客户现场"。
    （早期用 torchreid 直装在 Py3.12/numpy2 上不可行，改走 boxmot——pip 干净、权重自动管理。）
    """
    weights = Path(settings.reid_osnet_weights).expanduser()
    if not weights.is_file():
        raise FileNotFoundError(
            f"OSNet权重不存在：{weights}；请先运行 "
            "python scripts\\download_models.py --reid"
        )

    from boxmot.reid.core.auto_backend import ReidAutoBackend
    import torch

    # 传 torch.device 对象：boxmot 会跳过它的 select_device()，避免其把
    # CUDA_VISIBLE_DEVICES 设成非法字符串 'cuda' 而污染整个进程的 CUDA。
    dev = torch.device("cuda:0") if _reid_cuda() else torch.device("cpu")
    backend = ReidAutoBackend(weights=weights, device=dev, half=False)
    return {"backend": backend.model}


def _embed_osnet(model, crop) -> np.ndarray:
    """对一张人像 crop 提 OSNet 512 维 ReID 指纹（boxmot 已做 L2 归一化）。"""
    backend = model["backend"]
    bgr = np.asarray(crop.convert("RGB"))[:, :, ::-1]  # PIL RGB → BGR（boxmot/cv2 约定）
    height, width = bgr.shape[:2]
    box = np.asarray([[0, 0, width, height]], dtype=np.float32)
    feats = backend.get_features(box, bgr)
    feat = np.asarray(feats[0], dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(feat))
    return feat / n if n > 0 else feat


def _normalize_backend_name(value: str | None = None) -> str:
    raw = settings.reid_backend if value is None else value
    name = str(raw or "auto").strip().lower().replace("-", "_")
    return _ALIASES.get(name, name)


def register_backend(
    name: str,
    loader: BackendLoader | None,
    embedder: BackendEmbedder,
    *,
    dim: int | None,
    label: str | None = None,
    description: str = "",
    requires_cuda: bool = False,
    experimental: bool = False,
    replace: bool = False,
) -> None:
    normalized = _normalize_backend_name(name)
    if normalized == "auto":
        raise ValueError("'auto'是保留名称，不能注册")
    if loader is not None and not callable(loader):
        raise TypeError("loader必须可调用或为None")
    if not callable(embedder):
        raise TypeError("embedder必须可调用")
    if dim is not None and dim < 1:
        raise ValueError("embedding维度必须为正整数")
    with _registry_lock:
        if normalized in _backends and not replace:
            raise ValueError(f"ReID后端已注册：{normalized}")
        _backends[normalized] = ReIdBackend(
            name=normalized,
            loader=loader,
            embedder=embedder,
            dim=dim,
            label=label or normalized,
            description=description,
            requires_cuda=requires_cuda,
            experimental=experimental,
        )
        _states[normalized] = _BackendState(dim=dim)


def available_backends() -> tuple[str, ...]:
    with _registry_lock:
        return tuple(sorted(_backends))


def backend_metadata() -> dict[str, dict[str, Any]]:
    with _registry_lock:
        return {
            name: {
                "label": backend.label,
                "description": backend.description,
                "dim": backend.dim,
                "requires_cuda": backend.requires_cuda,
                "experimental": backend.experimental,
            }
            for name, backend in sorted(_backends.items())
        }


def validate_backend(value: str | None = None) -> str:
    normalized = _normalize_backend_name(value)
    if normalized == "auto":
        return normalized
    with _registry_lock:
        if normalized not in _backends:
            choices = ", ".join(("auto", *sorted(_backends)))
            raise ValueError(f"未知人形ReID后端：{value!r}；可选 {choices}")
    return normalized


def _load_backend(name: str) -> _BackendState:
    with _registry_lock:
        backend = _backends[name]
        state = _states[name]
    if state.ready:
        return state
    if state.load_error is not None:
        raise RuntimeError(state.load_error)
    with state.lock:
        if state.ready:
            return state
        if state.load_error is not None:
            raise RuntimeError(state.load_error)
        try:
            model = backend.loader() if backend.loader is not None else None
            loaded_dim = getattr(model, "dim", None)
            state.model = model
            state.dim = int(loaded_dim or backend.dim or 0)
            if state.dim < 1:
                raise RuntimeError("后端未声明embedding维度")
            state.ready = True
        except Exception as exc:
            state.model = None
            state.load_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(state.load_error) from exc
    return state


def _ensure_backend() -> str:
    global _active_backend
    if _active_backend is not None:
        return _active_backend
    with _active_lock:
        if _active_backend is not None:
            return _active_backend
        requested = validate_backend()
        if requested != "auto":
            try:
                _load_backend(requested)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"人形ReID后端 {requested} 加载失败：{exc}"
                ) from exc
            _active_backend = requested
            return requested

        failures = []
        for name in _AUTO_ORDER:
            try:
                _load_backend(name)
                _active_backend = name
                return name
            except RuntimeError as exc:
                failures.append(f"{name}: {exc}")
        raise RuntimeError("auto无法加载任何人形ReID后端：" + "; ".join(failures))


def active_backend() -> str:
    return _ensure_backend()


def reset_backend(value: str | None = None) -> None:
    """Release loaded model state so the next call uses current settings."""
    global _active_backend
    with _active_lock, _registry_lock:
        if value is None:
            names = tuple(_states)
            _active_backend = None
        else:
            normalized = validate_backend(value)
            names = tuple(_states) if normalized == "auto" else (normalized,)
            if _active_backend in names:
                _active_backend = None
        for name in names:
            _states[name] = _BackendState(dim=_backends[name].dim)


def embed_dim() -> int:
    name = _ensure_backend()
    return int(_load_backend(name).dim)


def embed(crop) -> np.ndarray:
    """对一张 PIL 裁图提归一化外观指纹向量（维度由当前 backend 决定）。"""
    name = _ensure_backend()
    with _registry_lock:
        backend = _backends[name]
    state = _load_backend(name)
    value = np.asarray(
        backend.embedder(state.model, crop),
        dtype=np.float32,
    ).reshape(-1)
    if value.size != state.dim:
        raise ValueError(
            f"{name}返回{value.size}维向量，注册维度为{state.dim}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name}返回非有限embedding")
    norm = float(np.linalg.norm(value))
    if norm <= 0:
        raise ValueError(f"{name}返回零向量")
    return value / norm


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


register_backend(
    "coarse",
    None,
    lambda _model, crop: _embed_coarse(crop),
    dim=_COARSE_DIM,
    label="颜色直方图",
    description="零依赖的颜色外观兜底，不是真正的身份ReID模型。",
)
register_backend(
    "resnet50",
    _load_resnet50,
    _embed_resnet50,
    dim=2048,
    label="ResNet50 ImageNet",
    description="通用视觉特征兜底，不是专用ReID模型。",
)
register_backend(
    "osnet",
    _load_osnet,
    _embed_osnet,
    dim=512,
    label="OSNet-AIN MSMT17",
    description="默认行人ReID后端，面向跨摄像头外观匹配。",
)

from .identity.body_reid_backends import clipreid, differ, siglip2

clipreid.register(register_backend, settings)
siglip2.register(register_backend, settings)
differ.register(register_backend, settings)


__all__ = [
    "active_backend",
    "assess_quality",
    "available_backends",
    "backend_metadata",
    "embed",
    "embed_dim",
    "register_backend",
    "reset_backend",
    "validate_backend",
]
