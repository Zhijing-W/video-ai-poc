from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from ...config import settings

_lock = threading.Lock()
_sr_state: dict = {"ready": False, "model": None, "error": None}


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

__all__ = ["enhance", "superres_error"]
