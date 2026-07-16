"""CR-FIQA人脸识别可用性评分包装器。

官方源码与权重作为外部模型资产放在models/CR-FIQA，不复制到产品源码。
CR-FIQA官方代码使用CC BY-NC 4.0许可，商业使用前必须完成许可确认。
"""
from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import numpy as np

from ....config import settings


_lock = threading.Lock()
_state: dict = {
    "backend": None,
    "model": None,
    "torch": None,
    "device": None,
    "error": None,
}


def _resolve_device(torch, configured: str) -> str:
    device = (configured or "auto").strip().lower()
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"FACE_FIQA_DEVICE={configured}，但当前PyTorch不可用CUDA")
    return device


def _load_iresnet_module(root: Path):
    source = root / "backbones" / "iresnet.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"CR-FIQA官方源码不存在：{source}。"
            "请将https://github.com/fdbtrs/CR-FIQA放到FACE_FIQA_ROOT。"
        )
    spec = importlib.util.spec_from_file_location("_cr_fiqa_iresnet", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载CR-FIQA网络定义：{source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_model():
    backend = (settings.face_fiqa_backend or "off").strip().lower()
    if backend in {"off", "none", ""}:
        return None
    if backend != "cr_fiqa":
        raise ValueError(f"未知FACE_FIQA_BACKEND：{backend}")

    if _state["model"] is not None:
        return _state["model"]
    if _state["error"] is not None:
        raise RuntimeError(f"CR-FIQA此前加载失败：{_state['error']}")

    with _lock:
        if _state["model"] is not None:
            return _state["model"]
        if _state["error"] is not None:
            raise RuntimeError(f"CR-FIQA此前加载失败：{_state['error']}")
        try:
            import torch

            root = Path(settings.face_fiqa_root).expanduser()
            weights = Path(settings.face_fiqa_weights).expanduser()
            if not weights.is_file():
                raise FileNotFoundError(
                    f"CR-FIQA权重不存在：{weights}。"
                    "请下载官方CR-FIQA(S)的32572backbone.pth。"
                )

            module = _load_iresnet_module(root)
            builder = getattr(module, settings.face_fiqa_arch, None)
            if builder is None:
                raise ValueError(
                    f"CR-FIQA网络不存在：{settings.face_fiqa_arch}，"
                    "可选iresnet50或iresnet100"
                )
            model = builder(num_features=512, qs=1, use_se=False)
            state_dict = torch.load(weights, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            device = _resolve_device(torch, settings.face_fiqa_device)
            model = model.to(device)

            _state.update(
                {
                    "backend": backend,
                    "model": model,
                    "torch": torch,
                    "device": device,
                    "error": None,
                }
            )
            return model
        except Exception as exc:
            _state["error"] = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"CR-FIQA加载失败：{_state['error']}") from exc


def score(aligned_bgr: np.ndarray | None) -> float | None:
    """对112×112对齐BGR人脸输出CR-FIQA原始质量分数。"""
    if (settings.face_fiqa_backend or "off").strip().lower() in {"off", "none", ""}:
        return None
    if aligned_bgr is None:
        raise ValueError("CR-FIQA需要SCRFD五点关键点对齐后的人脸")

    model = _ensure_model()
    torch = _state["torch"]
    try:
        import cv2

        bgr = np.asarray(aligned_bgr)
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"CR-FIQA输入必须是H×W×3，收到{bgr.shape}")
        if bgr.shape[:2] != (112, 112):
            bgr = cv2.resize(bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
        rgb = bgr[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)[None]).float()
        tensor = tensor.to(_state["device"])
        tensor = (tensor / 255.0 - 0.5) / 0.5
        with torch.no_grad():
            _, quality = model(tensor)
        value = float(quality.reshape(-1)[0].detach().cpu().item())
        if not np.isfinite(value):
            raise RuntimeError(f"CR-FIQA输出非有限数：{value}")
        return value
    except Exception as exc:
        raise RuntimeError(f"CR-FIQA推理失败：{type(exc).__name__}: {exc}") from exc


def active_backend() -> str:
    return (settings.face_fiqa_backend or "off").strip().lower()


def load_error() -> str | None:
    return _state.get("error")


def reset_for_tests() -> None:
    """清理进程内模型状态，仅供测试使用。"""
    with _lock:
        _state.update(
            {
                "backend": None,
                "model": None,
                "torch": None,
                "device": None,
                "error": None,
            }
        )


__all__ = ["active_backend", "load_error", "score"]
