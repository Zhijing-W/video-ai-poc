from __future__ import annotations

import importlib
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

_IMPORT_LOCK = threading.RLock()


def require_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")
    return path


def require_directory(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label}不存在：{path}")
    return path


def select_device(torch, configured: str | None, *, require_cuda: bool = False):
    name = str(configured or "auto").strip().lower()
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"未知REID_DEVICE：{configured!r}")
    has_cuda = bool(torch.cuda.is_available())
    if require_cuda and not has_cuda:
        raise RuntimeError("该ReID后端的官方实现需要CUDA，但当前PyTorch不可用CUDA")
    if name == "cuda" and not has_cuda:
        raise RuntimeError("REID_DEVICE=cuda，但当前PyTorch不可用CUDA")
    if require_cuda and name == "cpu":
        raise RuntimeError("该ReID后端的官方实现不支持REID_DEVICE=cpu")
    return torch.device("cuda:0" if has_cuda and name != "cpu" else "cpu")


def image_tensor(
    torch,
    image: Image.Image,
    *,
    size: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
):
    resized = image.convert("RGB").resize(size, Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def first_tensor(value: Any):
    if hasattr(value, "image_embeds"):
        return value.image_embeds
    if hasattr(value, "pooler_output"):
        return value.pooler_output
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state[:, 0]
    if hasattr(value, "detach"):
        return value
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"模型没有返回可识别的张量：{type(value).__name__}")


def state_dict_from_checkpoint(checkpoint: Any) -> dict[str, Any]:
    value = checkpoint
    for key in ("state_dict", "model", "module"):
        if isinstance(value, dict) and key in value and isinstance(value[key], dict):
            value = value[key]
    if not isinstance(value, dict):
        raise TypeError("checkpoint不包含state_dict")
    return {
        (key[7:] if key.startswith("module.") else key): tensor
        for key, tensor in value.items()
    }


@contextmanager
def isolated_source_imports(
    source_root: Path,
    top_level_packages: tuple[str, ...],
) -> Iterator[None]:
    """Temporarily isolate upstream projects that reuse names like config/model."""
    with _IMPORT_LOCK:
        original_path = list(sys.path)
        saved = {
            name: module
            for name, module in list(sys.modules.items())
            if any(
                name == package or name.startswith(f"{package}.")
                for package in top_level_packages
            )
        }
        for name in saved:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(source_root))
        importlib.invalidate_caches()
        try:
            yield
        finally:
            for name in list(sys.modules):
                if any(
                    name == package or name.startswith(f"{package}.")
                    for package in top_level_packages
                ):
                    sys.modules.pop(name, None)
            sys.modules.update(saved)
            sys.path[:] = original_path
            importlib.invalidate_caches()
