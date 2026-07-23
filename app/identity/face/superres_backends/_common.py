from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

_URL_PREFIXES = ("http://", "https://")
DOWNLOAD_TIMEOUT_SECONDS = 60


def select_device(torch, configured: str | None):
    name = str(configured or "auto").strip().lower()
    if name not in {"auto", "cuda", "cpu"}:
        raise ValueError(
            f"unsupported face device {configured!r}; expected auto, cuda, or cpu"
        )
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("face_device='cuda' requested but CUDA is unavailable")
        return torch.device("cuda")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def resolve_weights(
    configured: str | None,
    *,
    default_url: str,
    cache_namespace: str,
) -> Path:
    value = str(configured or "").strip()
    if value and not value.startswith(_URL_PREFIXES):
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"model weights not found: {path}")
        return path

    url = value or default_url
    parsed = urlparse(url)
    basename = Path(unquote(parsed.path)).name or "model.pth"
    if value:
        basename = f"{hashlib.sha256(url.encode()).hexdigest()[:12]}-{basename}"
    cache_dir = Path.home() / ".cache" / "event-monitor" / "superres" / cache_namespace
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / basename
    if destination.is_file():
        return destination

    partial = destination.with_name(f"{destination.name}.part-{os.getpid()}")
    try:
        with urllib.request.urlopen(
            url,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def rgb_tensor(torch, image: Image.Image, *, size: tuple[int, int] | None = None):
    rgb = image.convert("RGB")
    if size is not None and rgb.size != size:
        rgb = rgb.resize(size, resample=Image.Resampling.BILINEAR)
    array = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
    chw = np.ascontiguousarray(array.transpose(2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def image_from_tensor(tensor, *, value_range: tuple[float, float]) -> Image.Image:
    output = tensor.detach().cpu()
    if output.ndim == 4:
        if output.shape[0] != 1:
            raise ValueError(f"expected one output image, got batch size {output.shape[0]}")
        output = output[0]
    if output.ndim != 3 or output.shape[0] != 3:
        raise ValueError(f"expected RGB CHW output, got shape {tuple(output.shape)}")

    low, high = value_range
    output = output.float().clamp(low, high)
    output = (output - low) / (high - low)
    array = (
        output.mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    return Image.fromarray(array)


def require_image_descriptor(descriptor, image_descriptor_type, architecture: str):
    if not isinstance(descriptor, image_descriptor_type):
        raise TypeError(f"{architecture} weights did not load as an image model")
    actual = str(getattr(getattr(descriptor, "architecture", None), "id", ""))
    if actual.lower() != architecture.lower():
        raise ValueError(
            f"expected {architecture} weights, but Spandrel detected {actual or 'unknown'}"
        )
    return descriptor
