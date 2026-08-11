from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from ._common import (
    image_from_tensor,
    require_image_descriptor,
    resolve_weights,
    rgb_tensor,
    select_device,
)

OFFICIAL_WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.1/RealESRGAN_x2plus.pth"
)


@dataclass(frozen=True)
class LoadedRealESRGAN:
    descriptor: Any
    torch: Any
    device: Any


def preprocess(torch, image: Image.Image):
    """Convert PIL RGB deterministically to a float NCHW tensor in [0,1]."""
    return rgb_tensor(torch, image)


def _load(settings) -> LoadedRealESRGAN:
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

    weights = resolve_weights(
        settings.face_realesrgan_x2plus_weights,
        default_url=OFFICIAL_WEIGHTS_URL,
        cache_namespace="realesrgan-x2plus-v0.2.1",
    )
    descriptor = require_image_descriptor(
        ModelLoader().load_from_file(weights),
        ImageModelDescriptor,
        "ESRGAN",
    )
    if int(getattr(descriptor, "scale", 0)) != 2:
        raise ValueError(
            f"RealESRGAN_x2plus weights must have scale 2, got {descriptor.scale!r}"
        )
    device = select_device(torch, settings.face_device)
    descriptor = descriptor.eval().to(device)
    return LoadedRealESRGAN(descriptor, torch, device)


def _enhance(
    loaded: LoadedRealESRGAN,
    image: Image.Image,
    aligned: bool,
) -> Image.Image:
    del aligned
    tensor = preprocess(loaded.torch, image).to(loaded.device)
    with loaded.torch.inference_mode():
        output = loaded.descriptor(tensor)
    return image_from_tensor(output, value_range=(0.0, 1.0))


def register(register_backend, settings) -> None:
    """Register raw RealESRGAN_x2plus inference; no GFPGAN path is enabled."""
    register_backend(
        "realesrgan_x2plus",
        lambda: _load(settings),
        _enhance,
        replace=True,
    )


__all__ = [
    "LoadedRealESRGAN",
    "OFFICIAL_WEIGHTS_URL",
    "preprocess",
    "register",
]
