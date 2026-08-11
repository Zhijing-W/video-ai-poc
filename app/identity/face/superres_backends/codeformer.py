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

# CodeFormer source and official weights use S-Lab License 1.0, which permits
# non-commercial use. Commercial/production users must obtain appropriate rights.
OFFICIAL_WEIGHTS_URL = (
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
)


@dataclass(frozen=True)
class LoadedCodeFormer:
    descriptor: Any
    torch: Any
    device: Any
    settings: Any


def validate_fidelity(value: Any) -> float:
    try:
        fidelity = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("CodeFormer fidelity must be a number in [0, 1]") from exc
    if not 0.0 <= fidelity <= 1.0:
        raise ValueError(
            f"CodeFormer fidelity must be in [0, 1], got {fidelity!r}"
        )
    return fidelity


def preprocess(torch, image: Image.Image):
    """Resize RGB to 512x512 bilinearly and normalize [0,1] to [-1,1]."""
    return rgb_tensor(torch, image, size=(512, 512)).mul(2.0).sub(1.0)


def _load(settings) -> LoadedCodeFormer:
    import torch
    import spandrel_extra_arches
    from spandrel import ImageModelDescriptor, ModelLoader

    validate_fidelity(settings.face_codeformer_fidelity)
    weights = resolve_weights(
        settings.face_codeformer_weights,
        default_url=OFFICIAL_WEIGHTS_URL,
        cache_namespace="codeformer-v0.1.0",
    )
    spandrel_extra_arches.install(ignore_duplicates=True)
    descriptor = require_image_descriptor(
        ModelLoader().load_from_file(weights),
        ImageModelDescriptor,
        "CodeFormer",
    )
    device = select_device(torch, settings.face_device)
    descriptor = descriptor.eval().to(device)
    return LoadedCodeFormer(descriptor, torch, device, settings)


def _enhance(
    loaded: LoadedCodeFormer,
    image: Image.Image,
    aligned: bool,
) -> Image.Image:
    if not aligned:
        raise ValueError("CodeFormer requires an already aligned face crop")
    fidelity = validate_fidelity(loaded.settings.face_codeformer_fidelity)
    tensor = preprocess(loaded.torch, image).to(loaded.device)
    with loaded.torch.inference_mode():
        output = loaded.descriptor.model(
            tensor,
            weight=fidelity,
            adain=True,
        )[0]
    return image_from_tensor(output, value_range=(-1.0, 1.0))


def register(register_backend, settings) -> None:
    """Register lazy aligned-face CodeFormer inference without detection/alignment."""
    register_backend(
        "codeformer",
        lambda: _load(settings),
        _enhance,
        replace=True,
    )


__all__ = [
    "LoadedCodeFormer",
    "OFFICIAL_WEIGHTS_URL",
    "preprocess",
    "register",
    "validate_fidelity",
]
