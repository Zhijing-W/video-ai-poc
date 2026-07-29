from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from ._common import first_tensor, require_directory, select_device


@dataclass(frozen=True)
class LoadedSiglip2:
    model: Any
    processor: Any
    torch: Any
    device: Any
    dim: int


def load(settings) -> LoadedSiglip2:
    import torch
    from transformers import AutoModel, AutoProcessor

    model_path = require_directory(settings.reid_siglip2_model, "SigLIP2模型目录")
    device = select_device(torch, settings.reid_device)
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
    ).eval().to(device)
    loaded = LoadedSiglip2(model, processor, torch, device, dim=0)
    probe = embed(loaded, Image.new("RGB", (224, 224)))
    return LoadedSiglip2(model, processor, torch, device, dim=int(probe.size))


def embed(loaded: LoadedSiglip2, image: Image.Image):
    inputs = loaded.processor(
        images=image.convert("RGB"),
        return_tensors="pt",
    )
    inputs = {
        name: value.to(loaded.device) if hasattr(value, "to") else value
        for name, value in inputs.items()
    }
    with loaded.torch.inference_mode():
        output = loaded.model.get_image_features(**inputs)
    return first_tensor(output).squeeze(0).detach().float().cpu().numpy()


def register(register_backend, settings) -> None:
    register_backend(
        "siglip2",
        lambda: load(settings),
        embed,
        dim=None,
        label="SigLIP2 Person ReID",
        description="图文对齐的人体外观检索模型，支持图像ReID与后续文字找人。",
        experimental=True,
        replace=True,
    )
