from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from PIL import Image

from ._common import (
    first_tensor,
    image_tensor,
    isolated_source_imports,
    require_directory,
    require_file,
    select_device,
    state_dict_from_checkpoint,
)


@dataclass(frozen=True)
class LoadedClipReid:
    model: Any
    torch: Any
    device: Any
    dim: int = 1280


def load(settings) -> LoadedClipReid:
    import torch

    source_root = require_directory(
        settings.reid_clipreid_root,
        "CLIP-ReID官方源码目录",
    )
    config_path = require_file(
        settings.reid_clipreid_config,
        "CLIP-ReID配置",
    )
    checkpoint_path = require_file(
        settings.reid_clipreid_weights,
        "CLIP-ReID微调权重",
    )
    base_weights = require_file(
        settings.reid_clipreid_base_weights,
        "CLIP ViT-B/16基础权重",
    )
    device = select_device(torch, settings.reid_device, require_cuda=True)

    with isolated_source_imports(source_root, ("config", "model")):
        config_module = importlib.import_module("config")
        model_module = importlib.import_module("model.make_model_clipreid")
        clip_module = importlib.import_module("model.clip.clip")
        cfg = config_module.cfg.clone()
        cfg.merge_from_file(str(config_path))
        cfg.TEST.NECK_FEAT = "before"
        cfg.MODEL.SIE_CAMERA = False
        cfg.MODEL.SIE_VIEW = False

        original_download = clip_module._download
        clip_module._download = lambda *args, **kwargs: str(base_weights)
        try:
            model = model_module.make_model(
                cfg,
                num_class=settings.reid_clipreid_num_classes,
                camera_num=settings.reid_clipreid_camera_count,
                view_num=1,
            )
        finally:
            clip_module._download = original_download

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(state_dict_from_checkpoint(checkpoint), strict=True)

    return LoadedClipReid(model.eval().to(device), torch, device)


def embed(loaded: LoadedClipReid, image: Image.Image):
    tensor = image_tensor(
        loaded.torch,
        image,
        size=(128, 256),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ).to(loaded.device)
    with loaded.torch.inference_mode():
        output = loaded.model(tensor)
    return first_tensor(output).squeeze(0).detach().float().cpu().numpy()


def register(register_backend, settings) -> None:
    register_backend(
        "clipreid",
        lambda: load(settings),
        embed,
        dim=1280,
        label="CLIP-ReID ViT-B/16",
        description="官方MSMT17 ViT-CLIP-ReID，不依赖产品摄像头编号。",
        requires_cuda=True,
        experimental=True,
        replace=True,
    )
