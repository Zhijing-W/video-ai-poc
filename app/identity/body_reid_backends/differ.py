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
class LoadedDiffer:
    model: Any
    torch: Any
    device: Any
    dim: int = 1024


def _infer_classifier_count(state_dict: dict[str, Any]) -> int:
    for key, value in state_dict.items():
        if key.endswith("head.weight") and getattr(value, "ndim", 0) == 2:
            return int(value.shape[0])
    raise ValueError("无法从DIFFER checkpoint推断训练身份数")


def _infer_camera_count(state_dict: dict[str, Any]) -> int:
    for key, value in state_dict.items():
        if key.endswith("camera_embed") and getattr(value, "ndim", 0) >= 2:
            return int(value.shape[0])
    raise ValueError("无法从DIFFER checkpoint推断摄像头数量")


def load(settings) -> LoadedDiffer:
    import torch

    source_root = require_directory(settings.reid_differ_root, "DIFFER官方源码目录")
    config_path = require_file(settings.reid_differ_config, "DIFFER配置")
    checkpoint_path = require_file(settings.reid_differ_weights, "DIFFER权重")
    device = select_device(torch, settings.reid_device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = state_dict_from_checkpoint(checkpoint)
    num_classes = _infer_classifier_count(state_dict)
    camera_count = _infer_camera_count(state_dict)

    with isolated_source_imports(source_root, ("config", "model")):
        config_module = importlib.import_module("config")
        build_module = importlib.import_module("model.build")
        cfg = config_module.cfg.clone()
        cfg.merge_from_file(str(config_path))
        # Product crops have no stable training-camera ID. Disable the learned
        # camera offset instead of assigning every deployment image to camera 0.
        cfg.MODEL.CAMERA_XISHU = 0
        factory = build_module.factory[cfg.MODEL.NAME]
        model = factory(
            pretrained=False,
            config=cfg,
            num_classes=num_classes,
            camera_num=camera_count,
        )
        model.load_state_dict(state_dict, strict=True)

    return LoadedDiffer(model.eval().to(device), torch, device)


def embed(loaded: LoadedDiffer, image: Image.Image):
    tensor = image_tensor(
        loaded.torch,
        image,
        size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ).to(loaded.device)
    camera = loaded.torch.zeros((1,), dtype=loaded.torch.long, device=loaded.device)
    with loaded.torch.inference_mode():
        output = loaded.model(tensor, cam_label=camera)
    return first_tensor(output).squeeze(0).detach().float().cpu().numpy()


def register(register_backend, settings) -> None:
    register_backend(
        "differ",
        lambda: load(settings),
        embed,
        dim=1024,
        label="DIFFER EVA02-L",
        description="CVPR 2025换衣ReID模型，使用LTCC官方checkpoint。",
        experimental=True,
        replace=True,
    )
