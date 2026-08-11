from __future__ import annotations

from functools import wraps
from pathlib import Path

import numpy as np
from PIL import Image


OFFICIAL_WEIGHTS_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/"
    "v1.3.0/GFPGANv1.3.pth"
)


def _patch_basicsr() -> None:
    """Patch a removed torchvision compatibility import used by basicsr."""
    try:
        import torchvision.transforms.functional as functional
        import torchvision.transforms as transforms

        if not hasattr(transforms, "functional_tensor"):
            import sys
            import types

            module = types.ModuleType("torchvision.transforms.functional_tensor")
            module.rgb_to_grayscale = functional.rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = module
    except Exception:
        pass


def _weights_path(settings) -> str:
    configured = (settings.face_gfpgan_weights or "").strip()
    if configured and not configured.startswith(("http://", "https://")):
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GFPGAN weights not found: {path}")
        return str(path)

    url = configured or OFFICIAL_WEIGHTS_URL
    cache_dir = Path.home() / ".cache" / "gfpgan"
    cache_dir.mkdir(parents=True, exist_ok=True)
    from basicsr.utils.download_util import load_file_from_url

    return load_file_from_url(
        url=url,
        model_dir=str(cache_dir),
        progress=True,
    )


def _make_gfpgan_deterministic(restorer):
    generator = restorer.gfpgan
    if getattr(generator, "_deterministic_noise", False):
        return restorer
    original_forward = generator.forward

    @wraps(original_forward)
    def deterministic_forward(*args, **kwargs):
        kwargs["randomize_noise"] = False
        return original_forward(*args, **kwargs)

    generator.forward = deterministic_forward
    generator._deterministic_noise = True
    return restorer


def _load(settings):
    _patch_basicsr()
    from gfpgan import GFPGANer

    return _make_gfpgan_deterministic(
        GFPGANer(
            model_path=_weights_path(settings),
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
    )


def _enhance(
    restorer,
    image: Image.Image,
    aligned: bool,
) -> Image.Image | None:
    bgr = np.asarray(image.convert("RGB"))[:, :, ::-1]
    _, restored_faces, restored = restorer.enhance(
        bgr,
        has_aligned=aligned,
        only_center_face=not aligned,
        paste_back=not aligned,
    )
    if restored is None and restored_faces:
        restored = restored_faces[0]
    if restored is None:
        return None
    rgb = np.asarray(restored)[:, :, ::-1]
    return Image.fromarray(rgb)


def register(register_backend, settings) -> None:
    register_backend(
        "gfpgan",
        lambda: _load(settings),
        _enhance,
        replace=True,
    )


__all__ = [
    "OFFICIAL_WEIGHTS_URL",
    "_make_gfpgan_deterministic",
    "register",
]
