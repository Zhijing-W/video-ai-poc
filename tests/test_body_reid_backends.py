from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from app import body_reid
from app.config import settings
from app.identity.body_reid_backends import clipreid, differ, siglip2
from app.identity.body_reid_backends._common import image_tensor


def _image(width=17, height=31):
    values = np.arange(width * height * 3, dtype=np.uint8).reshape(
        height,
        width,
        3,
    )
    return Image.fromarray(values, mode="RGB")


def test_registry_lists_new_backends_and_normalizes_aliases():
    assert {"clipreid", "siglip2", "differ"} <= set(
        body_reid.available_backends()
    )
    assert body_reid.validate_backend("clip-reid") == "clipreid"
    assert body_reid.backend_metadata()["differ"]["dim"] == 1024

    with pytest.raises(ValueError, match="未知人形ReID后端"):
        body_reid.validate_backend("missing")


def test_explicit_backend_failure_does_not_fall_back():
    def fail():
        raise FileNotFoundError("missing checkpoint")

    body_reid.register_backend(
        "unit-failure",
        fail,
        lambda model, image: np.ones(3),
        dim=3,
        replace=True,
    )
    body_reid.reset_backend()
    try:
        with settings.override(reid_backend="unit-failure"):
            with pytest.raises(RuntimeError, match="unit_failure.*加载失败"):
                body_reid.active_backend()
    finally:
        body_reid.reset_backend()


@pytest.mark.parametrize(
    ("setting", "loader", "label"),
    [
        ("reid_resnet50_weights", body_reid._load_resnet50, "ResNet50"),
        ("reid_osnet_weights", body_reid._load_osnet, "OSNet"),
    ],
)
def test_builtin_reid_loaders_never_download_missing_weights(
    tmp_path,
    setting,
    loader,
    label,
):
    with settings.override(**{setting: str(tmp_path / "missing.pth")}):
        with pytest.raises(FileNotFoundError, match=label):
            loader()


def test_auto_is_the_only_mode_that_falls_back(monkeypatch):
    body_reid.register_backend(
        "unit-auto-failure",
        lambda: (_ for _ in ()).throw(RuntimeError("unavailable")),
        lambda model, image: np.ones(3),
        dim=3,
        replace=True,
    )
    monkeypatch.setattr(
        body_reid,
        "_AUTO_ORDER",
        ("unit_auto_failure", "coarse"),
    )
    body_reid.reset_backend()
    try:
        with settings.override(reid_backend="auto"):
            assert body_reid.active_backend() == "coarse"
    finally:
        body_reid.reset_backend()


def test_registered_backend_output_is_float32_l2_normalized():
    loaded = SimpleNamespace(dim=3)
    body_reid.register_backend(
        "unit-vector",
        lambda: loaded,
        lambda model, image: [3.0, 4.0, 0.0],
        dim=None,
        replace=True,
    )
    body_reid.reset_backend()
    try:
        with settings.override(reid_backend="unit-vector"):
            value = body_reid.embed(_image())
            assert value.dtype == np.float32
            assert value.shape == (3,)
            assert np.isclose(np.linalg.norm(value), 1.0)
            assert body_reid.embed_dim() == 3
    finally:
        body_reid.reset_backend()


def test_clipreid_preprocessing_uses_official_shape_and_normalization():
    calls = []

    class Model:
        def __call__(self, tensor):
            calls.append(tensor)
            return torch.tensor([[3.0, 4.0]])

    loaded = clipreid.LoadedClipReid(
        model=Model(),
        torch=torch,
        device=torch.device("cpu"),
        dim=2,
    )
    value = clipreid.embed(loaded, Image.new("RGB", (10, 20), (255, 0, 127)))

    assert calls[0].shape == (1, 3, 256, 128)
    assert torch.isclose(calls[0][0, 0, 0, 0], torch.tensor(1.0))
    assert torch.isclose(calls[0][0, 1, 0, 0], torch.tensor(-1.0))
    assert value.tolist() == [3.0, 4.0]


def test_siglip2_uses_processor_and_image_features():
    calls = []

    class Processor:
        def __call__(self, *, images, return_tensors):
            calls.append((images.mode, return_tensors))
            return {"pixel_values": torch.zeros((1, 3, 224, 224))}

    class Model:
        def get_image_features(self, **inputs):
            assert inputs["pixel_values"].shape == (1, 3, 224, 224)
            return torch.tensor([[1.0, 2.0, 3.0]])

    loaded = siglip2.LoadedSiglip2(
        model=Model(),
        processor=Processor(),
        torch=torch,
        device=torch.device("cpu"),
        dim=3,
    )

    assert siglip2.embed(loaded, _image()).tolist() == [1.0, 2.0, 3.0]
    assert calls == [("RGB", "pt")]


def test_differ_preprocessing_uses_imagenet_and_neutral_camera():
    calls = []

    class Model:
        def __call__(self, tensor, *, cam_label):
            calls.append((tensor, cam_label))
            return torch.tensor([[5.0, 12.0]])

    loaded = differ.LoadedDiffer(
        model=Model(),
        torch=torch,
        device=torch.device("cpu"),
        dim=2,
    )
    value = differ.embed(loaded, Image.new("RGB", (32, 64), "white"))

    tensor, camera = calls[0]
    assert tensor.shape == (1, 3, 224, 224)
    assert camera.tolist() == [0]
    assert value.tolist() == [5.0, 12.0]


def test_differ_infers_checkpoint_dimensions():
    state = {
        "head.weight": torch.zeros((152, 1024)),
        "camera_embed": torch.zeros((12, 1024)),
    }

    assert differ._infer_classifier_count(state) == 152
    assert differ._infer_camera_count(state) == 12


def test_image_tensor_resizes_width_then_height():
    tensor = image_tensor(
        torch,
        _image(),
        size=(11, 23),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    assert tensor.shape == (1, 3, 23, 11)
