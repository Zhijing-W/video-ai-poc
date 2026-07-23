from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from app.identity.face.superres_backends import _common, codeformer, realesrgan


def _settings(**overrides):
    values = {
        "face_device": "cpu",
        "face_codeformer_weights": "codeformer.pth",
        "face_codeformer_fidelity": 0.7,
        "face_realesrgan_x2plus_weights": "realesrgan.pth",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _image(width=13, height=7):
    values = np.arange(width * height * 3, dtype=np.uint8).reshape(
        height, width, 3
    )
    return Image.fromarray(values, mode="RGB")


def test_codeformer_preprocess_shape_and_range():
    tensor = codeformer.preprocess(torch, _image())

    assert tensor.shape == (1, 3, 512, 512)
    assert tensor.dtype == torch.float32
    assert -1.0 <= tensor.min() <= tensor.max() <= 1.0


def test_realesrgan_preprocess_is_rgb_nchw_in_unit_range():
    image = Image.new("RGB", (2, 1))
    image.putdata([(255, 0, 128), (0, 64, 255)])

    tensor = realesrgan.preprocess(torch, image)

    assert tensor.shape == (1, 3, 1, 2)
    assert torch.equal(tensor[0, :, 0, 0], torch.tensor([1.0, 0.0, 128 / 255]))
    assert 0.0 <= tensor.min() <= tensor.max() <= 1.0


def test_codeformer_forwards_current_fidelity_and_returns_512_image():
    calls = []

    class Network:
        def __call__(self, tensor, *, weight, adain):
            calls.append((tensor.shape, weight, adain))
            return (torch.zeros((1, 3, 512, 512)),)

    loaded = codeformer.LoadedCodeFormer(
        descriptor=SimpleNamespace(model=Network()),
        torch=torch,
        device=torch.device("cpu"),
        settings=_settings(face_codeformer_fidelity=0.25),
    )

    original = _image()
    result = codeformer._enhance(loaded, original, aligned=True)

    assert calls == [((1, 3, 512, 512), 0.25, True)]
    assert isinstance(result, Image.Image)
    assert result.size == (512, 512)
    assert result is not original


@pytest.mark.parametrize("value", [-0.01, 1.01, "invalid", None])
def test_codeformer_rejects_invalid_fidelity(value):
    with pytest.raises(ValueError, match="fidelity"):
        codeformer.validate_fidelity(value)


def test_codeformer_requires_aligned_input():
    loaded = codeformer.LoadedCodeFormer(
        descriptor=SimpleNamespace(model=None),
        torch=torch,
        device=torch.device("cpu"),
        settings=_settings(),
    )

    with pytest.raises(ValueError, match="aligned"):
        codeformer._enhance(loaded, _image(), aligned=False)


def test_realesrgan_returns_model_output_size_and_new_image():
    class Descriptor:
        def __call__(self, tensor):
            assert tensor.shape == (1, 3, 7, 13)
            return torch.ones((1, 3, 14, 26))

    original = _image()
    loaded = realesrgan.LoadedRealESRGAN(
        descriptor=Descriptor(),
        torch=torch,
        device=torch.device("cpu"),
    )

    result = realesrgan._enhance(loaded, original, aligned=True)

    assert isinstance(result, Image.Image)
    assert result.size == (26, 14)
    assert result is not original


@pytest.mark.parametrize(
    ("module", "name"),
    [(codeformer, "codeformer"), (realesrgan, "realesrgan_x2plus")],
)
def test_registration_is_lazy_and_uses_registry_contract(monkeypatch, module, name):
    calls = []
    settings = _settings()

    def fake_load(received):
        calls.append(received)
        return object()

    monkeypatch.setattr(module, "_load", fake_load)
    registrations = []
    module.register(
        lambda *args, **kwargs: registrations.append((args, kwargs)),
        settings,
    )

    assert calls == []
    args, kwargs = registrations[0]
    assert args[0] == name
    assert callable(args[1])
    assert callable(args[2])
    assert kwargs == {"replace": True}
    assert args[1]() is not None
    assert calls == [settings]


def test_weight_download_uses_bounded_network_timeout(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []

    class Response:
        payload = b"weights"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            payload, self.payload = self.payload, b""
            return payload

    monkeypatch.setattr(_common.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        _common.urllib.request,
        "urlopen",
        lambda url, timeout: calls.append((url, timeout)) or Response(),
    )

    path = _common.resolve_weights(
        "https://example.test/model.pth",
        default_url="https://unused.test/default.pth",
        cache_namespace="timeout-test",
    )

    assert path.read_bytes() == b"weights"
    assert calls == [
        (
            "https://example.test/model.pth",
            _common.DOWNLOAD_TIMEOUT_SECONDS,
        )
    ]
