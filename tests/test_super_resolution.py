from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from app.identity.face import super_resolution


class _FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def forward(self, *args, **kwargs):
        self.calls.append(kwargs)
        return kwargs


class _FakeRestorer:
    def __init__(self) -> None:
        self.gfpgan = _FakeGenerator()


def test_gfpgan_forward_always_disables_random_noise() -> None:
    restorer = super_resolution._make_gfpgan_deterministic(_FakeRestorer())

    result = restorer.gfpgan.forward(
        "face",
        return_rgb=False,
        randomize_noise=True,
    )

    assert result["randomize_noise"] is False
    assert restorer.gfpgan.calls == [
        {"return_rgb": False, "randomize_noise": False}
    ]


def test_gfpgan_deterministic_wrapper_is_idempotent() -> None:
    restorer = _FakeRestorer()

    first = super_resolution._make_gfpgan_deterministic(restorer)
    second = super_resolution._make_gfpgan_deterministic(restorer)
    second.gfpgan.forward("face")

    assert first is second
    assert len(restorer.gfpgan.calls) == 1


def test_registered_backend_is_selected_explicitly_and_loaded_once() -> None:
    loads = []
    calls = []
    model = object()

    def load():
        loads.append("load")
        return model

    def enhance(loaded, image, aligned):
        calls.append((loaded, image.size, aligned))
        return Image.new("RGB", image.size, (12, 34, 56))

    super_resolution.register_backend(
        "unit-test",
        load,
        enhance,
        replace=True,
    )
    original = Image.new("RGB", (16, 16), "black")

    first = super_resolution.enhance(
        original,
        aligned=True,
        backend="unit-test",
    )
    second = super_resolution.enhance(
        original,
        aligned=False,
        backend="unit_test",
    )

    assert first is not original
    assert second is not original
    assert loads == ["load"]
    assert calls == [
        (model, (16, 16), True),
        (model, (16, 16), False),
    ]
    assert "unit_test" in super_resolution.available_backends()


def test_unknown_backend_is_rejected_before_inference() -> None:
    with pytest.raises(ValueError, match="未知人脸超分后端"):
        super_resolution.enhance(
            Image.new("RGB", (8, 8)),
            backend="not-registered",
        )


def test_backend_load_errors_are_isolated() -> None:
    super_resolution.register_backend(
        "unit-broken",
        lambda: (_ for _ in ()).throw(RuntimeError("broken")),
        lambda model, image, aligned: image,
        replace=True,
    )
    super_resolution.register_backend(
        "unit-healthy",
        lambda: object(),
        lambda model, image, aligned: Image.new("RGB", image.size, "white"),
        replace=True,
    )
    original = Image.new("RGB", (8, 8), "black")

    assert super_resolution.enhance(
        original,
        backend="unit-broken",
    ) is original
    assert "broken" in (super_resolution.superres_error("unit-broken") or "")
    assert super_resolution.enhance(
        original,
        backend="unit-healthy",
    ) is not original
    assert super_resolution.superres_error("unit-healthy") is None


def test_off_backend_is_a_no_op() -> None:
    original = Image.new("RGB", (8, 8), "black")

    assert super_resolution.enhance(original, backend="off") is original
    assert super_resolution.validate_backend("none") == "off"


def test_concurrent_calls_load_registered_backend_once() -> None:
    workers = 8
    barrier = threading.Barrier(workers)
    counter_lock = threading.Lock()
    load_count = 0

    def load():
        nonlocal load_count
        with counter_lock:
            load_count += 1
        time.sleep(0.03)
        return object()

    super_resolution.register_backend(
        "unit-concurrent",
        load,
        lambda model, image, aligned: image.copy(),
        replace=True,
    )

    def run():
        barrier.wait()
        return super_resolution.enhance(
            Image.new("RGB", (8, 8)),
            backend="unit-concurrent",
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        outputs = list(executor.map(lambda _: run(), range(workers)))

    assert load_count == 1
    assert all(isinstance(output, Image.Image) for output in outputs)
