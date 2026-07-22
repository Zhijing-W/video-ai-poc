from __future__ import annotations

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
