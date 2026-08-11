from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image

from ...config import settings
from .superres_backends import codeformer, gfpgan, realesrgan


BackendLoader = Callable[[], Any]
BackendEnhancer = Callable[[Any, Image.Image, bool], Image.Image | None]


@dataclass(frozen=True)
class SuperResolutionBackend:
    name: str
    loader: BackendLoader
    enhancer: BackendEnhancer


@dataclass
class _BackendState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    ready: bool = False
    model: Any = None
    load_error: str | None = None
    enhance_error: str | None = None


_registry_lock = threading.RLock()
_backends: dict[str, SuperResolutionBackend] = {}
_states: dict[str, _BackendState] = {}
_DISABLED_BACKENDS = {"", "off", "none", "disabled"}
_ALIASES = {"gfp_gan": "gfpgan"}


def _normalize_backend_name(value: str | None = None) -> str:
    raw = settings.face_superres if value is None else value
    name = str(raw or "").strip().lower().replace("-", "_")
    if name in _DISABLED_BACKENDS:
        return "off"
    return _ALIASES.get(name, name)


def register_backend(
    name: str,
    loader: BackendLoader,
    enhancer: BackendEnhancer,
    *,
    replace: bool = False,
) -> None:
    """Register a lazy face-restoration backend.

    The enhancer receives ``(loaded_model, PIL_image, aligned)`` and returns a
    new PIL image, or ``None`` when no restoration output is available.
    """
    normalized = _normalize_backend_name(name)
    if normalized == "off":
        raise ValueError("'off' is reserved and cannot be registered")
    if not callable(loader) or not callable(enhancer):
        raise TypeError("loader and enhancer must be callable")
    with _registry_lock:
        if normalized in _backends and not replace:
            raise ValueError(f"超分后端已注册：{normalized}")
        _backends[normalized] = SuperResolutionBackend(
            name=normalized,
            loader=loader,
            enhancer=enhancer,
        )
        _states[normalized] = _BackendState()


def available_backends() -> tuple[str, ...]:
    with _registry_lock:
        return tuple(sorted(_backends))


def validate_backend(value: str | None = None) -> str:
    normalized = _normalize_backend_name(value)
    if normalized == "off":
        return normalized
    with _registry_lock:
        if normalized not in _backends:
            choices = ", ".join(("off", *sorted(_backends)))
            raise ValueError(
                f"未知人脸超分后端：{value!r}；可选 {choices}"
            )
    return normalized


def reset_backend(value: str | None = None) -> None:
    """Drop loaded model/error state while keeping backend registration."""
    if value is None:
        with _registry_lock:
            for name in _backends:
                _states[name] = _BackendState()
        return
    normalized = validate_backend(value)
    if normalized == "off":
        return
    with _registry_lock:
        _states[normalized] = _BackendState()


def _backend_and_state(
    value: str | None,
) -> tuple[str, SuperResolutionBackend | None, _BackendState | None]:
    normalized = validate_backend(value)
    if normalized == "off":
        return normalized, None, None
    with _registry_lock:
        return normalized, _backends[normalized], _states[normalized]


def _ensure_superres(backend: str | None = None):
    """Lazily load the selected registered backend."""
    _, spec, state = _backend_and_state(backend)
    if spec is None or state is None:
        return None
    if state.ready or state.load_error is not None:
        return state.model
    with state.lock:
        if state.ready or state.load_error is not None:
            return state.model
        try:
            model = spec.loader()
            if model is None:
                raise RuntimeError("backend loader returned no model")
            state.model = model
            state.ready = True
        except Exception as exc:  # noqa: BLE001
            state.load_error = f"{type(exc).__name__}: {exc}"
        return state.model


def superres_error(backend: str | None = None) -> str | None:
    _, _, state = _backend_and_state(backend)
    if state is None:
        return None
    return state.load_error or state.enhance_error


def enhance(
    image,
    *,
    aligned: bool = False,
    backend: str | None = None,
):
    """Restore a face with an explicit backend or ``settings.face_superres``.

    Selection is independent of the product quality gate: callers decide
    whether a face should be restored; this function only dispatches the
    requested algorithm.
    """
    normalized, spec, state = _backend_and_state(backend)
    if normalized == "off":
        return image
    if not isinstance(image, Image.Image):
        return image
    model = _ensure_superres(normalized)
    if model is None or spec is None or state is None:
        return image
    try:
        restored = spec.enhancer(model, image, aligned)
        if restored is None:
            return image
        if not isinstance(restored, Image.Image):
            raise TypeError("backend enhancer must return PIL.Image or None")
        state.enhance_error = None
        return restored
    except Exception as exc:  # noqa: BLE001
        state.enhance_error = f"{type(exc).__name__}: {exc}"
        return image


_make_gfpgan_deterministic = gfpgan._make_gfpgan_deterministic
gfpgan.register(register_backend, settings)
codeformer.register(register_backend, settings)
realesrgan.register(register_backend, settings)


__all__ = [
    "SuperResolutionBackend",
    "_ensure_superres",
    "available_backends",
    "enhance",
    "register_backend",
    "reset_backend",
    "superres_error",
    "validate_backend",
]
