from __future__ import annotations

import importlib
import importlib.metadata
import math
import pkgutil
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from PIL import Image

from ...config import settings


BackendLoader = Callable[[], Any]
BackendEnhancer = Callable[..., Image.Image | None]
_PLUGIN_ENTRY_POINT_GROUP = "event_monitor.superres_backends"
_OPTION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_OPTION_KINDS = {"number", "integer", "boolean", "string", "select"}


@dataclass(frozen=True)
class SuperResolutionOption:
    name: str
    label: str
    kind: str = "string"
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[tuple[str, str], ...] = ()
    help_text: str | None = None

    @classmethod
    def from_value(
        cls,
        value: SuperResolutionOption | Mapping[str, Any],
    ) -> SuperResolutionOption:
        if isinstance(value, cls):
            option = value
        elif isinstance(value, Mapping):
            raw_choices = value.get("choices") or ()
            choices = tuple(
                (
                    str(choice["value"]),
                    str(choice.get("label", choice["value"])),
                )
                if isinstance(choice, Mapping)
                else (str(choice), str(choice))
                for choice in raw_choices
            )
            option = cls(
                name=str(value.get("name") or "").strip().lower(),
                label=str(value.get("label") or value.get("name") or "").strip(),
                kind=str(value.get("type") or value.get("kind") or "string")
                .strip()
                .lower(),
                default=value.get("default"),
                minimum=_optional_float(value.get("min", value.get("minimum"))),
                maximum=_optional_float(value.get("max", value.get("maximum"))),
                step=_optional_float(value.get("step")),
                choices=choices,
                help_text=(
                    str(value.get("help") or value.get("help_text")).strip()
                    if value.get("help") or value.get("help_text")
                    else None
                ),
            )
        else:
            raise TypeError("backend options must be mappings or SuperResolutionOption")
        option.validate_definition()
        return option

    def validate_definition(self) -> None:
        if not _OPTION_NAME.fullmatch(self.name):
            raise ValueError(
                f"invalid super-resolution option name: {self.name!r}"
            )
        if not self.label:
            raise ValueError(f"option {self.name!r} requires a label")
        if self.kind not in _OPTION_KINDS:
            raise ValueError(
                f"option {self.name!r} has unsupported type {self.kind!r}"
            )
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError(
                    f"option {self.name!r} minimum exceeds maximum"
                )
        if self.kind == "select" and not self.choices:
            raise ValueError(f"select option {self.name!r} requires choices")
        if self.default is not None:
            self.coerce(self.default)

    def coerce(self, value: Any) -> Any:
        if self.kind == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "on"}:
                    return True
                if lowered in {"0", "false", "no", "off"}:
                    return False
            raise ValueError(f"{self.label} must be a boolean")
        if self.kind in {"number", "integer"}:
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{self.label} must be numeric") from exc
            if not math.isfinite(number):
                raise ValueError(f"{self.label} must be finite")
            if self.kind == "integer":
                if not number.is_integer():
                    raise ValueError(f"{self.label} must be an integer")
                parsed: int | float = int(number)
            else:
                parsed = number
            if self.minimum is not None and parsed < self.minimum:
                raise ValueError(
                    f"{self.label} must be >= {self.minimum:g}"
                )
            if self.maximum is not None and parsed > self.maximum:
                raise ValueError(
                    f"{self.label} must be <= {self.maximum:g}"
                )
            return parsed
        parsed = str(value)
        if self.kind == "select":
            allowed = {choice[0] for choice in self.choices}
            if parsed not in allowed:
                raise ValueError(
                    f"{self.label} must be one of {', '.join(sorted(allowed))}"
                )
        return parsed

    def public_metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "type": self.kind,
            "default": (
                self.coerce(self.default)
                if self.default is not None
                else None
            ),
        }
        if self.minimum is not None:
            result["min"] = self.minimum
        if self.maximum is not None:
            result["max"] = self.maximum
        if self.step is not None:
            result["step"] = self.step
        if self.choices:
            result["choices"] = [
                {"value": value, "label": label}
                for value, label in self.choices
            ]
        if self.help_text:
            result["help"] = self.help_text
        return result


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class SuperResolutionBackend:
    name: str
    display_name: str
    loader: BackendLoader
    enhancer: BackendEnhancer
    options: tuple[SuperResolutionOption, ...] = ()
    accepts_options: bool = False


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
_plugin_errors: dict[str, str] = {}
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
    display_name: str | None = None,
    options: Sequence[
        SuperResolutionOption | Mapping[str, Any]
    ] = (),
    accepts_options: bool = False,
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
    parsed_options = tuple(
        SuperResolutionOption.from_value(option) for option in options
    )
    option_names = [option.name for option in parsed_options]
    if len(set(option_names)) != len(option_names):
        raise ValueError(f"duplicate options for backend {normalized!r}")
    with _registry_lock:
        if normalized in _backends and not replace:
            raise ValueError(f"超分后端已注册：{normalized}")
        _backends[normalized] = SuperResolutionBackend(
            name=normalized,
            display_name=(display_name or normalized).strip(),
            loader=loader,
            enhancer=enhancer,
            options=parsed_options,
            accepts_options=accepts_options,
        )
        _states[normalized] = _BackendState()


def available_backends() -> tuple[str, ...]:
    with _registry_lock:
        return tuple(sorted(_backends))


def backend_metadata() -> dict[str, dict[str, Any]]:
    with _registry_lock:
        return {
            name: {
                "label": backend.display_name,
                "parameters": [
                    option.public_metadata() for option in backend.options
                ],
            }
            for name, backend in sorted(_backends.items())
        }


def plugin_errors() -> dict[str, str]:
    with _registry_lock:
        return dict(_plugin_errors)


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


def validate_backend_options(
    backend: str | None,
    values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = validate_backend(backend)
    raw = dict(values or {})
    if normalized == "off":
        if raw:
            raise ValueError("关闭超分时不能设置后端参数")
        return {}
    with _registry_lock:
        spec = _backends[normalized]
    definitions = {option.name: option for option in spec.options}
    unknown = sorted(set(raw) - set(definitions))
    if unknown:
        raise ValueError(
            f"{spec.display_name} 不支持参数：{', '.join(unknown)}"
        )
    parsed = {}
    for name, option in definitions.items():
        if name in raw:
            parsed[name] = option.coerce(raw[name])
        elif option.default is not None:
            parsed[name] = option.coerce(option.default)
    return parsed


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
    options: Mapping[str, Any] | None = None,
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
    selected_options = validate_backend_options(
        normalized,
        settings.face_superres_options if options is None else options,
    )
    model = _ensure_superres(normalized)
    if model is None or spec is None or state is None:
        return image
    try:
        restored = (
            spec.enhancer(model, image, aligned, selected_options)
            if spec.accepts_options
            else spec.enhancer(model, image, aligned)
        )
        if restored is None:
            return image
        if not isinstance(restored, Image.Image):
            raise TypeError("backend enhancer must return PIL.Image or None")
        state.enhance_error = None
        return restored
    except Exception as exc:  # noqa: BLE001
        state.enhance_error = f"{type(exc).__name__}: {exc}"
        return image


def _register_plugin(plugin: Any, source: str) -> None:
    register = getattr(plugin, "register", None)
    if register is None and callable(plugin):
        register = plugin
    if not callable(register):
        raise TypeError(
            f"{source} must expose register(register_backend, settings)"
        )
    register(register_backend, settings)


def _builtin_plugin_names() -> list[str]:
    package = importlib.import_module(f"{__package__}.superres_backends")
    prefix = f"{package.__name__}."
    return [
        info.name
        for info in sorted(
        pkgutil.iter_modules(package.__path__, prefix),
        key=lambda item: item.name,
        )
        if not info.name.rsplit(".", 1)[-1].startswith("_")
    ]


def _external_plugin_entry_points():
    discovered = importlib.metadata.entry_points()
    if hasattr(discovered, "select"):
        return tuple(discovered.select(group=_PLUGIN_ENTRY_POINT_GROUP))
    return tuple(discovered.get(_PLUGIN_ENTRY_POINT_GROUP, ()))


def discover_backends() -> None:
    """Discover bundled adapters and installed Python entry-point plugins."""
    try:
        builtin_names = _builtin_plugin_names()
    except Exception as exc:  # noqa: BLE001
        with _registry_lock:
            _plugin_errors["builtins"] = f"{type(exc).__name__}: {exc}"
        builtin_names = []
    for module_name in builtin_names:
        try:
            _register_plugin(
                importlib.import_module(module_name),
                module_name,
            )
            with _registry_lock:
                _plugin_errors.pop(module_name, None)
        except Exception as exc:  # noqa: BLE001
            with _registry_lock:
                _plugin_errors[module_name] = (
                    f"{type(exc).__name__}: {exc}"
                )
    try:
        entry_points = _external_plugin_entry_points()
    except Exception as exc:  # noqa: BLE001
        with _registry_lock:
            _plugin_errors["entrypoints"] = f"{type(exc).__name__}: {exc}"
        entry_points = ()
    for entry_point in entry_points:
        source = f"entrypoint:{entry_point.name}"
        try:
            _register_plugin(entry_point.load(), source)
            with _registry_lock:
                _plugin_errors.pop(source, None)
        except Exception as exc:  # noqa: BLE001
            with _registry_lock:
                _plugin_errors[source] = f"{type(exc).__name__}: {exc}"


discover_backends()


def _make_gfpgan_deterministic(restorer):
    from .superres_backends.gfpgan import _make_gfpgan_deterministic as wrap

    return wrap(restorer)


__all__ = [
    "SuperResolutionBackend",
    "SuperResolutionOption",
    "_ensure_superres",
    "available_backends",
    "backend_metadata",
    "discover_backends",
    "enhance",
    "plugin_errors",
    "register_backend",
    "reset_backend",
    "superres_error",
    "validate_backend",
    "validate_backend_options",
]
