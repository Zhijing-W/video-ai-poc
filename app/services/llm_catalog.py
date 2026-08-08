from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from ..core.config import settings


_CATALOG_API_VERSION = "2023-03-15-preview"
_CACHE_SECONDS = 300.0
_cache_lock = threading.Lock()
_cache: tuple[float, dict[str, Any]] | None = None


def _configured_default() -> str | None:
    return settings.event_llm_deployment or settings.azure_openai_deployment


def _supports_image_input(model: str | None) -> bool:
    normalized = (model or "").strip().lower()
    return normalized.startswith(("gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-5"))


def _configured_catalog(warning: str | None = None) -> dict[str, Any]:
    default = _configured_default()
    return {
        "default": default,
        "models": [],
        "source": "configured",
        "warning": warning,
    }


def _fetch_catalog() -> dict[str, Any]:
    endpoint = (settings.azure_openai_endpoint or "").rstrip("/")
    api_key = settings.azure_openai_api_key
    if not endpoint or not api_key:
        return _configured_catalog("Azure OpenAI deployment discovery is not configured")

    request = urllib.request.Request(
        f"{endpoint}/openai/deployments?api-version={_CATALOG_API_VERSION}",
        headers={"api-key": api_key},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        return _configured_catalog(
            f"Azure OpenAI deployment discovery failed: {type(exc).__name__}"
        )

    default = _configured_default()
    models_by_deployment: dict[str, dict[str, Any]] = {}
    for item in payload.get("data", []):
        deployment = str(item.get("id") or item.get("name") or "").strip()
        model = str(
            item.get("model")
            or item.get("properties", {}).get("model", {}).get("name")
            or deployment
        ).strip()
        status = str(
            item.get("status")
            or item.get("properties", {}).get("provisioningState")
            or ""
        ).strip().lower()
        if not deployment or status not in {"succeeded", "success"}:
            continue
        if not _supports_image_input(model):
            continue
        label = deployment if deployment == model else f"{deployment} ({model})"
        models_by_deployment[deployment] = {
            "deployment": deployment,
            "model": model,
            "label": label,
            "default": deployment == default,
        }

    models = sorted(
        models_by_deployment.values(),
        key=lambda item: item["label"].lower(),
    )
    warning = None
    if default not in models_by_deployment:
        if models:
            default = models[0]["deployment"]
            warning = (
                "Configured default is unavailable; using the first active "
                "vision-capable deployment"
            )
        else:
            default = None
            warning = "No active vision-capable Azure OpenAI deployment was found"
    for item in models:
        item["default"] = item["deployment"] == default
    models.sort(key=lambda item: (not item["default"], item["label"].lower()))
    return {
        "default": default,
        "models": models,
        "source": "azure",
        "warning": warning,
    }


def event_llm_catalog(*, refresh: bool = False) -> dict[str, Any]:
    global _cache

    now = time.monotonic()
    with _cache_lock:
        if not refresh and _cache and now - _cache[0] < _CACHE_SECONDS:
            return _cache[1]

    catalog = _fetch_catalog()
    with _cache_lock:
        _cache = (now, catalog)
    return catalog


def resolve_event_llm_deployment(value: str | None) -> str | None:
    requested = (value or "").strip()
    catalog = event_llm_catalog()
    default = catalog.get("default")
    if not requested:
        return default

    allowed = {
        item["deployment"]
        for item in catalog.get("models", [])
        if item.get("deployment")
    }
    if catalog.get("source") == "configured" and requested == default:
        return requested
    if requested not in allowed:
        raise ValueError(f"未知或不可用的事件分析 AI deployment：{requested}")
    return requested


def reset_event_llm_catalog() -> None:
    global _cache
    with _cache_lock:
        _cache = None


__all__ = [
    "event_llm_catalog",
    "reset_event_llm_catalog",
    "resolve_event_llm_deployment",
]
