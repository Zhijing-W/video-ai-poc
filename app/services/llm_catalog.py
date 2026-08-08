"""Discover callable Microsoft Foundry deployments without exposing credentials."""
from __future__ import annotations

from copy import deepcopy
import json
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

from ..core.config import settings

_ARM_API_VERSION = "2023-05-01"
_OPENAI_API_VERSION = "2023-03-15-preview"
_CACHE_SECONDS = 300.0
_RESOURCE_ID = re.compile(
    r"^/subscriptions/[^/?#]+/resourceGroups/[^/?#]+/"
    r"providers/Microsoft\.CognitiveServices/accounts/[^/?#]+$",
    re.IGNORECASE,
)
_cache_lock = threading.Lock()
_success_cache: tuple[float, dict[str, Any]] | None = None


class _DiscoveryError(RuntimeError):
    """A transient discovery failure that must not replace a good catalog."""


def _resolved_auth_mode() -> str:
    if settings.azure_openai_auth == "auto":
        return "api_key" if settings.azure_openai_api_key else "managed_identity"
    return settings.azure_openai_auth


def _model_capabilities(model: str | None, raw: object = None) -> dict[str, bool]:
    """Return only capabilities which can be established from deployment metadata."""
    value = (model or "").strip().lower()
    capabilities = raw if isinstance(raw, dict) else {}
    normalized = {str(key).lower(): str(item).lower() for key, item in capabilities.items()}

    def explicit(*names: str) -> bool | None:
        for name in names:
            if name.lower() in normalized:
                return normalized[name.lower()] in {"true", "1", "yes", "enabled"}
        return None

    known_chat = value.startswith(
        ("gpt-35", "gpt-3.5", "gpt-4", "gpt-5", "o1", "o3", "o4")
    )
    known_image = value.startswith(("gpt-4o", "gpt-4.1", "gpt-4.5", "gpt-5"))
    chat = explicit("chat", "chatcompletion", "chat_completions")
    image = explicit("imageinput", "image_input", "vision")
    return {
        "chat": known_chat if chat is None else chat,
        "image_input": known_image if image is None else image,
    }


def _target(
    deployment: object,
    model: object,
    status: object,
    capabilities: object = None,
) -> dict[str, Any] | None:
    deployment_name = str(deployment or "").strip()
    model_name = str(model or deployment_name).strip()
    state = str(status or "").strip().lower()
    if not deployment_name or state not in {"succeeded", "success"}:
        return None
    return {
        "deployment": deployment_name,
        "model": model_name,
        "label": (
            deployment_name
            if deployment_name == model_name
            else f"{deployment_name} ({model_name})"
        ),
        "capabilities": _model_capabilities(model_name, capabilities),
    }


def _dedupe_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for item in targets:
        if item:
            unique.setdefault(item["deployment"], item)
    return sorted(unique.values(), key=lambda item: item["label"].lower())


def _configured_targets() -> list[dict[str, Any]]:
    configured = (
        (settings.foundry_analysis_deployment, settings.foundry_analysis_model),
        (settings.foundry_chat_deployment, settings.foundry_chat_model),
        (settings.event_llm_deployment, settings.foundry_analysis_model),
        (settings.azure_openai_deployment, settings.foundry_analysis_model),
    )
    targets = [
        _target(deployment, model, "succeeded")
        for deployment, model in configured
        if deployment
    ]
    return _dedupe_targets([item for item in targets if item])


def _catalog(
    targets: list[dict[str, Any]],
    *,
    source: str,
    warning: str | None = None,
) -> dict[str, Any]:
    # A model name is metadata; only callable_targets contain deployment names
    # and only those targets are allowed through request validation.
    models = sorted(
        {item["model"] for item in targets if item.get("model")},
        key=str.lower,
    )
    return {
        "source": source,
        "warning": warning,
        "catalog_models": [{"model": model} for model in models],
        "callable_targets": targets,
    }


def _credential():
    client_id = settings.azure_openai_managed_identity_client_id
    if client_id:
        return ManagedIdentityCredential(client_id=client_id)
    return DefaultAzureCredential(exclude_interactive_browser_credential=True)


def _fetch_arm_targets() -> list[dict[str, Any]]:
    resource_id = (settings.azure_openai_resource_id or "").rstrip("/")
    if not resource_id:
        raise _DiscoveryError("ARM discovery is not configured")
    if not _RESOURCE_ID.fullmatch(resource_id):
        raise _DiscoveryError("AZURE_OPENAI_RESOURCE_ID is invalid")
    try:
        token = _credential().get_token("https://management.azure.com/.default").token
        url = (
            f"https://management.azure.com{resource_id}/deployments"
            f"?api-version={_ARM_API_VERSION}"
        )
        items: list[dict[str, Any]] = []
        while url:
            request = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.load(response)
            items.extend(payload.get("value", []))
            next_link = payload.get("nextLink")
            if next_link and not str(next_link).startswith(
                "https://management.azure.com/"
            ):
                raise _DiscoveryError("ARM returned an invalid pagination link")
            url = str(next_link or "")
    except (AzureError, OSError, ValueError, urllib.error.HTTPError) as exc:
        raise _DiscoveryError(f"ARM discovery failed: {type(exc).__name__}") from exc

    targets = []
    for item in items:
        properties = item.get("properties") or {}
        model = properties.get("model") or {}
        targets.append(
            _target(
                item.get("name"),
                model.get("name") if isinstance(model, dict) else model,
                properties.get("provisioningState"),
                properties.get("capabilities"),
            )
        )
    return _dedupe_targets([item for item in targets if item])


def _fetch_api_targets() -> list[dict[str, Any]]:
    """Compatibility path for local API-key development only."""
    endpoint = (settings.azure_openai_endpoint or "").rstrip("/")
    api_key = settings.azure_openai_api_key
    if not endpoint or not api_key:
        raise _DiscoveryError("No ARM resource ID or local API-key discovery is configured")
    request = urllib.request.Request(
        f"{endpoint}/openai/deployments?api-version={_OPENAI_API_VERSION}",
        headers={"api-key": api_key},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        raise _DiscoveryError(
            f"Azure OpenAI deployment discovery failed: {type(exc).__name__}"
        ) from exc

    targets = []
    for item in payload.get("data", []):
        properties = item.get("properties") or {}
        model = item.get("model") or (properties.get("model") or {}).get("name")
        targets.append(
            _target(
                item.get("id") or item.get("name"),
                model,
                item.get("status") or properties.get("provisioningState"),
                properties.get("capabilities"),
            )
        )
    return _dedupe_targets([item for item in targets if item])


def _discover() -> dict[str, Any]:
    if settings.azure_openai_resource_id:
        return _catalog(_fetch_arm_targets(), source="arm")
    return _catalog(_fetch_api_targets(), source="azure_openai_api")


def event_llm_catalog(*, refresh: bool = False) -> dict[str, Any]:
    """Return server-discovered callable targets, retaining a good cache on errors."""
    global _success_cache
    now = time.monotonic()
    with _cache_lock:
        if not refresh and _success_cache and now - _success_cache[0] < _CACHE_SECONDS:
            return deepcopy(_success_cache[1])

    try:
        catalog = _discover()
    except _DiscoveryError as exc:
        with _cache_lock:
            if _success_cache:
                cached = deepcopy(_success_cache[1])
                cached["source"] = "cache"
                cached["warning"] = "Deployment discovery temporarily failed; using cached targets"
                return cached
        return _catalog(
            _configured_targets(),
            source="configured",
            warning="Deployment discovery is unavailable; using configured compatible targets",
        )

    with _cache_lock:
        _success_cache = (now, catalog)
    return deepcopy(catalog)


def reset_event_llm_catalog() -> None:
    global _success_cache
    with _cache_lock:
        _success_cache = None


__all__ = ["event_llm_catalog", "reset_event_llm_catalog"]
