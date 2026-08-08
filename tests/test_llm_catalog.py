from __future__ import annotations

import io
import json
from types import SimpleNamespace
import urllib.error

import pytest

from app.core.config import settings
from app.services import llm_catalog


@pytest.fixture(autouse=True)
def reset_catalog_cache():
    llm_catalog.reset_event_llm_catalog()
    yield
    llm_catalog.reset_event_llm_catalog()


def _arm_response(items: list[dict]) -> io.BytesIO:
    return io.BytesIO(json.dumps({"value": items}).encode())


def test_arm_catalog_lists_deployed_targets_without_credentials(monkeypatch) -> None:
    observed = {}
    items = [
        {
            "name": "event-gpt41",
            "properties": {
                "model": {"name": "gpt-4.1"},
                "provisioningState": "Succeeded",
            },
        },
        {
            "name": "embedding",
            "properties": {
                "model": {"name": "text-embedding-3-large"},
                "provisioningState": "Succeeded",
            },
        },
        {
            "name": "pending",
            "properties": {
                "model": {"name": "gpt-5"},
                "provisioningState": "Creating",
            },
        },
    ]
    monkeypatch.setattr(
        llm_catalog,
        "_credential",
        lambda: SimpleNamespace(
            get_token=lambda scope: SimpleNamespace(token="not-a-real-token")
        ),
    )
    def open_arm(request, timeout):
        observed["authorization"] = request.get_header("Authorization")
        return _arm_response(items)

    monkeypatch.setattr(llm_catalog.urllib.request, "urlopen", open_arm)

    with settings.override(
        azure_openai_resource_id=(
            "/subscriptions/unit/resourceGroups/rg/providers/"
            "Microsoft.CognitiveServices/accounts/foundry"
        ),
        azure_openai_api_key=None,
    ):
        catalog = llm_catalog.event_llm_catalog()

    assert catalog["source"] == "arm"
    assert [target["deployment"] for target in catalog["callable_targets"]] == [
        "event-gpt41",
    ]
    assert catalog["callable_targets"][0]["label"] == "GPT-4.1"
    assert catalog["callable_targets"][0]["provider"] == "openai"
    assert catalog["catalog_models"] == [{"model": "gpt-4.1"}]
    assert observed["authorization"] == "Bearer not-a-real-token"
    assert "not-a-real-token" not in json.dumps(catalog)


def test_transient_arm_failure_keeps_last_successful_targets(monkeypatch) -> None:
    item = {
        "name": "event-gpt41",
        "properties": {
            "model": {"name": "gpt-4.1"},
            "provisioningState": "Succeeded",
        },
    }
    monkeypatch.setattr(
        llm_catalog,
        "_credential",
        lambda: SimpleNamespace(get_token=lambda scope: SimpleNamespace(token="token")),
    )
    monkeypatch.setattr(
        llm_catalog.urllib.request,
        "urlopen",
        lambda request, timeout: _arm_response([item]),
    )
    resource_id = (
        "/subscriptions/unit/resourceGroups/rg/providers/"
        "Microsoft.CognitiveServices/accounts/foundry"
    )
    with settings.override(azure_openai_resource_id=resource_id):
        first = llm_catalog.event_llm_catalog()
        monkeypatch.setattr(
            llm_catalog.urllib.request,
            "urlopen",
            lambda request, timeout: (_ for _ in ()).throw(
                urllib.error.URLError("offline")
            ),
        )
        fallback = llm_catalog.event_llm_catalog(refresh=True)

    assert first["source"] == "arm"
    assert fallback["source"] == "cache"
    assert fallback["callable_targets"] == first["callable_targets"]
    assert "temporarily failed" in fallback["warning"]


def test_configured_fallback_does_not_expose_unreviewed_models() -> None:
    with settings.override(
        azure_openai_resource_id=None,
        azure_openai_endpoint=None,
        azure_openai_api_key=None,
        azure_openai_deployment=None,
        event_llm_deployment=None,
        foundry_analysis_deployment="analysis-unit",
        foundry_analysis_model="unverified-model",
        foundry_chat_deployment=None,
    ):
        catalog = llm_catalog.event_llm_catalog()

    assert catalog["source"] == "configured"
    assert catalog["callable_targets"] == []


def test_reviewed_gpt5_metadata_is_callable_for_image_analysis() -> None:
    target = llm_catalog._target(
        "event-quality-gpt54",
        "gpt-5.4",
        "Succeeded",
        {"chatCompletion": "true"},
    )

    assert target == {
        "deployment": "event-quality-gpt54",
        "model": "gpt-5.4",
        "label": "GPT-5.4",
        "provider": "openai",
        "group": "Quality",
        "description": "Highest quality for complex visual evidence.",
        "context": "Large context",
        "performance": "Complex tasks",
        "capabilities": {"chat": True, "image_input": True},
    }
