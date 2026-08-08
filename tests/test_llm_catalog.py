from __future__ import annotations

import io
import json

import pytest

from app.core.config import settings
from app.services import llm_catalog


@pytest.fixture(autouse=True)
def reset_catalog_cache():
    llm_catalog.reset_event_llm_catalog()
    yield
    llm_catalog.reset_event_llm_catalog()


def test_catalog_lists_only_successful_vision_deployments(monkeypatch) -> None:
    payload = {
        "data": [
            {"id": "gpt-4o", "model": "gpt-4o", "status": "succeeded"},
            {
                "id": "event-gpt41",
                "model": "gpt-4.1",
                "status": "succeeded",
            },
            {
                "id": "embedding",
                "model": "text-embedding-3-large",
                "status": "succeeded",
            },
            {"id": "pending", "model": "gpt-5", "status": "creating"},
        ]
    }
    monkeypatch.setattr(
        llm_catalog.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(payload).encode()),
    )

    with settings.override(
        azure_openai_endpoint="https://unit.openai.azure.com",
        azure_openai_api_key="not-a-real-key",
        azure_openai_deployment="gpt-4o",
        event_llm_deployment="",
    ):
        llm_catalog.reset_event_llm_catalog()
        catalog = llm_catalog.event_llm_catalog()

    assert catalog["source"] == "azure"
    assert catalog["default"] == "gpt-4o"
    assert catalog["models"] == [
        {
            "deployment": "gpt-4o",
            "model": "gpt-4o",
            "label": "gpt-4o",
            "default": True,
        },
        {
            "deployment": "event-gpt41",
            "model": "gpt-4.1",
            "label": "event-gpt41 (gpt-4.1)",
            "default": False,
        },
    ]


def test_catalog_rejects_unavailable_explicit_deployment(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_catalog,
        "event_llm_catalog",
        lambda: {
            "default": "gpt-4o",
            "models": [{"deployment": "gpt-4o"}],
        },
    )
    with settings.override(
        azure_openai_deployment="gpt-4o",
        event_llm_deployment="",
    ):
        assert llm_catalog.resolve_event_llm_deployment(None) == "gpt-4o"
        with pytest.raises(ValueError, match="未知或不可用"):
            llm_catalog.resolve_event_llm_deployment("not-deployed")


def test_catalog_does_not_reinsert_stale_configured_default(
    monkeypatch,
) -> None:
    payload = {
        "data": [
            {
                "id": "stale-default",
                "model": "text-embedding-3-large",
                "status": "succeeded",
            },
            {
                "id": "event-gpt41",
                "model": "gpt-4.1",
                "status": "succeeded",
            },
            {
                "id": "embedding",
                "model": "text-embedding-3-large",
                "status": "succeeded",
            },
        ]
    }
    monkeypatch.setattr(
        llm_catalog.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(payload).encode()),
    )

    with settings.override(
        azure_openai_endpoint="https://unit.openai.azure.com",
        azure_openai_api_key="not-a-real-key",
        azure_openai_deployment="stale-default",
        event_llm_deployment="",
    ):
        llm_catalog.reset_event_llm_catalog()
        catalog = llm_catalog.event_llm_catalog()
        resolved = llm_catalog.resolve_event_llm_deployment(None)

    assert catalog["default"] == "event-gpt41"
    assert [item["deployment"] for item in catalog["models"]] == [
        "event-gpt41"
    ]
    assert resolved == "event-gpt41"
    assert "stale-default" not in json.dumps(catalog)
