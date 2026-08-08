from __future__ import annotations

import json
from types import SimpleNamespace

from app.core import config as config_mod
from app.core.config import settings
from app.services import event_chat
from app.services import llm_models
from app.services.llm_models import (
    ModelSelection,
    chat_completion_options,
    model_catalog,
    resolve_model,
)


def _catalog(targets: list[dict]) -> dict:
    for target in targets:
        target.setdefault("capabilities", {}).setdefault("json_output", True)
    return {
        "source": "arm",
        "warning": None,
        "catalog_models": [],
        "callable_targets": targets,
    }


def test_auto_routing_uses_discovered_targets_and_reports_actual_reason(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        llm_models,
        "event_llm_catalog",
        lambda: _catalog(
            [
                {
                    "deployment": "analysis-gpt41",
                    "model": "gpt-4.1",
                    "label": "analysis-gpt41 (gpt-4.1)",
                    "capabilities": {"chat": True, "image_input": True},
                },
                {
                    "deployment": "analysis-gpt5",
                    "model": "gpt-5",
                    "label": "analysis-gpt5 (gpt-5)",
                    "capabilities": {"chat": True, "image_input": True},
                },
                {
                    "deployment": "chat-mini",
                    "model": "gpt-4.1-mini",
                    "label": "chat-mini (gpt-4.1-mini)",
                    "capabilities": {"chat": True, "image_input": True},
                },
            ]
        ),
    )
    with settings.override(
        foundry_chat_deployment="chat-mini",
    ):
        analysis = resolve_model("analysis", "auto", locale="en")
        simple = resolve_model("chat", "auto", prompt="谁离开了？", locale="en")
        complex_question = resolve_model(
            "chat",
            "auto",
            prompt="请详细解释异常事件的证据依据，并比较两个事件窗。",
            locale="en",
        )

    assert analysis.selected == "analysis-gpt5"
    assert analysis.deployment == "analysis-gpt5"
    assert "highest-ranked image-capable" in analysis.reason
    assert simple.selected == "chat-mini"
    assert simple.deployment == "chat-mini"
    assert "configured chat deployment" in simple.reason
    assert complex_question.selected == "analysis-gpt5"
    assert "evidence reasoning" in complex_question.reason


def test_legacy_environment_lookup_and_unconfigured_dry_run(monkeypatch) -> None:
    monkeypatch.delenv("FOUNDRY_ANALYSIS_DEPLOYMENT", raising=False)
    monkeypatch.delenv("FOUNDRY_CHAT_DEPLOYMENT", raising=False)
    monkeypatch.delenv("EVENT_LLM_DEPLOYMENT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "legacy-deployment")
    assert config_mod._first_env(
        "FOUNDRY_ANALYSIS_DEPLOYMENT",
        "EVENT_LLM_DEPLOYMENT",
        "AZURE_OPENAI_DEPLOYMENT",
    ) == "legacy-deployment"
    assert config_mod._first_env(
        "FOUNDRY_CHAT_DEPLOYMENT",
        "FOUNDRY_ANALYSIS_DEPLOYMENT",
        "AZURE_OPENAI_DEPLOYMENT",
    ) == "legacy-deployment"

    monkeypatch.setattr(
        llm_models,
        "event_llm_catalog",
        lambda: _catalog([]),
    )
    dry_run = resolve_model(
        "analysis",
        "auto",
        require_deployment=False,
    )
    assert dry_run.deployment == ""


def test_shared_client_validation_allows_chat_only_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "azure_openai_endpoint",
        "https://unit.openai.azure.com/",
    )
    monkeypatch.setattr(settings, "azure_openai_auth", "managed_identity")
    monkeypatch.setattr(settings, "azure_openai_api_key", None)
    monkeypatch.setattr(settings, "foundry_analysis_deployment", None)
    monkeypatch.setattr(settings, "azure_openai_deployment", None)
    monkeypatch.setattr(settings, "foundry_chat_deployment", "chat-only")
    monkeypatch.setattr(
        llm_models,
        "event_llm_catalog",
        lambda: _catalog(
            [
                {
                    "deployment": "chat-only",
                    "model": "gpt-4.1-mini",
                    "label": "chat-only (gpt-4.1-mini)",
                    "capabilities": {"chat": True, "image_input": True},
                }
            ]
        ),
    )

    settings.require_openai()
    selection = resolve_model("chat", "chat-only")
    assert selection.deployment == "chat-only"


def test_model_labels_and_auto_reasons_follow_page_language(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_models,
        "event_llm_catalog",
        lambda: _catalog(
            [
                {
                    "deployment": "vision",
                    "model": "gpt-4.1",
                    "label": "vision (gpt-4.1)",
                    "capabilities": {"chat": True, "image_input": True},
                }
            ]
        ),
    )

    english = model_catalog(locale="en")
    chinese = model_catalog(locale="zh-CN")
    english_selection = resolve_model("analysis", "auto", locale="en")
    chinese_selection = resolve_model("analysis", "auto", locale="zh-CN")

    assert english["analysis"][0]["label"] == "Auto (best compatible deployment)"
    assert "质量" not in english["analysis"][0]["label"]
    assert "highest-ranked" in english_selection.reason
    assert chinese["analysis"][0]["label"] == "自动（最佳兼容部署）"
    assert "自动选择" in chinese_selection.reason


def test_gpt5_uses_supported_completion_parameter() -> None:
    assert chat_completion_options(
        "gpt-5.4", max_tokens=400, temperature=0.2
    ) == {"max_completion_tokens": 400}
    assert chat_completion_options(
        "gpt-4.1", max_tokens=400, temperature=0.2
    ) == {"max_tokens": 400, "temperature": 0.2}


def test_chat_uses_model_identifier_not_opaque_gpt5_deployment(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(event_chat, "RUNS_DIR", tmp_path)
    event_chat.persist_run_snapshot(
        {"run_id": "abcdef123456", "video": "demo.mp4", "windows": []}
    )
    requests: list[dict] = []
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer":"ok","evidence":[],"limitations":""}'
                )
            )
        ],
        usage=None,
    )
    monkeypatch.setattr(
        event_chat,
        "get_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: requests.append(kwargs) or response
                )
            )
        ),
    )
    selection = ModelSelection(
        task="chat",
        requested="event-quality-gpt54",
        selected="event-quality-gpt54",
        model="gpt-5.4",
        deployment="event-quality-gpt54",
        reason="unit test",
    )

    event_chat.chat_about_run("abcdef123456", "What happened?", selection)

    assert requests[0]["model"] == "event-quality-gpt54"
    assert requests[0]["max_completion_tokens"] > 0
    assert "max_tokens" not in requests[0]
    assert "temperature" not in requests[0]


def test_partner_capabilities_only_offer_phi_for_json_chat(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_models,
        "event_llm_catalog",
        lambda: _catalog(
            [
                {
                    "deployment": "Phi-4-reasoning",
                    "model": "Phi-4-reasoning",
                    "label": "Phi-4 reasoning",
                    "provider": "microsoft",
                    "capabilities": {
                        "chat": True,
                        "image_input": False,
                        "json_output": True,
                    },
                },
                {
                    "deployment": "DeepSeek-V4-Flash",
                    "model": "DeepSeek-V4-Flash",
                    "label": "DeepSeek V4 Flash",
                    "provider": "deepseek",
                    "capabilities": {
                        "chat": True,
                        "image_input": False,
                        "json_output": True,
                    },
                },
            ]
        ),
    )

    catalog = model_catalog(locale="en")
    assert [item["alias"] for item in catalog["analysis"]] == ["auto"]
    assert {item["alias"] for item in catalog["chat"]} == {
        "auto",
        "Phi-4-reasoning",
        "DeepSeek-V4-Flash",
    }


def test_run_snapshot_omits_images_and_chat_persists_bounded_history(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(event_chat, "RUNS_DIR", tmp_path)
    payload = {
        "run_id": "abcdef123456",
        "video": "C:\\videos\\demo.mp4",
        "fps": 2,
        "frames_total": 10,
        "windows": [
            {
                "window_index": 1,
                "time_range": ["00:00", "00:05"],
                "event": {"summary": "主体#1 离开。"},
                "keyframes": [
                    {
                        "timestamp": "00:03",
                        "image": "data:image/jpeg;base64,secret-image",
                    }
                ],
            }
        ],
    }
    event_chat.persist_run_snapshot(payload)
    saved = json.loads(
        (tmp_path / "abcdef123456" / "result.json").read_text(encoding="utf-8")
    )
    assert saved["windows"][0]["keyframe_timestamps"] == ["00:03"]
    assert "secret-image" not in json.dumps(saved)

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "answer": "主体#1 在窗1的 00:03 左右离开。",
                            "evidence": [
                                {
                                    "window_index": 1,
                                    "time_range": ["00:00", "00:05"],
                                    "reason": "事件摘要和关键帧时间一致",
                                }
                            ],
                            "limitations": "",
                        },
                        ensure_ascii=False,
                    )
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        ),
    )
    fake_completions = SimpleNamespace(create=lambda **kwargs: response)
    monkeypatch.setattr(
        event_chat,
        "get_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        ),
    )
    selection = ModelSelection(
        task="chat",
        requested="auto",
        selected="gpt-4.1-mini",
        model="gpt-4.1-mini",
        deployment="chat-unit",
        reason="latency-sensitive follow-up",
    )

    result = event_chat.chat_about_run("abcdef123456", "谁离开了？", selection)

    assert result["answer"].startswith("主体#1")
    assert result["usage"]["total_tokens"] == 120
    assert result["selection"]["model"] == "gpt-4.1-mini"
    history = json.loads(
        (tmp_path / "abcdef123456" / "chat.json").read_text(encoding="utf-8")
    )
    assert [item["role"] for item in history] == ["user", "assistant"]


def test_chat_history_file_is_bounded(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(event_chat, "RUNS_DIR", tmp_path)
    event_chat.persist_run_snapshot(
        {"run_id": "123456abcdef", "video": "demo.mp4", "windows": []}
    )
    run_dir = tmp_path / "123456abcdef"
    (run_dir / "chat.json").write_text(
        json.dumps(
            [
                {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
                for index in range(30)
            ]
        ),
        encoding="utf-8",
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer":"ok","evidence":[],"limitations":""}'
                )
            )
        ],
        usage=None,
    )
    monkeypatch.setattr(
        event_chat,
        "get_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response)
            )
        ),
    )
    selection = ModelSelection(
        task="chat",
        requested="auto",
        selected="gpt-4.1-mini",
        model="gpt-4.1-mini",
        deployment="chat-unit",
        reason="latency-sensitive follow-up",
    )

    with settings.override(event_chat_history_turns=3):
        event_chat.chat_about_run("123456abcdef", "next", selection)

    history = json.loads((run_dir / "chat.json").read_text(encoding="utf-8"))
    assert len(history) == 6
    assert history[-2:] == [
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "ok"},
    ]


def test_chat_uses_saved_report_language(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(event_chat, "RUNS_DIR", tmp_path)
    event_chat.persist_run_snapshot(
        {
            "run_id": "abcdef123456",
            "video": "demo.mp4",
            "report_language": "en",
            "windows": [],
        }
    )
    requests: list[dict] = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
        usage=None,
    )
    monkeypatch.setattr(
        event_chat,
        "get_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: requests.append(kwargs) or response
                )
            )
        ),
    )
    selection = ModelSelection(
        task="chat",
        requested="auto",
        selected="gpt-4.1-mini",
        model="gpt-4.1-mini",
        deployment="chat-unit",
        reason="latency-sensitive follow-up",
    )

    result = event_chat.chat_about_run("abcdef123456", "What happened?", selection)

    assert result["answer"] == "The available evidence is insufficient to answer the question."
    assert any("in English" in message["content"] for message in requests[0]["messages"])
