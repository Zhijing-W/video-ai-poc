from __future__ import annotations

import json
from types import SimpleNamespace

from app.core import config as config_mod
from app.core.config import settings
from app.services import event_chat
from app.services.llm_models import ModelSelection, resolve_model


def test_auto_routing_prefers_quality_for_analysis_and_mini_for_simple_chat() -> None:
    with settings.override(
        foundry_analysis_deployment="analysis-unit",
        foundry_chat_deployment="chat-unit",
    ):
        analysis = resolve_model("analysis", "auto")
        simple = resolve_model("chat", "auto", prompt="谁离开了？")
        complex_question = resolve_model(
            "chat",
            "auto",
            prompt="请详细解释异常事件的证据依据，并比较两个事件窗。",
        )

    assert analysis.selected == "gpt-4.1"
    assert analysis.deployment == "analysis-unit"
    assert simple.selected == "gpt-4.1-mini"
    assert simple.deployment == "chat-unit"
    assert complex_question.selected == "gpt-4.1"


def test_legacy_deployment_fallback_and_unconfigured_dry_run(monkeypatch) -> None:
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
        settings,
        "foundry_analysis_deployment",
        "legacy-deployment",
    )
    monkeypatch.setattr(
        settings,
        "foundry_chat_deployment",
        "legacy-deployment",
    )
    analysis = resolve_model("analysis", "gpt-4.1")
    chat = resolve_model("chat", "gpt-4.1-mini")

    assert analysis.deployment == "legacy-deployment"
    assert chat.deployment == "legacy-deployment"

    monkeypatch.setattr(settings, "foundry_analysis_deployment", None)
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

    settings.require_openai()
    selection = resolve_model("chat", "gpt-4.1-mini")
    assert selection.deployment == "chat-only"


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
