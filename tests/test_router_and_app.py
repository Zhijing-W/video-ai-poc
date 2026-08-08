from __future__ import annotations

import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from app import face
from app.main import app
from app.routers import event_monitor
from app.services import event_chat, llm_models
from app.services.prompt_compaction import compact_evidence


def test_router_generates_unique_run_and_session_ids(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_analyze(video_path, run_dir, **kwargs):
        calls.append(
            {
                "video_path": str(video_path),
                "run_dir": str(run_dir),
                "session_id": kwargs["session_id"],
                "with_body": kwargs["with_body"],
            }
        )
        return {
            "video": str(video_path),
            "fps": kwargs["fps"],
            "frames_total": 0,
            "img_size": [0, 0],
            "session_id": kwargs["session_id"],
            "tracker_backend": "mock",
            "reid_backend": "mock",
            "reid_dim": 0,
            "with_face": kwargs["with_face"],
            "with_gait": kwargs["with_gait"],
            "with_ocr": kwargs["with_ocr"],
            "with_objects": kwargs["with_objects"],
            "gait_error": None,
            "ocr_backend": None,
            "ocr_error": None,
            "object_classes": None,
            "model": None,
            "dry_run": True,
            "elapsed_seconds": 0.0,
            "stage_timings": {"extract_frames": 0.0},
            "runtime": {"execution": "server", "detector_device": "cpu", "reid_device": "cuda:0"},
            "body_reid_timing": {
                "backend": "mock",
                "device": "cuda:0",
                "call_count": 0,
                "total_ms": 0.0,
                "mean_ms": None,
                "p95_ms": None,
            },
            "tracks": {},
            "windows": [],
            "overall": None,
        }

    monkeypatch.setattr(event_monitor, "analyze_event_stream", fake_analyze)

    client = TestClient(app)
    responses = [
        client.post(
            "/api/event-monitor/understand",
            files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
            data={"dry_run": "true"},
        )
        for _ in range(2)
    ]

    for response in responses:
        assert response.status_code == 200

    body1, body2 = [response.json() for response in responses]
    assert body1["run_id"] != body2["run_id"]
    assert body1["session_id"] == f"event-monitor-{body1['run_id']}"
    assert body2["session_id"] == f"event-monitor-{body2['run_id']}"
    assert calls[0]["run_dir"].endswith(body1["run_id"])
    assert calls[1]["run_dir"].endswith(body2["run_id"])
    assert calls[0]["run_dir"] != calls[1]["run_dir"]
    assert all(call["with_body"] is True for call in calls)
    assert all(
        response.json()["config_used"]["reid_backend"] == "differ"
        for response in responses
    )
    assert all(
        response.json()["config_used"]["reid_backend_effective"] == "mock"
        and response.json()["config_used"]["reid_device"] == "cuda:0"
        and response.json()["body_reid_timing"]["backend"] == "mock"
        for response in responses
    )

    for call in calls:
        shutil.rmtree(Path(call["run_dir"]), ignore_errors=True)


def test_router_returns_structured_fatal_reid_telemetry(monkeypatch) -> None:
    run_dirs: list[Path] = []
    timing = {
        "backend": "differ",
        "device": "cuda:0",
        "call_count": 1,
        "total_ms": 12.5,
        "mean_ms": 12.5,
        "p95_ms": 12.5,
        "failed_call_count": 1,
        "p95_definition": "nearest-rank",
        "timing_scope": "one crop",
        "by_purpose": {
            "tracking": {
                "call_count": 1,
                "total_ms": 12.5,
                "mean_ms": 12.5,
                "p95_ms": 12.5,
            }
        },
    }

    def fail_analysis(video_path, run_dir, **kwargs):
        run_dirs.append(Path(run_dir))
        raise event_monitor.EventAnalysisRunError(
            "embedding exploded",
            body_reid_timing=timing,
        )

    monkeypatch.setattr(event_monitor, "analyze_event_stream", fail_analysis)
    response = TestClient(app).post(
        "/api/event-monitor/understand",
        files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
        data={"dry_run": "true"},
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["message"] == "事件理解失败：embedding exploded"
    assert detail["body_reid_timing"] == timing
    assert detail["config_used"]["reid_backend"] == "differ"
    assert detail["config_used"]["reid_backend_effective"] == "differ"
    assert detail["config_used"]["reid_device"] == "cuda:0"
    assert len(run_dirs) == 1
    shutil.rmtree(run_dirs[0], ignore_errors=True)


def test_fastapi_health_page_and_openapi_are_available() -> None:
    client = TestClient(app)

    health = client.get("/health")
    page = client.get("/event-monitor")
    openapi = client.get("/openapi.json")

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "feature": "event-monitor"}
    assert page.status_code == 200
    assert "event-monitor" in page.text.lower()
    assert 'id="withBody" checked' in page.text
    assert "ArcFace (default)" in page.text
    assert "Off (default)" in page.text
    assert '<select id="reidBackend" class="em-input" disabled>' in page.text
    assert '<option value="">Use server default (loading…)</option>' in page.text
    assert '<option value="differ"' not in page.text
    assert 'id="reidDiagnostics"' in page.text
    reid_select = page.text.split('id="reidBackend"', 1)[1].split("</select>", 1)[0]
    assert '<option value="auto">' not in reid_select
    assert '<option value="resnet50">' not in page.text
    assert '<option value="coarse">' not in page.text
    assert 'id="analysisModel"' in page.text
    assert 'id="chatModel"' in page.text
    assert 'id="chatPanel"' in page.text
    assert 'id="promptFormat"' in page.text
    assert 'id="btnDownloadPrompt"' in page.text
    assert "Compact table (TSV/CSV-style)" in page.text
    assert openapi.status_code == 200
    assert "/api/event-monitor/understand" in openapi.text
    assert "/api/event-monitor/llm-models" in openapi.text
    assert "/api/event-monitor/runs/{run_id}/chat" in openapi.text
    assert "/api/event-monitor/runs/{run_id}/prompt" in openapi.text


def test_run_prompt_export_is_run_scoped_safe_and_uses_canonical_serializer(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(event_chat, "RUNS_DIR", tmp_path)
    run_id = "abcdef123456"
    payload = {
        "run_id": run_id,
        "video": "demo.mp4",
        "windows": [
            {
                "window_index": 1,
                "time_range": ["00:00", "00:05"],
                "ocr_evidence": [
                    {
                        "frame_index": 1,
                        "timestamp": "00:01",
                        "texts": [
                            {
                                "text": "comma, tab\t newline\nUnicode 中文",
                                "conf": 0.9,
                                "box": [1, 2, 3, 4],
                            }
                        ],
                    }
                ],
                "keyframes": [
                    {
                        "timestamp": "00:01",
                        "image": "data:image/jpeg;base64,DO_NOT_EXPORT_IMAGE",
                    }
                ],
            }
        ],
    }
    event_chat.persist_run_snapshot(payload)
    result_path = tmp_path / run_id / "result.json"
    snapshot = json.loads(result_path.read_text(encoding="utf-8"))
    snapshot["api_key"] = "do-not-export"
    snapshot["windows"][0]["unexpected_image"] = (
        "data:image/png;base64,ALSO_DO_NOT_EXPORT"
    )
    result_path.write_text(json.dumps(snapshot), encoding="utf-8")

    client = TestClient(app)
    tsv = client.get(f"/api/event-monitor/runs/{run_id}/prompt?format=tsv")
    exported_json = client.get(f"/api/event-monitor/runs/{run_id}/prompt?format=json")

    expected = compact_evidence(
        snapshot["windows"],
        overall=snapshot.get("overall"),
        run_metadata=snapshot,
    )
    assert tsv.status_code == 200
    assert tsv.text == expected + "\n"
    assert tsv.headers["content-type"].startswith("text/tab-separated-values")
    assert 'filename="event-monitor_abcdef123456_prompt.tsv"' in tsv.headers[
        "content-disposition"
    ]
    assert tsv.headers["cache-control"] == "no-store"
    assert tsv.headers["x-content-type-options"] == "nosniff"
    assert "comma, tab\\t newline\\nUnicode \\u4e2d\\u6587" in tsv.text
    assert "DO_NOT_EXPORT_IMAGE" not in tsv.text
    assert "ALSO_DO_NOT_EXPORT" not in tsv.text

    assert exported_json.status_code == 200
    assert exported_json.headers["content-type"].startswith("application/json")
    json_payload = exported_json.json()
    assert json_payload["api_key"] == "[redacted]"
    assert json_payload["windows"][0]["unexpected_image"] == "[image data omitted]"
    assert "DO_NOT_EXPORT_IMAGE" not in exported_json.text
    assert "ALSO_DO_NOT_EXPORT" not in exported_json.text

    assert client.get("/api/event-monitor/runs/not-a-run/prompt").status_code == 400
    assert (
        client.get(f"/api/event-monitor/runs/{run_id}/prompt?format=csv").status_code
        == 400
    )
    assert client.get("/api/event-monitor/runs/123456abcdef/prompt").status_code == 404


def test_superres_backend_catalog_and_unknown_request_validation() -> None:
    client = TestClient(app)

    catalog = client.get("/api/event-monitor/superres-backends")
    invalid = client.post(
        "/api/event-monitor/understand",
        files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
        data={"dry_run": "true", "face_superres": "not-registered"},
    )

    assert catalog.status_code == 200
    catalog_body = catalog.json()
    assert catalog_body["default"] == "off"
    assert {
        "off",
        "gfpgan",
        "codeformer",
        "realesrgan_x2plus",
    } <= set(catalog_body["backends"])
    assert catalog_body["metadata"]["codeformer"] == {
        "requires_fidelity": True,
        "fidelity_default": 1.0,
        "fidelity_min": 0.0,
        "fidelity_max": 1.0,
    }
    assert invalid.status_code == 400
    assert "未知人脸超分后端" in invalid.json()["detail"]


def test_router_propagates_explicit_body_identity_disable(monkeypatch) -> None:
    calls = []

    def fake_analyze(video_path, run_dir, **kwargs):
        calls.append(kwargs)
        return {
            "video": str(video_path),
            "fps": kwargs["fps"],
            "frames_total": 0,
            "img_size": [0, 0],
            "session_id": kwargs["session_id"],
            "tracker_backend": "mock",
            "reid_backend": None,
            "reid_dim": None,
            "with_body": kwargs["with_body"],
            "with_face": kwargs["with_face"],
            "with_gait": kwargs["with_gait"],
            "with_ocr": kwargs["with_ocr"],
            "with_objects": kwargs["with_objects"],
            "dry_run": True,
            "elapsed_seconds": 0.0,
            "stage_timings": {},
            "runtime": {"execution": "server", "detector_device": "cpu"},
            "body_reid_timing": {
                "backend": "tracking-unit",
                "device": "cpu",
                "call_count": 2,
                "total_ms": 4.0,
                "mean_ms": 2.0,
                "p95_ms": 2.5,
                "failed_call_count": 0,
                "by_purpose": {
                    "tracking": {
                        "call_count": 2,
                        "total_ms": 4.0,
                        "mean_ms": 2.0,
                        "p95_ms": 2.5,
                    }
                },
            },
            "tracks": {},
            "windows": [],
            "overall": None,
        }

    monkeypatch.setattr(event_monitor, "analyze_event_stream", fake_analyze)
    response = TestClient(app).post(
        "/api/event-monitor/understand",
        files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
        data={"dry_run": "true", "with_body": "false"},
    )

    assert response.status_code == 200
    assert calls[0]["with_body"] is False
    assert response.json()["config_used"]["with_body"] is False
    assert response.json()["config_used"]["reid_backend_effective"] == "tracking-unit"
    assert response.json()["config_used"]["reid_device"] == "cpu"
    shutil.rmtree(
        event_monitor.OUT_DIR / response.json()["run_id"],
        ignore_errors=True,
    )


def test_reid_backend_catalog_and_unknown_request_validation() -> None:
    client = TestClient(app)

    catalog = client.get("/api/event-monitor/reid-backends")
    invalid = client.post(
        "/api/event-monitor/understand",
        files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
        data={"dry_run": "true", "reid_backend": "not-registered"},
    )

    assert catalog.status_code == 200
    body = catalog.json()
    assert body["default"] == "differ"
    assert {
        "auto",
        "osnet",
        "resnet50",
        "coarse",
        "clipreid",
        "siglip2",
        "differ",
    } <= set(body["backends"])
    assert body["metadata"]["clipreid"]["requires_cuda"] is True
    assert body["metadata"]["siglip2"]["experimental"] is True
    assert invalid.status_code == 400
    assert "未知人形ReID后端" in invalid.json()["detail"]


def test_explicit_reid_backend_request_overrides_default(monkeypatch) -> None:
    monkeypatch.setattr(
        event_monitor,
        "analyze_event_stream",
        lambda *args, **kwargs: {},
    )

    response = TestClient(app).post(
        "/api/event-monitor/understand",
        files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
        data={"dry_run": "true", "reid_backend": "clipreid"},
    )

    assert response.status_code == 200
    assert response.json()["config_used"]["reid_backend"] == "clipreid"
    shutil.rmtree(
        event_monitor.OUT_DIR / response.json()["run_id"],
        ignore_errors=True,
    )


def test_omitted_reid_backend_preserves_configured_server_default(monkeypatch) -> None:
    monkeypatch.setattr(
        event_monitor,
        "analyze_event_stream",
        lambda *args, **kwargs: {},
    )

    with event_monitor.settings.override(reid_backend="clipreid"):
        response = TestClient(app).post(
            "/api/event-monitor/understand",
            files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
            data={"dry_run": "true"},
        )

    assert response.status_code == 200
    assert response.json()["config_used"]["reid_backend"] == "clipreid"
    shutil.rmtree(
        event_monitor.OUT_DIR / response.json()["run_id"],
        ignore_errors=True,
    )


def test_router_rejects_unknown_effective_default_when_face_is_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(event_monitor.settings, "face_superres", "missing-default")

    response = TestClient(app).post(
        "/api/event-monitor/understand",
        files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
        data={"dry_run": "true", "with_face": "true"},
    )

    assert response.status_code == 400
    assert "未知人脸超分后端" in response.json()["detail"]


def test_router_accepts_registered_superres_backend(monkeypatch) -> None:
    calls = []

    face.register_superres_backend(
        "unit-router",
        lambda: object(),
        lambda model, image, aligned: image.copy(),
        replace=True,
    )

    def fake_analyze(video_path, run_dir, **kwargs):
        calls.append(kwargs)
        return {
            "video": str(video_path),
            "fps": kwargs["fps"],
            "frames_total": 0,
            "img_size": [0, 0],
            "session_id": kwargs["session_id"],
            "tracker_backend": "mock",
            "reid_backend": "mock",
            "reid_dim": 0,
            "with_face": kwargs["with_face"],
            "with_gait": kwargs["with_gait"],
            "with_ocr": kwargs["with_ocr"],
            "with_objects": kwargs["with_objects"],
            "dry_run": True,
            "stage_timings": {},
            "tracks": {},
            "windows": [],
            "overall": None,
        }

    monkeypatch.setattr(event_monitor, "analyze_event_stream", fake_analyze)
    response = TestClient(app).post(
        "/api/event-monitor/understand",
        files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
        data={
            "dry_run": "true",
            "face_superres": "unit-router",
            "face_codeformer_fidelity": "0.85",
        },
    )

    assert response.status_code == 200
    assert response.json()["config_used"]["face_superres"] == "unit_router"
    assert response.json()["config_used"]["face_codeformer_fidelity"] == 0.85
    assert calls
    shutil.rmtree(
        event_monitor.OUT_DIR / response.json()["run_id"],
        ignore_errors=True,
    )


def test_waiting_request_does_not_inherit_inflight_superres_override(
    monkeypatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    observed = []

    def fake_analyze(video_path, run_dir, **kwargs):
        observed.append(event_monitor.settings.face_superres)
        if len(observed) == 1:
            entered.set()
            assert release.wait(timeout=5)
        return {
            "video": str(video_path),
            "fps": kwargs["fps"],
            "frames_total": 0,
            "img_size": [0, 0],
            "session_id": kwargs["session_id"],
            "tracker_backend": "mock",
            "reid_backend": "mock",
            "reid_dim": 0,
            "with_face": kwargs["with_face"],
            "with_gait": kwargs["with_gait"],
            "with_ocr": kwargs["with_ocr"],
            "with_objects": kwargs["with_objects"],
            "dry_run": True,
            "stage_timings": {},
            "tracks": {},
            "windows": [],
            "overall": None,
        }

    monkeypatch.setattr(event_monitor, "analyze_event_stream", fake_analyze)

    with TestClient(app) as client:
        def request(data):
            return client.post(
                "/api/event-monitor/understand",
                files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
                data=data,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                request,
                {
                    "dry_run": "true",
                    "with_face": "true",
                    "face_superres": "codeformer",
                },
            )
            assert entered.wait(timeout=5)
            second = executor.submit(
                request,
                {
                    "dry_run": "true",
                    "with_face": "true",
                },
            )
            time.sleep(0.05)
            release.set()
            responses = [first.result(timeout=5), second.result(timeout=5)]

    assert [response.status_code for response in responses] == [200, 200]
    assert observed == ["codeformer", event_monitor._STARTUP_FACE_SUPERRES]
    for response in responses:
        shutil.rmtree(
            event_monitor.OUT_DIR / response.json()["run_id"],
            ignore_errors=True,
        )


def test_router_rejects_invalid_codeformer_fidelity_before_processing(
    monkeypatch,
) -> None:
    called = False

    def fake_analyze(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(event_monitor, "analyze_event_stream", fake_analyze)
    response = TestClient(app).post(
        "/api/event-monitor/understand",
        files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
        data={
            "dry_run": "true",
            "face_superres": "codeformer",
            "face_codeformer_fidelity": "1.01",
        },
    )

    assert response.status_code == 400
    assert "[0, 1]" in response.json()["detail"]
    assert called is False


def test_llm_model_catalog_and_unknown_analysis_model_validation(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_models,
        "event_llm_catalog",
        lambda: {
            "source": "arm",
            "warning": None,
            "catalog_models": [{"model": "gpt-4.1"}],
            "callable_targets": [
                {
                    "deployment": "analysis-unit",
                    "model": "gpt-4.1",
                    "label": "analysis-unit (gpt-4.1)",
                    "capabilities": {
                        "chat": True,
                        "image_input": True,
                        "json_output": True,
                    },
                },
                {
                    "deployment": "chat-unit",
                    "model": "gpt-4.1-mini",
                    "label": "chat-unit (gpt-4.1-mini)",
                    "capabilities": {
                        "chat": True,
                        "image_input": True,
                        "json_output": True,
                    },
                },
                {
                    "deployment": "embedding",
                    "model": "text-embedding-3-large",
                    "label": "embedding (text-embedding-3-large)",
                    "capabilities": {
                        "chat": False,
                        "image_input": False,
                        "json_output": False,
                    },
                },
            ],
        },
    )
    client = TestClient(app)
    with event_monitor.settings.override(
        foundry_analysis_deployment="analysis-unit",
        foundry_chat_deployment="chat-unit",
    ):
        catalog = client.get("/api/event-monitor/llm-models")

    assert catalog.status_code == 200
    body = catalog.json()
    assert body["defaults"] == {"analysis": "auto", "chat": "auto"}
    assert body["auth"] in {"managed_identity", "api_key"}
    assert {"auto", "analysis-unit", "chat-unit"} <= {
        item["alias"] for item in body["analysis"]
    }
    assert "embedding" not in {item["alias"] for item in body["analysis"]}
    assert body["catalog_models"] == [{"model": "gpt-4.1"}]

    called = False

    def fake_analyze(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(event_monitor, "analyze_event_stream", fake_analyze)
    invalid = client.post(
        "/api/event-monitor/understand",
        files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
        data={"dry_run": "true", "analysis_model": "arbitrary-deployment"},
    )

    assert invalid.status_code == 400
    assert "未知或不可用的可调用部署" in invalid.json()["detail"]
    assert called is False


def test_run_chat_uses_validated_backend_model_alias(monkeypatch) -> None:
    observed = {}

    def fake_chat(run_id, question, selection):
        observed.update(
            run_id=run_id,
            question=question,
            selection=selection.public_dict(),
        )
        return {
            "answer": "主体#1 在窗1离开。",
            "evidence": [{"window_index": 1}],
            "selection": selection.public_dict(),
        }

    monkeypatch.setattr(event_monitor, "chat_about_run", fake_chat)
    monkeypatch.setattr(
        llm_models,
        "event_llm_catalog",
        lambda: {
            "source": "arm",
            "warning": None,
            "catalog_models": [],
            "callable_targets": [
                {
                    "deployment": "chat-unit",
                    "model": "gpt-4.1-mini",
                    "label": "chat-unit (gpt-4.1-mini)",
                    "capabilities": {
                        "chat": True,
                        "image_input": True,
                        "json_output": True,
                    },
                }
            ],
        },
    )
    with event_monitor.settings.override(
        foundry_analysis_deployment="analysis-unit",
        foundry_chat_deployment="chat-unit",
    ):
        response = TestClient(app).post(
            "/api/event-monitor/runs/abcdef123456/chat",
            json={"question": "谁离开了？", "model": "chat-unit", "language": "en"},
        )

    assert response.status_code == 200
    assert observed == {
        "run_id": "abcdef123456",
        "question": "谁离开了？",
        "selection": {
            "task": "chat",
            "requested": "chat-unit",
            "selected": "chat-unit",
            "model": "gpt-4.1-mini",
            "reason": "Explicit callable deployment selection.",
        },
    }
