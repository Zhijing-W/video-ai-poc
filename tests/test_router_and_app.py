from __future__ import annotations

import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app import face
from app.event_monitor_i18n import MESSAGES
from app.routers import event_monitor


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
    zh_page = client.get("/event-monitor/zh")
    openapi = client.get("/openapi.json")

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "feature": "event-monitor"}
    assert page.status_code == 200
    assert "<html lang=\"en\">" in page.text
    assert "<title>Video Surveillance PoC</title>" in page.text
    assert "Video Surveillance PoC" in page.text
    assert "Combining computer vision for cross-frame person identification with multimodal LLM reasoning for event report generation." in page.text
    assert 'id="withBody" checked' in page.text
    assert "ArcFace (default)" in page.text
    assert '<option value="arcface">' not in page.text
    assert "BoT-SORT (default · motion + appearance association)" in page.text
    assert '<option value="botsort_reid">' not in page.text
    assert "Off (default)" in page.text
    assert '<option value="off">' not in page.text
    assert 'id="aiModel"' in page.text
    assert "AI analysis" in page.text
    assert '<select id="reidBackend" class="em-input" disabled>' in page.text
    assert '<option value="">Use server default (loading…)</option>' in page.text
    assert '<option value="differ"' not in page.text
    assert 'id="reidDiagnostics"' in page.text
    assert '<option value="auto">' not in page.text
    assert '<option value="resnet50">' not in page.text
    assert '<option value="coarse">' not in page.text
    assert 'href="/event-monitor/zh"' in page.text
    assert '"reportLanguage": "en"' in page.text
    assert zh_page.status_code == 200
    assert "<html lang=\"zh-CN\">" in zh_page.text
    assert "<title>视频监控 PoC</title>" in zh_page.text
    assert "视频监控 PoC" in zh_page.text
    assert "结合计算机视觉进行跨帧人员识别，并通过多模态大语言模型推理生成事件报告。" in zh_page.text
    assert "ArcFace（默认）" in zh_page.text
    assert "关闭（默认）" in zh_page.text
    assert "AI 分析" in zh_page.text
    assert "BoT-SORT（默认 · 运动+外观关联）" in zh_page.text
    assert 'href="/event-monitor"' in zh_page.text
    assert '"reportLanguage": "zh-CN"' in zh_page.text
    assert openapi.status_code == 200
    assert "/api/event-monitor/understand" in openapi.text


def test_root_and_legacy_event_monitor_redirect_to_english_page() -> None:
    client = TestClient(app)

    root = client.get("/", follow_redirects=False)
    legacy = client.get("/eventmonitor", follow_redirects=False)

    assert root.status_code in {302, 307}
    assert root.headers["location"] == "/event-monitor"
    assert legacy.status_code in {302, 307}
    assert legacy.headers["location"] == "/event-monitor"


def test_english_ui_bundle_has_no_chinese_runtime_copy() -> None:
    def flatten(value: dict, prefix: str = "") -> dict[str, str]:
        result: dict[str, str] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                result.update(flatten(item, path))
            else:
                result[path] = str(item)
        return result

    english = flatten(MESSAGES["en"])
    chinese = flatten(MESSAGES["zh-CN"])

    assert set(english) == set(chinese)
    assert [
        (key, value)
        for key, value in english.items()
        if key != "nav.chinese_short" and re.search(r"[\u4e00-\u9fff]", value)
    ] == []


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


def test_router_propagates_report_language_to_analysis(monkeypatch) -> None:
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
            "reid_backend": "mock",
            "reid_dim": 0,
            "with_face": kwargs["with_face"],
            "with_gait": kwargs["with_gait"],
            "with_ocr": kwargs["with_ocr"],
            "with_objects": kwargs["with_objects"],
            "dry_run": True,
            "elapsed_seconds": 0.0,
            "stage_timings": {},
            "tracks": {},
            "windows": [],
            "overall": None,
            "report_language": kwargs["report_language"] or "zh-CN",
        }

    monkeypatch.setattr(event_monitor, "analyze_event_stream", fake_analyze)
    monkeypatch.setattr(
        event_monitor.llm_catalog_mod,
        "resolve_event_llm_deployment",
        lambda value: value or "gpt-4o",
    )
    response = TestClient(app).post(
        "/api/event-monitor/understand",
        files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
        data={
            "dry_run": "true",
            "language": "en",
            "llm_model": "event-gpt41",
        },
    )

    assert response.status_code == 200
    assert calls[0]["report_language"] == "en"
    assert calls[0]["llm_model"] == "event-gpt41"
    assert response.json()["report_language"] == "en"
    shutil.rmtree(
        event_monitor.OUT_DIR / response.json()["run_id"],
        ignore_errors=True,
    )


def test_complete_route_propagates_language_to_llm_helpers(monkeypatch) -> None:
    calls = {"understand": [], "summary": []}

    def fake_understand(frames, identity, **kwargs):
        calls["understand"].append(kwargs)
        return {
            "events": [],
            "summary": "stub",
            "subjects_involved": [],
            "alert_level": "normal",
            "notification": "",
        }

    def fake_summary(windows, **kwargs):
        calls["summary"].append(kwargs)
        return {
            "overall_summary": "stub overall",
            "story": [],
            "subjects": [],
            "overall_alert_level": "normal",
            "notification": "",
        }

    monkeypatch.setattr(event_monitor, "understand_event", fake_understand)
    monkeypatch.setattr(event_monitor, "summarize_event_windows", fake_summary)
    monkeypatch.setattr(
        event_monitor.llm_catalog_mod,
        "resolve_event_llm_deployment",
        lambda value: value or "gpt-4o",
    )
    response = TestClient(app).post(
        "/api/event-monitor/complete",
        json={
            "payload": {
                "windows": [
                    {
                        "keyframes": [{"image": "data:image/png;base64,abc", "timestamp": "00:00:01"}],
                        "identity_context": "identity",
                        "scene_context": "scene",
                        "object_context": "objects",
                    }
                ],
            },
            "language": "en",
            "llm_model": "event-gpt41",
        },
    )

    assert response.status_code == 200
    assert calls["understand"][0]["language"] == "en"
    assert calls["understand"][0]["scene_context"] == "scene"
    assert calls["understand"][0]["object_context"] == "objects"
    assert calls["understand"][0]["model"] == "event-gpt41"
    assert calls["summary"][0]["language"] == "en"
    assert calls["summary"][0]["model"] == "event-gpt41"
    assert response.json()["report_language"] == "en"
    assert response.json()["model"] == "event-gpt41"


def test_complete_route_rejects_non_inline_keyframe_images(
    monkeypatch,
) -> None:
    called = False

    def fake_understand(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(event_monitor, "understand_event", fake_understand)
    response = TestClient(app).post(
        "/api/event-monitor/complete",
        json={
            "payload": {
                "windows": [
                    {
                        "keyframes": [
                            {
                                "image": "README.md",
                                "timestamp": "00:00:01",
                            }
                        ]
                    }
                ]
            },
            "llm_model": None,
        },
    )

    assert response.status_code == 400
    assert "image data URI" in response.json()["detail"]
    assert called is False


def test_complete_route_can_switch_back_to_current_default_model(
    monkeypatch,
) -> None:
    selected_values = []
    model_calls = []

    def resolve(value):
        selected_values.append(value)
        return "gpt-4o" if not value else value

    monkeypatch.setattr(
        event_monitor.llm_catalog_mod,
        "resolve_event_llm_deployment",
        resolve,
    )
    monkeypatch.setattr(
        event_monitor,
        "understand_event",
        lambda *args, **kwargs: (
            model_calls.append(kwargs["model"])
            or {
                "events": [],
                "summary": "stub",
                "subjects_involved": [],
                "alert_level": "normal",
                "notification": "",
            }
        ),
    )
    monkeypatch.setattr(
        event_monitor,
        "summarize_event_windows",
        lambda *args, **kwargs: {},
    )
    response = TestClient(app).post(
        "/api/event-monitor/complete",
        json={
            "payload": {
                "model": "event-gpt41",
                "windows": [
                    {
                        "keyframes": [
                            {
                                "image": "data:image/png;base64,abc",
                                "timestamp": "00:00:01",
                            }
                        ]
                    }
                ],
            },
            "llm_model": None,
        },
    )

    assert response.status_code == 200
    assert selected_values == [None]
    assert model_calls == ["gpt-4o"]
    assert response.json()["model"] == "gpt-4o"


def test_llm_model_catalog_endpoint_and_unknown_model_validation(
    monkeypatch,
) -> None:
    catalog = {
        "default": "gpt-4o",
        "models": [
            {
                "deployment": "gpt-4o",
                "model": "gpt-4o",
                "label": "gpt-4o",
                "default": True,
            }
        ],
        "source": "azure",
        "warning": None,
    }
    monkeypatch.setattr(
        event_monitor.llm_catalog_mod,
        "event_llm_catalog",
        lambda: catalog,
    )

    def resolve(value):
        if value and value != "gpt-4o":
            raise ValueError("未知或不可用的事件分析 AI deployment")
        return "gpt-4o"

    monkeypatch.setattr(
        event_monitor.llm_catalog_mod,
        "resolve_event_llm_deployment",
        resolve,
    )
    client = TestClient(app)

    response = client.get("/api/event-monitor/llm-models")
    invalid = client.post(
        "/api/event-monitor/understand",
        files={"file": ("clip.mp4", b"fake video bytes", "video/mp4")},
        data={"dry_run": "true", "llm_model": "not-deployed"},
    )

    assert response.status_code == 200
    assert response.json() == catalog
    assert invalid.status_code == 400
    assert "未知或不可用" in invalid.json()["detail"]


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
