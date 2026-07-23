from __future__ import annotations

import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app import face
from app.routers import event_monitor


def test_router_generates_unique_run_and_session_ids(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_analyze(video_path, run_dir, **kwargs):
        calls.append(
            {
                "video_path": str(video_path),
                "run_dir": str(run_dir),
                "session_id": kwargs["session_id"],
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

    for call in calls:
        shutil.rmtree(Path(call["run_dir"]), ignore_errors=True)


def test_fastapi_health_page_and_openapi_are_available() -> None:
    client = TestClient(app)

    health = client.get("/health")
    page = client.get("/event-monitor")
    openapi = client.get("/openapi.json")

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "feature": "event-monitor"}
    assert page.status_code == 200
    assert "event-monitor" in page.text.lower()
    assert openapi.status_code == 200
    assert "/api/event-monitor/understand" in openapi.text


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
