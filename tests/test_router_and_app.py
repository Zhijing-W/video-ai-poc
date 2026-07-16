from __future__ import annotations

import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
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
