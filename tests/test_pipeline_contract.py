from __future__ import annotations

from pathlib import Path

from app import event_analysis_pipeline as pipeline
from app.video_processor import Frame
from tests.conftest import write_image


def test_analyze_event_stream_keeps_top_level_contract_with_lightweight_mocks(monkeypatch, runtime_dir: Path) -> None:
    frame_dir = runtime_dir / "frames"
    frames = [
        Frame(frame_id="frame_001", timestamp="00:00:00", local_path=str(write_image(frame_dir / "frame_001.jpg", (32, 64, 96)))),
        Frame(frame_id="frame_002", timestamp="00:00:01", local_path=str(write_image(frame_dir / "frame_002.jpg", (96, 64, 32)))),
    ]

    monkeypatch.setattr(pipeline, "extract_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr(pipeline.tracker_mod, "reset_tracker", lambda session_id: True)
    monkeypatch.setattr(pipeline.gallery_mod, "reset_gallery", lambda session_id: True)
    monkeypatch.setattr(pipeline.reid_mod, "embed_dim", lambda: 512)
    monkeypatch.setattr(pipeline.tracker_mod, "track_objects", lambda raw, session_id=None: {"detections": []})
    monkeypatch.setattr(pipeline.tracker_mod, "active_backend", lambda: "mock-tracker")
    monkeypatch.setattr(pipeline.reid_mod, "active_backend", lambda: "mock-reid")

    result = pipeline.analyze_event_stream(
        video_path="ignored.mp4",
        out_dir=runtime_dir / "analysis",
        fps=1.0,
        run_llm=False,
        session_id="characterize-session",
    )

    assert result["video"] == "ignored.mp4"
    assert result["frames_total"] == 2
    assert result["session_id"] == "characterize-session"
    assert result["tracker_backend"] == "mock-tracker"
    assert result["reid_backend"] == "mock-reid"
    assert result["dry_run"] is True
    assert result["tracks"] == {}
    assert len(result["windows"]) == 1
    assert result["windows"][0]["keyframe_indices"] == [0]
    assert result["windows"][0]["time_range"] == ["00:00:00", "00:00:01"]
    assert {"extract_frames", "detect_track", "merge_fusion_thumb", "windows_select"} <= set(result["stage_timings"])
