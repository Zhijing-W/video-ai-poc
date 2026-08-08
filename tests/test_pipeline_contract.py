from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app import event_analysis_pipeline as pipeline
from app.ocr import format_scene_context
from app.pipeline.object_context import format_object_context
from app.video_processor import Frame
from tests.conftest import write_image


def test_analyze_event_stream_keeps_top_level_contract_with_lightweight_mocks(monkeypatch, runtime_dir: Path) -> None:
    frame_dir = runtime_dir / "frames"
    frames = [
        Frame(frame_id="frame_001", timestamp="00:00:00", local_path=str(write_image(frame_dir / "frame_001.jpg", (32, 64, 96)))),
        Frame(frame_id="frame_002", timestamp="00:00:01", local_path=str(write_image(frame_dir / "frame_002.jpg", (96, 64, 32)))),
    ]

    monkeypatch.setattr(pipeline, "extract_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr(pipeline.detector_mod, "prepare", lambda: None)
    monkeypatch.setattr(pipeline.detector_mod, "active_device", lambda: "cpu")
    monkeypatch.setattr(pipeline.tracker_mod, "reset_tracker", lambda session_id: True)
    monkeypatch.setattr(pipeline.gallery_mod, "reset_gallery", lambda session_id: True)
    monkeypatch.setattr(pipeline.reid_mod, "embed_dim", lambda: 512)
    monkeypatch.setattr(pipeline.tracker_mod, "track_objects", lambda raw, session_id=None: {"detections": []})
    monkeypatch.setattr(pipeline.tracker_mod, "active_backend", lambda: "mock-tracker")
    monkeypatch.setattr(pipeline.reid_mod, "active_backend", lambda: "mock-reid")
    monkeypatch.setattr(pipeline.reid_mod, "active_device", lambda: "cpu")

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
    assert {
        "extract_frames",
        "pipeline_setup",
        "frame_preprocess",
        "event_preparation",
    } <= set(result["stage_timings"])
    assert result["runtime"] == {
        "execution": "server",
        "detector_device": "cpu",
        "reid_device": "cpu",
    }
    assert result["body_reid_timing"]["backend"] == "mock-reid"
    assert result["body_reid_timing"]["device"] == "cpu"
    assert result["body_reid_timing"]["call_count"] == 0
    assert result["body_reid_timing"]["mean_ms"] is None
    assert result["body_reid_timing"]["p95_ms"] is None


def test_analyze_event_stream_can_disable_body_identity_without_blocking_other_routes(
    monkeypatch,
    runtime_dir: Path,
) -> None:
    frame_dir = runtime_dir / "frames"
    frames = [
        Frame(
            frame_id="frame_001",
            timestamp="00:00:00",
            local_path=str(write_image(frame_dir / "frame_001.jpg", (32, 64, 96))),
        ),
        Frame(
            frame_id="frame_002",
            timestamp="00:00:01",
            local_path=str(write_image(frame_dir / "frame_002.jpg", (96, 64, 32))),
        ),
    ]
    face_calls: list[dict] = []

    monkeypatch.setattr(pipeline, "extract_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr(pipeline.detector_mod, "prepare", lambda: None)
    monkeypatch.setattr(pipeline.detector_mod, "active_device", lambda: "cuda:0")
    monkeypatch.setattr(pipeline.settings, "track_min_frames", 0)
    monkeypatch.setattr(pipeline.tracker_mod, "reset_tracker", lambda session_id: True)
    monkeypatch.setattr(pipeline.gallery_mod, "reset_gallery", lambda session_id: True)
    monkeypatch.setattr(
        pipeline.tracker_mod,
        "track_objects",
        lambda raw, session_id=None: {
            "detections": [
                {
                    "label": "person",
                    "track_id": 1,
                    "box": [0, 0, 24, 24],
                    "confidence": 0.95,
                }
            ],
            "infer_ms": 1.0,
            "track_ms": 1.0,
        },
    )
    monkeypatch.setattr(pipeline.tracker_mod, "active_backend", lambda: "mock-tracker")
    monkeypatch.setattr(
        pipeline.reid_mod,
        "assess_quality",
        lambda crop: {"sharpness": 1.0, "area": 576},
    )
    monkeypatch.setattr(
        pipeline.reid_mod,
        "embed_dim",
        lambda: (_ for _ in ()).throw(AssertionError("body identity must stay unloaded")),
    )
    monkeypatch.setattr(
        pipeline.reid_mod,
        "embed",
        lambda crop: (_ for _ in ()).throw(AssertionError("body identity must stay unused")),
    )
    monkeypatch.setattr(
        pipeline.reid_mod,
        "active_backend",
        lambda: (_ for _ in ()).throw(AssertionError("body backend must stay unresolved")),
    )
    monkeypatch.setattr(pipeline.gait_mod, "available", lambda: True)
    monkeypatch.setattr(pipeline.gait_mod, "extract_persons", lambda image: [])
    monkeypatch.setattr(pipeline.gait_mod, "load_error", lambda: None)

    def fake_attach_faces(frames, tracks, identities, session_id, **kwargs):
        face_calls.append(kwargs)
        identities[1]["face"] = {"observed": True}

    monkeypatch.setattr(pipeline, "_attach_faces", fake_attach_faces)

    result = pipeline.analyze_event_stream(
        video_path="ignored.mp4",
        out_dir=runtime_dir / "analysis-no-body",
        fps=1.0,
        run_llm=False,
        session_id="no-body-session",
        with_body=False,
        with_face=True,
        with_gait=True,
    )

    assert result["with_body"] is False
    assert result["reid_backend"] is None
    assert result["reid_dim"] is None
    assert result["with_face"] is True
    assert result["with_gait"] is True
    assert result["runtime"]["detector_device"] == "cuda:0"
    assert result["runtime"]["reid_device"] is None
    assert result["body_reid_timing"]["call_count"] == 0
    assert result["body_reid_timing"]["backend"] is None
    assert result["body_reid_timing"]["device"] is None
    assert "body_identity" not in result["stage_timings"]
    assert result["stage_timings"]["object_detection"] == 0.002
    assert result["stage_timings"]["multi_object_tracking"] == 0.002
    assert face_calls == [
        {
            "body_embeddings": {},
            "body_consistency_enabled": False,
        }
    ]


def test_fatal_run_preserves_failed_reid_telemetry_without_leaking(
    monkeypatch,
    runtime_dir: Path,
) -> None:
    frame_path = write_image(runtime_dir / "fatal-frame.jpg")
    frames = [
        Frame(
            frame_id="frame_001",
            timestamp="00:00:00",
            local_path=str(frame_path),
        )
    ]
    loaded = SimpleNamespace(dim=3, device="cpu")

    def fail_embedding(model, crop):
        raise RuntimeError("embedding exploded")

    pipeline.reid_mod.register_backend(
        "unit-fatal-telemetry",
        lambda: loaded,
        fail_embedding,
        dim=None,
        replace=True,
    )
    monkeypatch.setattr(pipeline, "extract_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr(pipeline.detector_mod, "prepare", lambda: None)
    monkeypatch.setattr(pipeline.tracker_mod, "reset_tracker", lambda session_id: True)
    monkeypatch.setattr(pipeline.gallery_mod, "reset_gallery", lambda session_id: True)
    monkeypatch.setattr(
        pipeline.tracker_mod,
        "track_objects",
        lambda raw, session_id=None: pipeline.reid_mod.embed(
            Image.new("RGB", (16, 32)),
            purpose="tracking",
        ),
    )

    pipeline.reid_mod.reset_backend()
    try:
        with pipeline.settings.override(reid_backend="unit-fatal-telemetry"):
            with pytest.raises(
                pipeline.EventAnalysisRunError,
                match="embedding exploded",
            ) as first:
                pipeline.analyze_event_stream(
                    "ignored.mp4",
                    runtime_dir / "fatal-analysis",
                    run_llm=False,
                )

            first_timing = first.value.body_reid_timing
            assert first_timing["backend"] == "unit_fatal_telemetry"
            assert first_timing["device"] == "cpu"
            assert first_timing["call_count"] == 1
            assert first_timing["failed_call_count"] == 1
            assert first_timing["by_purpose"]["tracking"]["call_count"] == 1

            monkeypatch.setattr(
                pipeline.tracker_mod,
                "track_objects",
                lambda raw, session_id=None: (_ for _ in ()).throw(
                    RuntimeError("tracker exploded")
                ),
            )
            with pytest.raises(pipeline.EventAnalysisRunError) as second:
                pipeline.analyze_event_stream(
                    "ignored.mp4",
                    runtime_dir / "second-fatal-analysis",
                    run_llm=False,
                )

            second_timing = second.value.body_reid_timing
            assert second_timing["call_count"] == 0
            assert second_timing["failed_call_count"] == 0
            assert second_timing["by_purpose"] == {}
    finally:
        pipeline.reid_mod.reset_backend()


def test_tracking_reid_telemetry_survives_disabled_identity_branch(
    monkeypatch,
    runtime_dir: Path,
) -> None:
    frame_path = write_image(runtime_dir / "tracking-only-frame.jpg")
    frames = [
        Frame(
            frame_id="frame_001",
            timestamp="00:00:00",
            local_path=str(frame_path),
        )
    ]
    loaded = SimpleNamespace(dim=3, device="cpu")
    pipeline.reid_mod.register_backend(
        "unit-tracking-only",
        lambda: loaded,
        lambda model, crop: [1.0, 0.0, 0.0],
        dim=None,
        replace=True,
    )
    monkeypatch.setattr(pipeline, "extract_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr(pipeline.detector_mod, "prepare", lambda: None)
    monkeypatch.setattr(pipeline.detector_mod, "active_device", lambda: "cpu")
    monkeypatch.setattr(pipeline.tracker_mod, "reset_tracker", lambda session_id: True)
    monkeypatch.setattr(pipeline.gallery_mod, "reset_gallery", lambda session_id: True)

    def track_with_reid(raw, session_id=None):
        pipeline.reid_mod.embed(
            Image.new("RGB", (16, 32)),
            purpose="tracking",
        )
        return {"detections": []}

    monkeypatch.setattr(pipeline.tracker_mod, "track_objects", track_with_reid)
    monkeypatch.setattr(pipeline.tracker_mod, "active_backend", lambda: "botsort_reid")

    pipeline.reid_mod.reset_backend()
    try:
        with pipeline.settings.override(reid_backend="unit-tracking-only"):
            result = pipeline.analyze_event_stream(
                "ignored.mp4",
                runtime_dir / "tracking-only-analysis",
                run_llm=False,
                with_body=False,
            )

        assert result["with_body"] is False
        assert result["reid_backend"] is None
        assert result["runtime"]["reid_device"] == "cpu"
        assert result["body_reid_timing"]["backend"] == "unit_tracking_only"
        assert result["body_reid_timing"]["device"] == "cpu"
        assert result["body_reid_timing"]["call_count"] == 1
        assert result["body_reid_timing"]["failed_call_count"] == 0
        assert result["body_reid_timing"]["by_purpose"]["tracking"]["call_count"] == 1
    finally:
        pipeline.reid_mod.reset_backend()


def test_scene_context_uses_requested_report_language() -> None:
    frames = [
        {
            "frame_index": 3,
            "timestamp": "00:00:01",
            "texts": [{"text": "CAM-01"}],
        }
    ]

    english = format_scene_context(frames, language="en")
    chinese = format_scene_context(frames)

    assert english.startswith("[Scene text")
    assert "Per-frame text" in english
    assert "画面文字" not in english
    assert chinese.startswith("【画面文字")
    assert "逐帧文字" in chinese


def test_object_context_uses_requested_report_language() -> None:
    objects = [
        {
            "track_id": 7,
            "label": "backpack",
            "label_cn": "背包",
            "first_ts": "00:00:01",
            "last_ts": "00:00:03",
            "first_frame": 2,
            "last_frame": 6,
            "direction": "left_to_right",
            "frames_present": 5,
            "conf": 0.9,
        }
    ]

    english = format_object_context(objects, language="en")
    chinese = format_object_context(objects)

    assert english.startswith("[Objects in frame")
    assert "backpack track#7" in english
    assert "画面中的物体" not in english
    assert chinese.startswith("【画面中的物体")
    assert "背包(backpack)" in chinese
