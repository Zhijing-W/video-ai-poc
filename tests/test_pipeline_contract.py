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
    monkeypatch.setattr(pipeline.detector_mod, "prepare", lambda: None)
    monkeypatch.setattr(pipeline.detector_mod, "active_device", lambda: "cpu")
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
    assert {
        "extract_frames",
        "pipeline_setup",
        "frame_preprocess",
        "event_preparation",
    } <= set(result["stage_timings"])
    assert result["runtime"] == {
        "execution": "server",
        "detector_device": "cpu",
    }


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
    assert "body_identity" not in result["stage_timings"]
    assert result["stage_timings"]["object_detection"] == 0.002
    assert result["stage_timings"]["multi_object_tracking"] == 0.002
    assert face_calls == [
        {
            "body_embeddings": {},
            "body_consistency_enabled": False,
        }
    ]


def test_analyze_event_stream_seeds_demo_gallery_names_for_new_subjects(
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
            local_path=str(write_image(frame_dir / "frame_002.jpg", (32, 64, 96))),
        ),
        Frame(
            frame_id="frame_003",
            timestamp="00:00:02",
            local_path=str(write_image(frame_dir / "frame_003.jpg", (32, 64, 96))),
        ),
    ]
    renamed: list[tuple[int, str | None]] = []

    monkeypatch.setattr(pipeline, "extract_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr(pipeline.detector_mod, "prepare", lambda: None)
    monkeypatch.setattr(pipeline.detector_mod, "active_device", lambda: "cpu")
    monkeypatch.setattr(pipeline.tracker_mod, "reset_tracker", lambda session_id: True)
    monkeypatch.setattr(pipeline.gallery_mod, "reset_gallery", lambda session_id: True)
    monkeypatch.setattr(pipeline.reid_mod, "embed_dim", lambda: 512)
    monkeypatch.setattr(
        pipeline.reid_mod,
        "assess_quality",
        lambda crop: {"sharpness": 1.0, "area": 4096},
    )
    monkeypatch.setattr(pipeline.reid_mod, "embed", lambda crop: [1.0] * 512)
    monkeypatch.setattr(
        pipeline.tracker_mod,
        "track_objects",
        lambda raw, session_id=None: {
            "detections": [
                {
                    "label": "person",
                    "track_id": 1,
                    "box": [0, 0, 48, 48],
                    "confidence": 0.95,
                }
            ],
            "infer_ms": 1.0,
            "track_ms": 1.0,
        },
    )
    monkeypatch.setattr(pipeline.tracker_mod, "active_backend", lambda: "mock-tracker")
    monkeypatch.setattr(pipeline.reid_mod, "active_backend", lambda: "mock-reid")
    monkeypatch.setattr(pipeline.settings, "track_min_frames", 0)
    monkeypatch.setattr(pipeline.settings, "demo_gallery_enabled", True)
    monkeypatch.setattr(pipeline.settings, "demo_gallery_names", "Alice,Bob,Carol")

    class FakeGallery:
        def identify_or_enroll(self, *args, **kwargs):
            return {
                "subject_id": 7,
                "score": 0.91,
                "decision": "new",
                "enrolled": True,
                "quality_ok": True,
                "label": None,
            }

        def rename_subject(self, subject_id: int, label: str | None) -> None:
            renamed.append((subject_id, label))

    fake_gallery = FakeGallery()
    monkeypatch.setattr(
        pipeline.gallery_mod,
        "with_gallery_locked",
        lambda session_id, dim, callback: callback(fake_gallery),
    )

    result = pipeline.analyze_event_stream(
        video_path="ignored.mp4",
        out_dir=runtime_dir / "analysis-demo",
        fps=1.0,
        run_llm=False,
        session_id="demo-session",
    )

    assert renamed == [(7, "Alice")]
    assert result["windows"][0]["people"][0]["db_identity"] == "Alice"
    assert "Alice" in result["windows"][0]["identity_context"]
