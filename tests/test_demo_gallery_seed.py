from __future__ import annotations

from pathlib import Path

from app import event_analysis_pipeline as pipeline
from app.video_processor import Frame
from tests.conftest import write_image


def test_analyze_event_stream_seeds_demo_gallery_names_for_new_subjects(
    monkeypatch,
    runtime_dir: Path,
) -> None:
    frame_dir = runtime_dir / "frames"
    frames = [
        Frame(
            frame_id=f"frame_{index:03d}",
            timestamp=f"00:00:0{index}",
            local_path=str(
                write_image(frame_dir / f"frame_{index:03d}.jpg", (32, 64, 96))
            ),
        )
        for index in range(3)
    ]
    renamed: list[tuple[int, str | None]] = []

    monkeypatch.setattr(pipeline, "extract_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr(pipeline.detector_mod, "prepare", lambda: None)
    monkeypatch.setattr(pipeline.detector_mod, "active_device", lambda: "cpu")
    monkeypatch.setattr(pipeline.tracker_mod, "reset_tracker", lambda session_id: True)
    monkeypatch.setattr(pipeline.gallery_mod, "reset_gallery", lambda session_id: True)
    monkeypatch.setattr(pipeline.reid_mod, "embed_dim", lambda: 512)
    monkeypatch.setattr(pipeline.reid_mod, "active_backend", lambda: "mock-reid")
    monkeypatch.setattr(pipeline.reid_mod, "active_device", lambda: "cpu")
    monkeypatch.setattr(
        pipeline.reid_mod,
        "assess_quality",
        lambda crop: {"sharpness": 1.0, "area": 4096},
    )
    monkeypatch.setattr(
        pipeline.reid_mod,
        "embed",
        lambda crop, **kwargs: [1.0] * 512,
    )
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
