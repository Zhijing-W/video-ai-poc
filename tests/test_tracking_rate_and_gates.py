from __future__ import annotations

from pathlib import Path

import numpy as np

from app import event_analysis_pipeline as pipeline
from app import tracker
from app.core import config
from app.utils.image_utils import seconds_to_timestamp
from app.video_processor import Frame
from tests.conftest import write_image


def test_tracker_buffer_is_time_based_across_frame_rates() -> None:
    with tracker.settings.override(track_buffer_seconds=1.0):
        assert tracker._build_args("botsort", 2).track_buffer == 2
        assert tracker._build_args("botsort", 15).track_buffer == 15
        assert tracker._build_args("botsort", 30).track_buffer == 30
        reid_tracker = tracker._build_tracker("botsort_reid", 15)
        assert reid_tracker.args.with_reid is True
        assert reid_tracker.args.model == "app_reid"
        assert isinstance(reid_tracker.encoder, tracker._AppReIDEncoder)


def test_tracker_reid_encoder_uses_independent_osnet(monkeypatch) -> None:
    model = object()
    load_calls = []
    embed_calls = []
    monkeypatch.setattr(
        tracker.reid_mod,
        "_load_osnet",
        lambda: load_calls.append(True) or model,
    )
    monkeypatch.setattr(
        tracker.reid_mod,
        "_embed_osnet",
        lambda loaded, crop: embed_calls.append((loaded, crop.size))
        or np.ones(512, dtype=np.float32),
    )
    encoder = tracker._AppReIDEncoder()

    features = encoder(
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.asarray([[50, 50, 20, 40]], dtype=np.float32),
    )

    assert load_calls == [True]
    assert embed_calls == [(model, (20, 40))]
    assert len(features) == 1
    assert features[0].shape == (512,)


def test_legacy_frame_settings_convert_to_seconds(monkeypatch) -> None:
    monkeypatch.delenv("TRACK_BUFFER_SECONDS", raising=False)
    monkeypatch.setenv("TRACK_BUFFER", "60")
    monkeypatch.delenv("FACE_CANDIDATE_MIN_GAP_SECONDS", raising=False)
    monkeypatch.setenv("FACE_CANDIDATE_MIN_GAP_FRAMES", "3")

    assert config._seconds_setting(
        "TRACK_BUFFER_SECONDS",
        "TRACK_BUFFER",
        1.0,
        legacy_reference_fps=30,
    ) == 2.0
    assert config._seconds_setting(
        "FACE_CANDIDATE_MIN_GAP_SECONDS",
        "FACE_CANDIDATE_MIN_GAP_FRAMES",
        0.5,
        legacy_reference_fps=2,
    ) == 1.5


def test_semantic_sampling_is_independent_from_tracking_rate() -> None:
    indices = list(range(47))

    selected = pipeline._rate_limited_indices(
        indices,
        source_fps=15,
        target_fps=2,
    )

    assert selected == [0, 8, 15, 23, 30, 38, 45]


def test_tracking_frame_cap_preserves_semantic_video_coverage() -> None:
    assert pipeline._tracking_frame_cap(
        semantic_max_frames=300,
        tracking_max_frames=None,
        semantic_fps=2,
        tracking_fps=15,
    ) == 2250
    assert pipeline._tracking_frame_cap(
        semantic_max_frames=300,
        tracking_max_frames=900,
        semantic_fps=2,
        tracking_fps=15,
    ) == 900


def test_auto_enrollment_uses_time_and_observations() -> None:
    short_track = {
        "first": 0,
        "last": 2,
        "boxes": {0: [], 1: [], 2: []},
        "best_q": 100.0,
    }
    stable_track = {
        "first": 0,
        "last": 8,
        "boxes": {index: [] for index in range(9)},
        "best_q": 100.0,
    }
    sparse_track = {
        "first": 0,
        "last": 15,
        "boxes": {0: [], 15: []},
        "best_q": 100.0,
    }

    with pipeline.settings.override(
        track_enroll_min_seconds=0.5,
        track_enroll_min_observations=2,
    ):
        short = pipeline._track_enrollment_metadata(
            short_track,
            tracking_fps=15,
        )
        stable = pipeline._track_enrollment_metadata(
            stable_track,
            tracking_fps=15,
        )
        sparse = pipeline._track_enrollment_metadata(
            sparse_track,
            tracking_fps=15,
        )

    assert short["track_observations"] == 3
    assert short["track_span_seconds"] == 0.133
    assert short["track_observed_seconds"] == 0.2
    assert short["track_duration_seconds"] == 0.2
    assert short["enrollment_eligible"] is False
    assert short["enrollment_reasons"] == ["duration"]
    assert stable["track_span_seconds"] == 0.533
    assert stable["track_observed_seconds"] == 0.6
    assert stable["enrollment_eligible"] is True
    assert sparse["track_span_seconds"] == 1.0
    assert sparse["track_observed_seconds"] == 0.133
    assert sparse["enrollment_eligible"] is False


def test_temporal_provider_uses_independent_higher_rate() -> None:
    tracking_indices = list(range(60))

    semantic = pipeline._rate_limited_indices(
        tracking_indices,
        source_fps=15,
        target_fps=2,
    )
    gait = pipeline._rate_limited_indices(
        tracking_indices,
        source_fps=15,
        target_fps=10,
    )

    assert len(semantic) == 8
    assert len(gait) == 40
    assert len(gait) >= pipeline.settings.gait_min_frames


def test_timestamp_format_preserves_subsecond_precision() -> None:
    assert seconds_to_timestamp(0) == "00:00:00"
    assert seconds_to_timestamp(1 / 15) == "00:00:00.067"
    assert seconds_to_timestamp(4.5) == "00:00:04.5"


def test_short_track_can_match_known_gallery_without_auto_enrollment(
    monkeypatch,
    runtime_dir: Path,
) -> None:
    frames = [
        Frame(
            frame_id=f"frame_{index:03d}",
            timestamp="00:00:00",
            timestamp_seconds=index / 10,
            local_path=str(
                write_image(
                    runtime_dir / "frames" / f"frame_{index:03d}.jpg"
                )
            ),
        )
        for index in range(2)
    ]
    auto_enroll_values = []

    monkeypatch.setattr(
        pipeline,
        "extract_frames",
        lambda *args, **kwargs: frames,
    )
    monkeypatch.setattr(pipeline.detector_mod, "prepare", lambda: None)
    monkeypatch.setattr(pipeline.detector_mod, "active_device", lambda: "cpu")
    monkeypatch.setattr(
        pipeline.tracker_mod,
        "reset_tracker",
        lambda session_id: True,
    )
    monkeypatch.setattr(
        pipeline.gallery_mod,
        "reset_gallery",
        lambda session_id: True,
    )
    monkeypatch.setattr(pipeline.reid_mod, "embed_dim", lambda: 3)
    monkeypatch.setattr(pipeline.reid_mod, "active_backend", lambda: "mock")
    monkeypatch.setattr(pipeline.reid_mod, "active_device", lambda: "cpu")
    monkeypatch.setattr(
        pipeline.reid_mod,
        "assess_quality",
        lambda crop: {
            "area": 4096,
            "blur_var": 100.0,
            "aspect_ratio": 2.0,
        },
    )
    monkeypatch.setattr(
        pipeline.reid_mod,
        "embed",
        lambda crop, **kwargs: [1.0, 0.0, 0.0],
    )
    monkeypatch.setattr(
        pipeline.tracker_mod,
        "track_objects",
        lambda raw, **kwargs: {
            "detections": [
                {
                    "label": "person",
                    "track_id": 1,
                    "box": [10, 0, 60, 90],
                    "confidence": 0.95,
                }
            ],
            "infer_ms": 1.0,
            "track_ms": 1.0,
        },
    )
    monkeypatch.setattr(
        pipeline.tracker_mod,
        "active_backend",
        lambda: "mock-tracker",
    )

    class FakeGallery:
        def identify_or_enroll(self, *args, auto_enroll, **kwargs):
            auto_enroll_values.append(auto_enroll)
            return {
                "subject_id": 1,
                "score": 0.95,
                "decision": "hit",
                "enrolled": False,
                "quality_ok": True,
                "quality_reason": None,
                "label": "Alice",
            }

    monkeypatch.setattr(
        pipeline.gallery_mod,
        "with_gallery_locked",
        lambda session_id, dim, callback: callback(FakeGallery()),
    )

    with pipeline.settings.override(
        track_enroll_min_seconds=0.5,
        track_enroll_min_observations=2,
    ):
        result = pipeline.analyze_event_stream(
            "ignored.mp4",
            runtime_dir / "analysis",
            fps=2,
            tracking_fps=10,
            run_llm=False,
        )

    assert auto_enroll_values == [False]
    assert result["tracks"]["1"]["db_identity"] == "Alice"
    assert result["tracks"]["1"]["enrollment_eligible"] is False
