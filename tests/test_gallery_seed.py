from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app import event_analysis_pipeline as pipeline
from app.identity.embedding_gallery import SessionGallery
from app.identity.gallery_seed import (
    GallerySeedError,
    face_seed_quality_ok,
    load_gallery_seed,
)
from app.video_processor import Frame
from tests.conftest import write_image


def _write_seed(runtime_dir: Path) -> tuple[Path, object]:
    sample = runtime_dir / "samples" / "chokepoint_real.mp4"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(b"video")
    write_image(sample.parent / "gallery" / "alice" / "body_001.jpg")
    write_image(sample.parent / "gallery" / "alice" / "face_001.jpg")
    manifest = {
        "version": 1,
        "dataset": {
            "name": "ChokePoint",
            "sequence": "P1E_S1",
            "analysis_camera": "C1",
            "gallery_camera": "C2",
        },
        "subjects": [
            {
                "label": "Alice",
                "source_id": "0003",
                "body_images": ["gallery/alice/body_001.jpg"],
                "face_images": ["gallery/alice/face_001.jpg"],
            }
        ],
    }
    sample.with_suffix(".gallery.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return sample, load_gallery_seed(sample)


def test_load_gallery_seed_returns_safe_public_metadata(runtime_dir: Path) -> None:
    _, seed = _write_seed(runtime_dir)

    assert seed is not None
    assert seed.subjects[0].body_images[0].is_file()
    assert seed.public_dict() == {
        "version": 1,
        "manifest": "chokepoint_real.gallery.json",
        "dataset": {
            "name": "ChokePoint",
            "sequence": "P1E_S1",
            "analysis_camera": "C1",
            "gallery_camera": "C2",
        },
        "subject_count": 1,
        "subjects": [
            {
                "label": "Alice",
                "source_id": "0003",
                "body_shots": 1,
                "face_shots": 1,
            }
        ],
    }


def test_load_gallery_seed_rejects_path_escape(runtime_dir: Path) -> None:
    sample = runtime_dir / "samples" / "unsafe.mp4"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(b"video")
    sample.with_suffix(".gallery.json").write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": {"name": "ChokePoint"},
                "subjects": [
                    {
                        "label": "Alice",
                        "source_id": "0003",
                        "body_images": ["../outside.jpg"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GallerySeedError, match="escapes"):
        load_gallery_seed(sample)


def test_trusted_face_seed_accepts_direct_marginal_reference() -> None:
    accepted, reason = face_seed_quality_ok(
        {
            "enhanced": False,
            "eligibility": "direct",
            "category": "marginal",
            "can_match": True,
            "can_enroll": False,
        }
    )

    assert accepted is True
    assert reason is None
    assert face_seed_quality_ok(
        {
            "enhanced": False,
            "eligibility": "fallback",
            "category": "marginal",
            "can_match": True,
        }
    ) == (False, "face_not_direct")


def test_explicit_labeled_enrollment_reuses_one_subject(monkeypatch) -> None:
    class FakeIndex:
        def __init__(self, dim: int) -> None:
            self.dim = dim
            self.vectors = {}

        @property
        def ntotal(self) -> int:
            return len(self.vectors)

        def add_with_ids(self, vectors, ids) -> None:
            for vector, row_id in zip(vectors, ids):
                self.vectors[int(row_id)] = np.asarray(vector, dtype=np.float32)

        def remove_ids(self, ids) -> None:
            for row_id in ids:
                self.vectors.pop(int(row_id), None)

        def search(self, vector, count):
            ranked = sorted(
                (
                    (float(np.dot(vector[0], stored)), row_id)
                    for row_id, stored in self.vectors.items()
                ),
                reverse=True,
            )[:count]
            return (
                np.asarray([[score for score, _ in ranked]], dtype=np.float32),
                np.asarray([[row_id for _, row_id in ranked]], dtype=np.int64),
            )

    fake_faiss = SimpleNamespace(
        IndexFlatIP=lambda dim: dim,
        IndexIDMap2=lambda dim: FakeIndex(dim),
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    gallery = SessionGallery(3)

    first = gallery.enroll_labeled(
        [1.0, 0.0, 0.0],
        {"area": 4096, "blur_var": 100.0, "aspect_ratio": 2.0},
        label="Alice",
    )
    second = gallery.enroll_labeled(
        [0.99, 0.01, 0.0],
        {"area": 4096, "blur_var": 100.0, "aspect_ratio": 2.0},
        label="Alice",
    )
    labeled = gallery.identify_or_enroll(
        [0.0, 1.0, 0.0],
        {"area": 4096, "blur_var": 100.0, "aspect_ratio": 2.0},
        label="Alice",
        new_thresh=0.4,
    )
    hit = gallery.identify([1.0, 0.0, 0.0], hit_thresh=0.8)

    assert first["subject_id"] == second["subject_id"]
    assert labeled["subject_id"] == first["subject_id"]
    assert labeled["decision"] == "labeled"
    assert gallery.stats()["subjects"] == 1
    assert gallery.stats()["total_shots"] == 3
    assert hit["decision"] == "hit"
    assert hit["label"] == "Alice"


def test_face_label_is_persisted_into_body_gallery(monkeypatch) -> None:
    renamed = []

    class FakeGallery:
        def rename_subject(self, subject_id, label):
            renamed.append((subject_id, label))
            return 3

    monkeypatch.setattr(
        pipeline.gallery_mod,
        "with_gallery_locked",
        lambda session_id, dim, callback: callback(FakeGallery()),
    )
    identities = {
        1: {
            "subject_id": 7,
            "route_subject": {"route": "body", "local_subject_id": 7},
            "db_identity": None,
            "face": {"db_identity": "Alice"},
        }
    }

    pipeline._backfill_body_gallery_labels(identities, "seed-session", 1024)

    assert identities[1]["db_identity"] == "Alice"
    assert renamed == [(7, "Alice")]
    assert identities[1]["subject_id"] == 3
    assert identities[1]["route_subject"]["local_subject_id"] == 3


def test_face_only_label_backfill_does_not_require_body_gallery() -> None:
    identities = {
        1: {
            "subject_id": None,
            "db_identity": None,
            "face": {"db_identity": "Alice"},
        }
    }

    pipeline._backfill_body_gallery_labels(identities, "face-only", None)

    assert identities[1]["db_identity"] == "Alice"


def test_body_label_backfill_reuses_cached_merge(monkeypatch) -> None:
    rename_calls = []

    class FakeGallery:
        def rename_subject(self, subject_id, label):
            rename_calls.append((subject_id, label))
            return 3

    monkeypatch.setattr(
        pipeline.gallery_mod,
        "with_gallery_locked",
        lambda session_id, dim, callback: callback(FakeGallery()),
    )
    identities = {
        track_id: {
            "subject_id": 7,
            "route_subject": {"route": "body", "local_subject_id": 7},
            "db_identity": None,
            "face": {"db_identity": "Alice"},
        }
        for track_id in (1, 2)
    }

    pipeline._backfill_body_gallery_labels(identities, "seed-session", 1024)

    assert rename_calls == [(7, "Alice")]
    assert {item["subject_id"] for item in identities.values()} == {3}


def test_body_label_backfill_updates_tracks_without_face_evidence(
    monkeypatch,
) -> None:
    rename_calls = []

    class FakeGallery:
        def rename_subject(self, subject_id, label):
            rename_calls.append((subject_id, label))
            return 3

    monkeypatch.setattr(
        pipeline.gallery_mod,
        "with_gallery_locked",
        lambda session_id, dim, callback: callback(FakeGallery()),
    )
    identities = {
        1: {
            "subject_id": 7,
            "route_subject": {"route": "body", "local_subject_id": 7},
            "db_identity": None,
            "face": {"db_identity": "Alice"},
        },
        2: {
            "subject_id": 7,
            "route_subject": {"route": "body", "local_subject_id": 7},
            "db_identity": None,
            "face": None,
        },
    }

    pipeline._backfill_body_gallery_labels(identities, "seed-session", 1024)

    assert rename_calls == [(7, "Alice")]
    assert {item["subject_id"] for item in identities.values()} == {3}
    assert {item["db_identity"] for item in identities.values()} == {"Alice"}


def test_renaming_merges_into_existing_labeled_subject(monkeypatch) -> None:
    class FakeIndex:
        def __init__(self, dim: int) -> None:
            self.vectors = {}

        @property
        def ntotal(self) -> int:
            return len(self.vectors)

        def add_with_ids(self, vectors, ids) -> None:
            for vector, row_id in zip(vectors, ids):
                self.vectors[int(row_id)] = np.asarray(vector, dtype=np.float32)

        def remove_ids(self, ids) -> None:
            for row_id in ids:
                self.vectors.pop(int(row_id), None)

        def search(self, vector, count):
            ranked = sorted(
                (
                    (float(np.dot(vector[0], stored)), row_id)
                    for row_id, stored in self.vectors.items()
                ),
                reverse=True,
            )[:count]
            return (
                np.asarray([[score for score, _ in ranked]], dtype=np.float32),
                np.asarray([[row_id for _, row_id in ranked]], dtype=np.int64),
            )

    monkeypatch.setitem(
        sys.modules,
        "faiss",
        SimpleNamespace(
            IndexFlatIP=lambda dim: dim,
            IndexIDMap2=lambda dim: FakeIndex(dim),
        ),
    )
    gallery = SessionGallery(3)
    known = gallery.enroll_labeled(
        [1.0, 0.0, 0.0],
        {"area": 4096, "blur_var": 100.0, "aspect_ratio": 2.0},
        label="Alice",
    )
    unknown = gallery.identify_or_enroll(
        [0.0, 1.0, 0.0],
        {"area": 4096, "blur_var": 100.0, "aspect_ratio": 2.0},
        new_thresh=0.4,
    )

    canonical = gallery.rename_subject(unknown["subject_id"], "Alice")
    merged_hit = gallery.identify([0.0, 1.0, 0.0], hit_thresh=0.8)

    assert canonical == known["subject_id"]
    assert gallery.stats()["subjects"] == 1
    assert merged_hit["subject_id"] == known["subject_id"]
    assert merged_hit["label"] == "Alice"


def test_seed_assets_do_not_enter_video_tracking(
    monkeypatch,
    runtime_dir: Path,
) -> None:
    _, seed = _write_seed(runtime_dir)
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
    tracker_calls = []

    monkeypatch.setattr(pipeline, "extract_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr(pipeline.detector_mod, "prepare", lambda: None)
    monkeypatch.setattr(pipeline.detector_mod, "active_device", lambda: "cpu")
    monkeypatch.setattr(pipeline.tracker_mod, "reset_tracker", lambda session_id: True)
    monkeypatch.setattr(pipeline.gallery_mod, "reset_gallery", lambda session_id: True)
    monkeypatch.setattr(pipeline.reid_mod, "embed_dim", lambda: 3)
    monkeypatch.setattr(pipeline.reid_mod, "active_backend", lambda: "mock-reid")
    monkeypatch.setattr(pipeline.reid_mod, "active_device", lambda: "cpu")
    monkeypatch.setattr(
        pipeline.reid_mod,
        "assess_quality",
        lambda crop: {"area": 4096, "blur_var": 100.0, "aspect_ratio": 0.5},
    )
    monkeypatch.setattr(
        pipeline.reid_mod,
        "embed",
        lambda crop, **kwargs: [1.0, 0.0, 0.0],
    )

    def track_objects(raw, session_id=None):
        tracker_calls.append(raw)
        return {
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
        }

    monkeypatch.setattr(pipeline.tracker_mod, "track_objects", track_objects)
    monkeypatch.setattr(pipeline.tracker_mod, "active_backend", lambda: "mock-tracker")
    monkeypatch.setattr(pipeline.settings, "track_min_frames", 0)

    class FakeGallery:
        seeded = False

        def enroll_labeled(self, *args, label, **kwargs):
            self.seeded = True
            return {
                "subject_id": 7,
                "label": label,
                "decision": "seeded",
                "enrolled": True,
                "quality_ok": True,
            }

        def identify_or_enroll(self, *args, **kwargs):
            assert self.seeded
            return {
                "subject_id": 7,
                "score": 0.91,
                "decision": "hit",
                "enrolled": False,
                "quality_ok": True,
                "label": "Alice",
            }

    fake_gallery = FakeGallery()
    monkeypatch.setattr(
        pipeline.gallery_mod,
        "with_gallery_locked",
        lambda session_id, dim, callback: callback(fake_gallery),
    )

    result = pipeline.analyze_event_stream(
        video_path="ignored.mp4",
        out_dir=runtime_dir / "analysis",
        fps=1.0,
        run_llm=False,
        session_id="seed-session",
        gallery_seed=seed,
    )

    assert len(tracker_calls) == len(frames)
    assert result["frames_total"] == len(frames)
    assert result["gallery_seed"]["runtime"]["body"]["total_shots"] == 1
    assert result["windows"][0]["people"][0]["db_identity"] == "Alice"
