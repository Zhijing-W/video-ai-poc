from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app import face
from app.identity import embedding_gallery, face_attachment
from app.identity.evidence_selection import (
    body_quality_score,
    ensure_body_fallback,
    update_face_candidates,
)
from app.video_processor import Frame
from tests.conftest import write_image


def test_body_score_and_face_candidates_are_independent_and_bounded() -> None:
    assert body_quality_score({"blur_var": 50, "area": 20000}) == 50
    candidates = []
    for frame_index, score in [(0, 5.0), (1, 10.0), (3, 8.0), (6, 7.0)]:
        candidates = update_face_candidates(
            candidates,
            {
                "track_id": 9,
                "frame_index": frame_index,
                "timestamp": frame_index,
                "person_bbox": [0, 0, 10, 20],
                "proxy_score": score,
            },
            top_k=3,
            min_gap_frames=2,
        )

    assert [item["frame_index"] for item in candidates] == [1, 3, 6]
    body_best = {
        "track_id": 9,
        "frame_index": 4,
        "timestamp": 4,
        "person_bbox": [0, 0, 10, 20],
    }
    with_fallback = ensure_body_fallback(candidates, body_best, top_k=3)
    assert len(with_fallback) == 3
    assert any(item["frame_index"] == 4 for item in with_fallback)


def test_face_candidates_remove_all_temporal_neighbors_of_stronger_bridge() -> None:
    candidates = [
        {"frame_index": 10, "proxy_score": 100.0},
        {"frame_index": 12, "proxy_score": 90.0},
    ]

    selected = update_face_candidates(
        candidates,
        {"frame_index": 11, "proxy_score": 150.0},
        top_k=3,
        min_gap_frames=2,
    )

    assert [item["frame_index"] for item in selected] == [11]


def test_face_person_association_rejects_ambiguous_overlap(monkeypatch) -> None:
    monkeypatch.setattr(face.settings, "face_assoc_min_contain", 0.6)
    monkeypatch.setattr(face.settings, "face_assoc_ambiguity_margin", 0.08)
    monkeypatch.setattr(face.settings, "face_assoc_max_head_y_ratio", 0.48)
    faces = [{"bbox": [20, 10, 40, 30], "det_score": 0.9}]
    people = [
        {"track_id": 1, "label": "person", "box": [0, 0, 60, 100]},
        {"track_id": 2, "label": "person", "box": [0, 0, 60, 100]},
    ]

    assert face.associate_to_persons(faces, people) == {}


def test_body_gallery_quality_semantics_remain_unchanged() -> None:
    assert embedding_gallery.quality_ok(None) == (True, None)
    assert embedding_gallery.quality_ok({"area": 1, "blur_var": 100}) == (
        False,
        "too_small",
    )


def test_attach_faces_selects_face_best_separately_from_body_best(
    monkeypatch,
    runtime_dir: Path,
) -> None:
    frame0 = write_image(runtime_dir / "frame0.jpg", (30, 30, 30))
    frame1 = write_image(runtime_dir / "frame1.jpg", (220, 220, 220))
    frames = [
        Frame(frame_id="0", timestamp="00:00:00", local_path=str(frame0)),
        Frame(frame_id="1", timestamp="00:00:01", local_path=str(frame1)),
    ]
    tracks = {
        1: {
            "best_idx": 0,
            "best_box": [10, 0, 90, 150],
            "body_best": {
                "track_id": 1,
                "frame_index": 0,
                "timestamp": "00:00:00",
                "person_bbox": [10, 0, 90, 150],
            },
            "face_candidates": [
                {
                    "track_id": 1,
                    "frame_index": 0,
                    "timestamp": "00:00:00",
                    "person_bbox": [10, 0, 90, 150],
                    "proxy_score": 10,
                },
                {
                    "track_id": 1,
                    "frame_index": 1,
                    "timestamp": "00:00:01",
                    "person_bbox": [10, 0, 90, 150],
                    "proxy_score": 9,
                },
            ],
            "boxes": {0: [10, 0, 90, 150], 1: [10, 0, 90, 150]},
        }
    }
    identities = {1: {"track_id": 1}}

    detect_calls = []
    finalize_calls = []

    def fake_detect(image: Image.Image, **kwargs):
        detect_calls.append(kwargs)
        clear = np.asarray(image)[0, 0, 0] > 100
        return [
            {
                "bbox": [35, 10, 65, 45],
                "kps": [[42, 20], [58, 20], [50, 28], [44, 38], [56, 38]],
                "det_score": 0.95,
                "quality": {
                    "category": "clear" if clear else "poor",
                    "eligibility": "direct",
                    "quality": 0.9 if clear else 0.1,
                    "can_match": True,
                    "can_superres": False,
                    "can_enroll": clear,
                },
            }
        ]

    monkeypatch.setattr(face_attachment.face_mod, "detect", fake_detect)
    def finalize(image, frozen):
        finalize_calls.append(frozen)
        return {
            **frozen,
            "quality": frozen["quality"],
            "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
            "match_ready": True,
            "match_source": "original",
        }

    monkeypatch.setattr(face_attachment.face_mod, "finalize_identity", finalize)
    monkeypatch.setattr(
        face_attachment.reid_mod,
        "embed",
        lambda crop: np.asarray([1.0, 0.0], dtype=np.float32),
    )

    class FakeGallery:
        def identify_or_enroll(self, *args, **kwargs):
            assert args[1]["eligibility"] == "direct"
            assert kwargs["quality_gate"] is face_attachment.face_gallery_quality_ok
            return {
                "subject_id": 3,
                "score": 0.9,
                "decision": "new",
                "enrolled": True,
                "quality_ok": True,
            }

    monkeypatch.setattr(face_attachment.gallery_mod, "reset_gallery", lambda session_id: True)
    monkeypatch.setattr(
        face_attachment.gallery_mod,
        "with_gallery_locked",
        lambda session_id, dim, callback: callback(FakeGallery()),
    )

    face_attachment.attach_faces(
        frames,
        tracks,
        identities,
        "test",
        body_embeddings={1: np.asarray([1.0, 0.0], dtype=np.float32)},
    )

    assert tracks[1]["body_best"]["frame_index"] == 0
    assert tracks[1]["face_best"]["frame_index"] == 1
    assert identities[1]["face"]["evidence"]["frame_index"] == 1
    assert identities[1]["face"]["track_consistency_status"] == "passed"
    assert len(finalize_calls) == 1
    assert detect_calls
    assert all(call["with_identity"] is False for call in detect_calls)
    assert all(call["enhance_blurry"] is False for call in detect_calls)


def test_attach_faces_blocks_failed_cross_frame_track_provenance(
    monkeypatch,
    runtime_dir: Path,
) -> None:
    frame0 = write_image(runtime_dir / "body.jpg", (30, 30, 30))
    frame1 = write_image(runtime_dir / "face.jpg", (220, 220, 220))
    frames = [
        Frame(frame_id="0", timestamp="00:00:00", local_path=str(frame0)),
        Frame(frame_id="1", timestamp="00:00:01", local_path=str(frame1)),
    ]
    tracks = {
        1: {
            "best_idx": 0,
            "best_box": [10, 0, 90, 90],
            "body_best": {
                "track_id": 1,
                "frame_index": 0,
                "timestamp": "00:00:00",
                "person_bbox": [10, 0, 90, 90],
            },
            "face_candidates": [
                {
                    "track_id": 1,
                    "frame_index": 1,
                    "timestamp": "00:00:01",
                    "person_bbox": [10, 0, 90, 90],
                    "proxy_score": 10,
                }
            ],
            "boxes": {0: [10, 0, 90, 90], 1: [10, 0, 90, 90]},
        }
    }
    identities = {1: {"track_id": 1}}
    monkeypatch.setattr(
        face_attachment.face_mod,
        "detect",
        lambda *args, **kwargs: [
            {
                "bbox": [35, 10, 65, 40],
                "kps": [[42, 20], [58, 20], [50, 28], [44, 36], [56, 36]],
                "det_score": 0.95,
                "quality": {
                    "category": "clear",
                    "eligibility": "direct",
                    "quality": 0.9,
                    "can_match": True,
                    "can_superres": False,
                    "can_enroll": True,
                },
            }
        ],
    )
    monkeypatch.setattr(
        face_attachment.face_mod,
        "finalize_identity",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("failed provenance must not reach identity")
        ),
    )
    monkeypatch.setattr(
        face_attachment.gallery_mod,
        "with_gallery_locked",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("failed provenance must not query gallery")
        ),
    )
    monkeypatch.setattr(
        face_attachment.reid_mod,
        "embed",
        lambda crop: np.asarray([0.0, 1.0], dtype=np.float32),
    )

    face_attachment.attach_faces(
        frames,
        tracks,
        identities,
        "test",
        body_embeddings={1: np.asarray([1.0, 0.0], dtype=np.float32)},
    )

    record = identities[1]["face"]
    assert record["observed"] is True
    assert record["match_ready"] is False
    assert record["can_match"] is False
    assert record["can_enroll"] is False
    assert record["face_subject_id"] is None
    assert record["track_consistency_status"] == "failed"
