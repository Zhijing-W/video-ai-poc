from __future__ import annotations

import numpy as np

from app.identity.resolution import (
    group_people,
    merge_tracks_cross_route,
    split_subject_time_conflicts,
    stitch_orphans,
)
from app.keyframe import FrameMeta
from app.pipeline.windowing import split_windows


def test_split_windows_bridges_short_gaps_and_flushes_max_length() -> None:
    metas = [
        FrameMeta(index=0, active_tracks=[1]),
        FrameMeta(index=1, active_tracks=[1]),
        FrameMeta(index=2, active_tracks=[]),
        FrameMeta(index=3, active_tracks=[1]),
        FrameMeta(index=4, active_tracks=[2]),
        FrameMeta(index=5, active_tracks=[]),
        FrameMeta(index=6, active_tracks=[]),
        FrameMeta(index=7, active_tracks=[3]),
        FrameMeta(index=8, active_tracks=[3]),
        FrameMeta(index=9, active_tracks=[3]),
    ]

    assert split_windows(metas, quiet_frames=2, max_window_frames=4) == [
        [0, 1, 2, 3],
        [4, 5],
        [7, 8, 9],
    ]


def test_stitch_orphans_reuses_subject_then_falls_back_to_local_subject(monkeypatch) -> None:
    monkeypatch.setattr("app.identity.resolution.settings.event_local_stitch_thresh", 0.9)

    tracks = {
        1: {"first": 0, "last": 2},
        2: {"first": 3, "last": 5},
        3: {"first": 6, "last": 8},
    }
    identities = {
        1: {"subject_id": 10, "decision": "hit", "score": 0.95},
        2: {"subject_id": None, "decision": None, "score": None, "quality_ok": True},
        3: {"subject_id": None, "decision": None, "score": None, "quality_ok": False},
    }
    track_emb = {
        1: np.asarray([1.0, 0.0], dtype=np.float32),
        2: np.asarray([0.99, 0.01], dtype=np.float32),
        3: np.asarray([-1.0, 0.0], dtype=np.float32),
    }

    stitch_orphans(tracks, identities, track_emb, thresh=0.8)

    assert identities[2]["subject_id"] == 10
    assert identities[2]["decision"] == "stitched"
    assert identities[2]["reused"] is True
    assert identities[3]["local_subject"] is True
    assert identities[3]["decision"] == "local"
    assert identities[3]["subject_id"] not in {None, 10}


def test_split_subject_time_conflicts_breaks_overlapping_tracks() -> None:
    tracks = {
        1: {"first": 0, "last": 5},
        2: {"first": 3, "last": 7},
        3: {"first": 8, "last": 10},
    }
    identities = {
        1: {"subject_id": 1, "decision": "hit", "reused": True},
        2: {"subject_id": 1, "decision": "hit", "reused": True},
        3: {"subject_id": 1, "decision": "hit", "reused": True},
    }

    split_subject_time_conflicts(tracks, identities)

    assert identities[1]["subject_id"] == 1
    assert identities[3]["subject_id"] == 1
    assert identities[2]["subject_id"] == 2
    assert identities[2]["decision"] == "conflict_split"
    assert identities[2]["subject_conflict_split"] is True


def test_merge_tracks_cross_route_records_agreement_across_body_face_gait() -> None:
    identities = {
        1: {
            "subject_id": 7,
            "decision": "hit",
            "route_subject": {"route": "body", "local_subject_id": 7},
            "face": {
                "matched": True,
                "match_ready": True,
                "quality": "clear",
                "eligibility": "direct",
                "track_consistency_status": "same_frame",
                "route_subject": {"route": "face", "local_subject_id": 100},
            },
        },
        2: {
            "subject_id": None,
            "decision": None,
            "face": {
                "matched": True,
                "match_ready": True,
                "quality": "clear",
                "eligibility": "direct",
                "track_consistency_status": "passed",
                "route_subject": {"route": "face", "local_subject_id": 100},
            },
        },
        3: {
            "subject_id": 7,
            "decision": "hit",
            "route_subject": {"route": "body", "local_subject_id": 7},
            "gait": {
                "decision": "hit",
                "route_subject": {"route": "gait", "local_subject_id": 55},
            },
        },
        4: {
            "subject_id": None,
            "decision": None,
            "gait": {
                "decision": "hit",
                "route_subject": {"route": "gait", "local_subject_id": 55},
            },
        },
    }

    merge_tracks_cross_route(identities)

    for tid in identities:
        assert identities[tid]["subject_id"] == 7
        assert identities[tid]["merge_agree"] == 3
        assert identities[tid]["merge_routes"] == ["body", "face", "gait"]
        assert set(identities[tid]["route_subject_ids"]) <= {"body", "face", "gait"}
    assert identities[2]["decision"] == "merged"
    assert identities[4]["decision"] == "merged"


def test_group_people_merges_tracks_by_subject_and_keeps_best_representative() -> None:
    tracks = {
        1: {
            "boxes": {0: [0, 0, 10, 10], 1: [1, 1, 11, 11]},
            "centers": [(0, (0.1, 0.1)), (1, (0.2, 0.2))],
            "best_box": [1, 1, 11, 11],
        },
        2: {
            "boxes": {2: [2, 2, 12, 12], 3: [3, 3, 13, 13]},
            "centers": [(2, (0.3, 0.3)), (3, (0.4, 0.4))],
            "best_box": [3, 3, 13, 13],
        },
    }
    identities = {
        1: {
            "subject_id": 9,
            "score": 0.82,
            "face": {"quality": "clear"},
            "fused": {"confidence": 0.9},
            "merge_routes": ["body"],
        },
        2: {
            "subject_id": 9,
            "score": 0.61,
            "gait": {"score": 0.7},
            "fused": {"confidence": 0.6},
            "merge_routes": ["face", "gait"],
        },
    }

    people = group_people([1, 2], tracks, identities, [0, 1, 2, 3], 100, 100)

    assert len(people) == 1
    person = people[0]
    assert person["track_id"] == 1
    assert person["source_track_ids"] == [1, 2]
    assert person["subject_id"] == 9
    assert person["reid"] == {"score": 0.82}
    assert person["merge_routes"] == ["body", "face", "gait"]
    assert person["merge_agree"] == 3
    assert person["trajectory"] == [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]]
