from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app import face
from app.identity.face import quality as face_quality


class _FakeDetector:
    def detect(self, bgr, max_num=0, metric="default"):
        boxes = np.asarray([[10.0, 10.0, 90.0, 90.0, 0.95]], dtype=np.float32)
        keypoints = np.asarray(
            [[[30.0, 35.0], [70.0, 35.0], [50.0, 48.0], [36.0, 72.0], [64.0, 72.0]]],
            dtype=np.float32,
        )
        return boxes, keypoints


class _FakeFaceAnalysis:
    def __init__(self, models=None) -> None:
        self.det_model = _FakeDetector()
        self.models = models or {}


class _FakeLandmark3D:
    taskname = "landmark_3d_68"

    def get(self, bgr, face_obj) -> None:
        face_obj.landmark_3d_68 = np.arange(68 * 3, dtype=np.float32).reshape(68, 3)


def _prepare_detect_test(monkeypatch) -> None:
    monkeypatch.setattr(
        face,
        "_state",
        {"backend": "insightface", "model": {"app": _FakeFaceAnalysis()}},
    )
    monkeypatch.setattr(face.settings, "face_3d_cue", False)
    monkeypatch.setattr(face.settings, "face_superres", "off")
    monkeypatch.setattr(face.settings, "face_rec_backend", "arcface")
    monkeypatch.setattr(
        face,
        "_align_face",
        lambda bgr, kps: np.zeros((112, 112, 3), dtype=np.uint8),
    )


def test_face_detect_skips_identity_model_when_quality_rejects_match(monkeypatch) -> None:
    _prepare_detect_test(monkeypatch)
    monkeypatch.setattr(
        face,
        "assess_quality",
        lambda *args, **kwargs: {
            "category": "poor",
            "can_match": False,
            "can_superres": False,
        },
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("质量拒绝后不应调用ArcFace")

    monkeypatch.setattr(face, "embed_aligned_face", fail_if_called)

    result = face.detect(np.zeros((120, 120, 3), dtype=np.uint8), enhance_blurry=False)

    assert len(result) == 1
    assert result[0]["quality"]["can_match"] is False
    assert "embedding" not in result[0]


def test_face_detect_runs_quality_before_identity_model(monkeypatch) -> None:
    _prepare_detect_test(monkeypatch)
    calls: list[str] = []

    def assess(*args, **kwargs):
        calls.append("quality")
        return {
            "category": "clear",
            "can_match": True,
            "can_superres": False,
        }

    def embed(aligned_bgr, backend):
        calls.append("identity")
        return np.ones(512, dtype=np.float32) / np.sqrt(512)

    monkeypatch.setattr(face, "assess_quality", assess)
    monkeypatch.setattr(face, "embed_aligned_face", embed)

    result = face.detect(np.zeros((120, 120, 3), dtype=np.uint8), enhance_blurry=False)

    assert calls == ["quality", "identity"]
    assert result[0]["embedding"].shape == (512,)
    assert result[0]["rec_backend"] == "arcface"


def test_face_detect_keeps_optional_3d_geometry_after_detection_split(monkeypatch) -> None:
    _prepare_detect_test(monkeypatch)
    monkeypatch.setattr(
        face,
        "_state",
        {
            "backend": "insightface",
            "model": {"app": _FakeFaceAnalysis({"landmark_3d_68": _FakeLandmark3D()})},
        },
    )
    monkeypatch.setattr(face.settings, "face_3d_cue", True)
    monkeypatch.setattr(
        face,
        "assess_quality",
        lambda *args, **kwargs: {
            "category": "poor",
            "can_match": False,
            "can_superres": False,
        },
    )

    result = face.detect(np.zeros((120, 120, 3), dtype=np.uint8), enhance_blurry=False)

    assert result[0]["geom3d"].shape == (15,)
    assert "geom3d_error" not in result[0]


def test_face_assess_quality_returns_expected_fields_with_mocked_fiqa(monkeypatch) -> None:
    monkeypatch.setattr(face_quality, "_deep_fiqa_score", lambda aligned_bgr: 0.25)
    monkeypatch.setattr(face_quality, "_blur_var", lambda bgr, bbox: 120.0)

    bgr = np.random.default_rng(0).integers(0, 255, size=(120, 120, 3), dtype=np.uint8)
    payload = {
        "bbox": [10, 10, 90, 90],
        "det_score": 0.98,
        "kps": [[30, 35], [70, 35], [50, 48], [36, 72], [64, 72]],
    }

    result = face.assess_quality(payload, bgr=bgr, aligned_bgr=np.zeros((112, 112, 3), dtype=np.uint8))

    for key in [
        "category",
        "quality_ok",
        "reason",
        "defects",
        "can_enroll",
        "can_match",
        "can_superres",
        "match_weight",
        "fiqa",
        "rule_quality",
    ]:
        assert key in result
    assert result["category"] == "poor"
    assert result["quality_ok"] is False
    assert result["reason"] == "low_fiqa"
    assert "low_fiqa" in result["defects"]
    assert result["fiqa"] == 0.25
    assert result["can_enroll"] is False


def test_face_keeps_superres_experiment_compatibility(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(face, "_ensure_superres_impl", lambda: sentinel)

    assert face._ensure_superres() is sentinel


def test_face_detect_can_freeze_quality_without_identity(monkeypatch) -> None:
    monkeypatch.setattr(face, "_ensure_backend", lambda: None)
    monkeypatch.setitem(face._state, "model", {"app": object()})
    monkeypatch.setattr(face, "_to_bgr", lambda image: np.zeros((32, 32, 3), dtype=np.uint8))
    monkeypatch.setattr(
        face,
        "_detect_face_candidates",
        lambda app, bgr: [{"bbox": [1, 1, 20, 20], "kps": [[4, 4]] * 5, "det_score": 0.9, "_kps_array": np.ones((5, 2))}],
    )
    monkeypatch.setattr(face, "_align_face", lambda bgr, kps: np.zeros((112, 112, 3), dtype=np.uint8))
    monkeypatch.setattr(
        face,
        "assess_quality",
        lambda *args, **kwargs: {"category": "clear", "can_match": True, "can_superres": False},
    )
    monkeypatch.setattr(
        face,
        "_attach_optional_geometry",
        lambda *args: (_ for _ in ()).throw(AssertionError("geometry model must not run")),
    )
    monkeypatch.setattr(
        face,
        "embed_aligned_face",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("identity model must not run")),
    )

    result = face.detect(
        np.zeros((32, 32, 3), dtype=np.uint8),
        with_identity=False,
        with_geometry=False,
        enhance_blurry=False,
    )

    assert result[0]["quality"]["category"] == "clear"
    assert "embedding" not in result[0]


def test_recoverable_face_requires_successful_superres_before_embedding(monkeypatch) -> None:
    _prepare_detect_test(monkeypatch)
    monkeypatch.setattr(
        face,
        "assess_quality",
        lambda *args, **kwargs: {
            "category": "poor",
            "eligibility": "recoverable",
            "can_enroll": False,
            "can_match": False,
            "can_superres": True,
            "fiqa": None,
        },
    )
    calls = []
    monkeypatch.setattr(
        face,
        "enhance",
        lambda image, aligned=False: Image.fromarray(
            np.full((112, 112, 3), 180, dtype=np.uint8)
        ),
    )
    monkeypatch.setattr(face, "_deep_fiqa_score", lambda aligned: None)
    monkeypatch.setattr(
        face,
        "embed_aligned_face",
        lambda aligned, backend: calls.append(backend) or np.ones(512, dtype=np.float32),
    )

    restored = face.detect(np.zeros((120, 120, 3), dtype=np.uint8), enhance_blurry=True)
    blocked = face.detect(np.zeros((120, 120, 3), dtype=np.uint8), enhance_blurry=False)

    assert restored[0]["match_ready"] is True
    assert restored[0]["match_source"] == "superres"
    assert restored[0]["quality"]["enhanced"] is True
    assert blocked[0]["match_ready"] is False
    assert blocked[0]["match_source"] == "none"
    assert len(calls) == 1


def _quality_payload(short_side: float) -> dict:
    return {
        "bbox": [10.0, 10.0, 10.0 + short_side, 10.0 + short_side * 1.2],
        "det_score": 0.95,
        "kps": [
            [10.2 * 1 + short_side * 0.25, 10 + short_side * 0.35],
            [10 + short_side * 0.75, 10 + short_side * 0.35],
            [10 + short_side * 0.50, 10 + short_side * 0.52],
            [10 + short_side * 0.32, 10 + short_side * 0.78],
            [10 + short_side * 0.68, 10 + short_side * 0.78],
        ],
    }


def test_tiny_six_by_eight_face_is_unusable(monkeypatch) -> None:
    monkeypatch.setattr(face_quality, "_blur_var", lambda *args: 100.0)
    monkeypatch.setattr(face_quality, "_deep_fiqa_score", lambda *args: None)

    result = face_quality.assess_quality(
        {
            "bbox": [10.0, 10.0, 15.9, 17.5],
            "det_score": 0.504,
            "kps": [[11, 12], [14, 12], [12.5, 13], [11.5, 15], [13.5, 15]],
        },
        bgr=np.zeros((32, 32, 3), dtype=np.uint8),
    )

    assert result["eligibility"] == "unusable"
    assert result["can_match"] is False
    assert result["can_superres"] is False


@pytest.mark.parametrize(
    ("short_side", "eligibility", "can_match", "can_superres"),
    [
        (19, "unusable", False, False),
        (20, "recoverable", False, True),
        (27, "recoverable", False, True),
        (28, "direct", True, False),
    ],
)
def test_face_size_state_boundaries(
    monkeypatch,
    short_side,
    eligibility,
    can_match,
    can_superres,
) -> None:
    monkeypatch.setattr(face_quality, "_blur_var", lambda *args: 100.0)
    monkeypatch.setattr(face_quality, "_deep_fiqa_score", lambda *args: None)

    result = face_quality.assess_quality(
        _quality_payload(short_side),
        bgr=np.zeros((128, 128, 3), dtype=np.uint8),
    )

    assert result["eligibility"] == eligibility
    assert result["can_match"] is can_match
    assert result["can_superres"] is can_superres


def test_face_gallery_gate_rejects_restored_and_accepts_direct_clear() -> None:
    assert face_quality.face_gallery_quality_ok(
        {
            "category": "clear",
            "eligibility": "direct",
            "can_enroll": True,
            "enhanced": False,
        }
    ) == (True, None)
    accepted, reason = face_quality.face_gallery_quality_ok(
        {
            "category": "clear",
            "eligibility": "recoverable",
            "can_enroll": False,
            "enhanced": True,
        }
    )
    assert accepted is False
    assert reason == "restored_face_not_enrollable"
