from __future__ import annotations

import numpy as np

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
