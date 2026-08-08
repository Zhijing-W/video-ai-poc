from __future__ import annotations

import pytest

from app import gait
from app.core.config import settings


def test_missing_pose_weight_has_english_provisioning_diagnostic(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setitem(gait._state, "pose", None)

    with settings.override(pose_model=str(tmp_path / "yolov8n-pose.pt")):
        with pytest.raises(FileNotFoundError) as error:
            gait._pose_model()

    message = str(error.value)
    assert "Pose weight not found:" in message
    assert "download_models.py --include-optional-yolo" in message


def test_available_requires_pose_and_segmentation_assets(
    tmp_path,
    monkeypatch,
) -> None:
    pose = tmp_path / "yolov8n-pose.pt"
    segmentation = tmp_path / "yolov8m-seg.pt"
    pose.write_bytes(b"pose")
    monkeypatch.setattr(gait, "_ensure", lambda: True)
    monkeypatch.setitem(gait._state, "asset_error", None)

    with settings.override(
        pose_model=str(pose),
        gait_seg_model=str(segmentation),
    ):
        assert gait.available() is False
        assert "Gait segmentation weight not found:" in gait.load_error()

        segmentation.write_bytes(b"seg")
        assert gait.available() is True
        assert gait.load_error() is None
