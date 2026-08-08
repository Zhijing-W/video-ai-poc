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
