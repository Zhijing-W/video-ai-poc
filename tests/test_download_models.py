from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

from scripts import download_models


def test_prepare_yolo_downloads_outside_checkout(monkeypatch, tmp_path):
    work_dir = tmp_path / "work"
    model_root = tmp_path / "models"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    class FakeYolo:
        def __init__(self, name):
            Path(name).write_bytes(name.encode("ascii"))
            self.ckpt_path = name

    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(YOLO=FakeYolo),
    )

    download_models.prepare_yolo(
        model_root,
        include_optional=True,
        force=False,
    )

    expected = {"yolov8m.pt", "yolov8n-pose.pt", "yolov8m-seg.pt"}
    assert {path.name for path in (model_root / "detection" / "yolo").iterdir()} == expected
    assert not any(work_dir.iterdir())


def test_prepare_face_repairs_incomplete_directory(monkeypatch, tmp_path):
    model_root = tmp_path / "models"
    target = model_root / "face" / "insightface" / "models" / "buffalo_l"
    target.mkdir(parents=True)
    (target / "partial.onnx").write_bytes(b"partial")
    archive = tmp_path / "buffalo_l.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for name in download_models.INSIGHTFACE_REQUIRED_FILES:
            output.writestr(name, b"model")

    monkeypatch.setattr(
        download_models,
        "_download_url",
        lambda *args, **kwargs: archive,
    )

    download_models.prepare_face(model_root, force=False)

    assert not (target / "partial.onnx").exists()
    assert all(
        (target / name).read_bytes() == b"model"
        for name in download_models.INSIGHTFACE_REQUIRED_FILES
    )
