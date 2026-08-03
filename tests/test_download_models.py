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


def test_google_drive_folder_selects_exact_relative_path(monkeypatch, tmp_path):
    files = [
        SimpleNamespace(id="celeb", path=r"Celeb_light\eva02_l_bio_best.pth"),
        SimpleNamespace(id="ltcc", path=r"LTCC\eva02_l_bio_best.pth"),
        SimpleNamespace(id="prcc", path=r"PRCC\eva02_l_bio_best.pth"),
    ]
    monkeypatch.setitem(
        sys.modules,
        "gdown",
        SimpleNamespace(download_folder=lambda **kwargs: files),
    )
    selected = []
    monkeypatch.setattr(
        download_models,
        "_download_gdrive",
        lambda file_id, target, force=False: selected.append(file_id) or target,
    )

    download_models._download_gdrive_folder_file(
        "folder",
        "LTCC/eva02_l_bio_best.pth",
        tmp_path / "checkpoint.pth",
    )

    assert selected == ["ltcc"]


def test_siglip2_prepares_base_image_processor(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda *args, **kwargs: calls.append(
                ("snapshot", args, kwargs)
            ),
            hf_hub_download=lambda *args, **kwargs: calls.append(
                ("file", args, kwargs)
            ),
        ),
    )

    download_models.prepare_siglip2(tmp_path, force=False)

    file_call = next(call for call in calls if call[0] == "file")
    assert file_call[1] == (
        "google/siglip2-base-patch16-224",
    )
    assert file_call[2]["filename"] == "preprocessor_config.json"
    assert file_call[2]["revision"] == download_models.SIGLIP2_BASE_REVISION
