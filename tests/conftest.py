from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def runtime_dir() -> Path:
    path = Path(__file__).resolve().parent / "_runtime"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    if path.exists():
        shutil.rmtree(path)


def write_image(path: Path, color: tuple[int, int, int] = (64, 96, 160)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 96), color).save(path)
    return path
