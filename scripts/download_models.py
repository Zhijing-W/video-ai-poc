"""Prepare repository-local Ultralytics model weights."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def prepare(name: str, force: bool = False) -> Path:
    target = MODELS_DIR / name
    if target.exists() and not force:
        print(f"[skip] {target}")
        return target

    model = YOLO(name)
    source = Path(model.ckpt_path)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    print(f"[ready] {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-pose", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prepare("yolov8m.pt", force=args.force)
    if args.include_pose:
        prepare("yolov8n-pose.pt", force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
