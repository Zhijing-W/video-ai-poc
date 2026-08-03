"""Prepare model assets in one shared directory outside the Git checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = (
    Path(os.getenv("MODEL_ROOT") or Path.home() / ".cache" / "event-monitor" / "models")
    .expanduser()
    .resolve()
)

CLIP_REID_REVISION = "eb1898b72c882875f478bebfc6d41644eece0a5d"
DIFFER_REVISION = "67acb5d3658d103b4412c8c99f93ee8f085802fe"
SIGLIP2_REVISION = "196e5d6"
SIGLIP2_BASE_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
INSIGHTFACE_REQUIRED_FILES = (
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
)

URLS = {
    "gfpgan": (
        "https://github.com/TencentARC/GFPGAN/releases/download/"
        "v1.3.0/GFPGANv1.3.pth"
    ),
    "codeformer": (
        "https://github.com/sczhou/CodeFormer/releases/download/"
        "v0.1.0/codeformer.pth"
    ),
    "realesrgan": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.2.1/RealESRGAN_x2plus.pth"
    ),
    "resnet50": (
        "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
    ),
    "insightface": (
        "https://github.com/deepinsight/insightface/releases/download/"
        "v0.7/buffalo_l.zip"
    ),
    "clip_base": (
        "https://openaipublic.azureedge.net/clip/models/"
        "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/"
        "ViT-B-16.pt"
    ),
}

GDRIVE_IDS = {
    "osnet": "1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal",
    "clipreid": "1BVaZo93kOksYLjFNH3Gf7JxIbPlWSkcO",
    "differ_folder": "1RzAhSeSOgL2u8130mFAMdHI2k4sX9dQD",
}


def _print_ready(path: Path) -> Path:
    print(f"[ready] {path}")
    return path


def _download_url(
    url: str,
    target: Path,
    *,
    force: bool = False,
    expected_sha256: str | None = None,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and not force:
        return _print_ready(target)
    partial = target.with_name(f"{target.name}.part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "event-monitor-model-preparer/1.0"},
    )
    print(f"[download] {url}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if expected_sha256:
            digest = hashlib.sha256(partial.read_bytes()).hexdigest()
            if digest.lower() != expected_sha256.lower():
                raise RuntimeError(
                    f"SHA256 mismatch for {target.name}: {digest}"
                )
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return _print_ready(target)


def _download_gdrive(file_id: str, target: Path, *, force: bool = False) -> Path:
    if target.is_file() and not force:
        return _print_ready(target)
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Google Drive下载需要先安装gdown") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    print(f"[download] Google Drive file {file_id}")
    result = gdown.download(
        id=file_id,
        output=str(target),
        quiet=False,
        resume=True,
    )
    if result is None or not target.is_file():
        raise RuntimeError(f"Google Drive下载失败：{file_id}")
    return _print_ready(target)


def _download_gdrive_folder_file(
    folder_id: str,
    relative_path: str,
    target: Path,
    *,
    force: bool = False,
) -> Path:
    if target.is_file() and not force:
        return _print_ready(target)
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Google Drive下载需要先安装gdown") from exc
    print(f"[inspect] Google Drive folder {folder_id}")
    files = gdown.download_folder(
        id=folder_id,
        skip_download=True,
        remaining_ok=True,
        quiet=False,
    )
    expected = relative_path.replace("\\", "/").strip("/").casefold()
    matches = [
        item
        for item in files or []
        if str(item.path).replace("\\", "/").strip("/").casefold() == expected
    ]
    if len(matches) != 1:
        available = ", ".join(str(item.path) for item in files or [])
        raise RuntimeError(
            f"官方目录中应有且仅有一个{relative_path}；当前匹配{len(matches)}个。"
            f"目录内容：{available}"
        )
    return _download_gdrive(matches[0].id, target, force=force)


def _extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            resolved = (destination / member.filename).resolve()
            if root not in resolved.parents and resolved != root:
                raise RuntimeError(f"不安全的ZIP路径：{member.filename}")
        source.extractall(destination)


def _prepare_source_archive(
    owner: str,
    repository: str,
    revision: str,
    target: Path,
    *,
    force: bool = False,
) -> Path:
    marker = target / ".source-revision"
    if target.is_dir() and marker.is_file() and not force:
        if marker.read_text(encoding="utf-8").strip() == revision:
            return _print_ready(target)
    if target.exists():
        shutil.rmtree(target)
    url = (
        f"https://github.com/{owner}/{repository}/archive/{revision}.zip"
    )
    with tempfile.TemporaryDirectory(prefix="event-monitor-model-") as temp:
        temp_root = Path(temp)
        archive = _download_url(url, temp_root / "source.zip", force=True)
        extracted = temp_root / "extracted"
        _extract_zip(archive, extracted)
        roots = [path for path in extracted.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"无法识别{repository}源码归档根目录")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(roots[0]), target)
    marker.write_text(revision + "\n", encoding="utf-8")
    return _print_ready(target)


def prepare_yolo(model_root: Path, *, include_optional: bool, force: bool) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("YOLO下载需要先安装ultralytics") from exc

    names = ["yolov8m.pt"]
    if include_optional:
        names.extend(("yolov8n-pose.pt", "yolov8m-seg.pt"))
    target_dir = model_root / "detection" / "yolo"
    for name in names:
        target = target_dir / name
        if target.is_file() and not force:
            _print_ready(target)
            continue
        with tempfile.TemporaryDirectory(prefix="event-monitor-yolo-") as temp:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp)
                model = YOLO(name)
                source = Path(model.ckpt_path).resolve()
            finally:
                os.chdir(previous_cwd)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), target)
        _print_ready(target)


def prepare_superres(model_root: Path, *, force: bool) -> None:
    _download_url(
        URLS["gfpgan"],
        model_root / "superres" / "gfpgan" / "GFPGANv1.3.pth",
        force=force,
    )
    _download_url(
        URLS["codeformer"],
        model_root / "superres" / "codeformer" / "codeformer-v0.1.0.pth",
        force=force,
    )
    _download_url(
        URLS["realesrgan"],
        model_root
        / "superres"
        / "realesrgan"
        / "RealESRGAN_x2plus-v0.2.1.pth",
        force=force,
    )


def prepare_face(model_root: Path, *, force: bool) -> None:
    root = model_root / "face" / "insightface"
    archive = _download_url(
        URLS["insightface"],
        root / "buffalo_l.zip",
        force=force,
    )
    target = root / "models" / "buffalo_l"
    if (
        target.is_dir()
        and not force
        and all((target / name).is_file() for name in INSIGHTFACE_REQUIRED_FILES)
    ):
        _print_ready(target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="event-monitor-insightface-",
        dir=target.parent,
    ) as temp:
        extracted = Path(temp) / "buffalo_l"
        _extract_zip(archive, extracted)
        missing = [
            name
            for name in INSIGHTFACE_REQUIRED_FILES
            if not (extracted / name).is_file()
        ]
        if missing:
            raise RuntimeError(
                "InsightFace归档不完整，缺少：" + ", ".join(missing)
            )
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(extracted), target)
    _print_ready(target)


def prepare_builtin_reid(model_root: Path, *, force: bool) -> None:
    _download_url(
        URLS["resnet50"],
        model_root / "reid" / "resnet50" / "resnet50-11ad3fa6.pth",
        force=force,
    )
    _download_gdrive(
        GDRIVE_IDS["osnet"],
        model_root / "reid" / "osnet" / "osnet_ain_x1_0_msmt17.pt",
        force=force,
    )


def prepare_clipreid(model_root: Path, *, force: bool) -> None:
    root = model_root / "reid" / "clipreid"
    _prepare_source_archive(
        "Syliz517",
        "CLIP-ReID",
        CLIP_REID_REVISION,
        root / "source",
        force=force,
    )
    _download_url(
        URLS["clip_base"],
        root / "ViT-B-16.pt",
        force=force,
        expected_sha256=(
            "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
        ),
    )
    _download_gdrive(
        GDRIVE_IDS["clipreid"],
        root / "ViT-B-16_msmt17_60.pth",
        force=force,
    )


def prepare_siglip2(model_root: Path, *, force: bool) -> None:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError("SigLIP2下载需要先安装huggingface-hub") from exc
    target = model_root / "reid" / "siglip2" / "model"
    print(
        "[download] Hugging Face "
        "MarketaJu/siglip2-person-description-reid@"
        f"{SIGLIP2_REVISION}"
    )
    snapshot_download(
        "MarketaJu/siglip2-person-description-reid",
        revision=SIGLIP2_REVISION,
        local_dir=target,
        force_download=force,
        max_workers=4,
    )
    hf_hub_download(
        "google/siglip2-base-patch16-224",
        filename="preprocessor_config.json",
        revision=SIGLIP2_BASE_REVISION,
        local_dir=target,
        force_download=force,
    )
    _print_ready(target)


def prepare_differ(model_root: Path, *, force: bool) -> None:
    root = model_root / "reid" / "differ"
    _prepare_source_archive(
        "xliangp",
        "DIFFER",
        DIFFER_REVISION,
        root / "source",
        force=force,
    )
    _download_gdrive_folder_file(
        GDRIVE_IDS["differ_folder"],
        "LTCC/eva02_l_bio_best.pth",
        root / "eva02_l_bio_best.pth",
        force=force,
    )
    _download_gdrive_folder_file(
        GDRIVE_IDS["differ_folder"],
        "LTCC/config.yml",
        root / "config.yml",
        force=force,
    )


def write_inventory(model_root: Path) -> None:
    files = []
    for path in sorted(model_root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        files.append(
            {
                "path": path.relative_to(model_root).as_posix(),
                "size_bytes": path.stat().st_size,
            }
        )
    destination = model_root / "inventory.json"
    destination.write_text(
        json.dumps({"model_root": str(model_root), "files": files}, indent=2),
        encoding="utf-8",
    )
    _print_ready(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--reid", action="store_true")
    parser.add_argument("--superres", action="store_true")
    parser.add_argument("--face", action="store_true")
    parser.add_argument("--include-optional-yolo", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    model_root = args.model_root.expanduser().resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    print(f"MODEL_ROOT={model_root}")

    selected_profile = args.all or args.reid or args.superres or args.face
    tasks = []
    if args.all or not selected_profile or args.include_optional_yolo:
        tasks.append(
            (
                "YOLO",
                lambda: prepare_yolo(
                    model_root,
                    include_optional=args.include_optional_yolo or args.all,
                    force=args.force,
                ),
            )
        )
    if args.all or args.superres:
        tasks.append(
            (
                "super-resolution",
                lambda: prepare_superres(model_root, force=args.force),
            )
        )
    if args.all or args.face:
        tasks.append(
            ("InsightFace", lambda: prepare_face(model_root, force=args.force))
        )
    if args.all or args.reid:
        tasks.extend(
            (
                (
                    "built-in ReID",
                    lambda: prepare_builtin_reid(model_root, force=args.force),
                ),
                (
                    "CLIP-ReID",
                    lambda: prepare_clipreid(model_root, force=args.force),
                ),
                (
                    "SigLIP2",
                    lambda: prepare_siglip2(model_root, force=args.force),
                ),
                (
                    "DIFFER",
                    lambda: prepare_differ(model_root, force=args.force),
                ),
            )
        )

    failures = []
    for name, task in tasks:
        try:
            task()
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[failed] {name}: {type(exc).__name__}: {exc}")
    write_inventory(model_root)
    if failures:
        details = "; ".join(f"{name}: {error}" for name, error in failures)
        raise RuntimeError(f"部分模型准备失败：{details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
