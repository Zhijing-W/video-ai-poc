from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .common import (
    EXPERIMENT_DIR,
    GALLERY_ROLES,
    PACKAGE_DIR,
    QUERY_ROLES,
    REPO_ROOT,
    SCHEMA_VERSION,
    atomic_write_json,
    file_sha256,
    l2norm,
    parse_bool,
    resolve_source,
    stable_hash,
    validate_manifest_pair,
)


DEFAULT_BACKENDS = ("osnet", "clipreid", "siglip2", "differ")
DEFAULT_FMR_TARGETS = (0.01, 0.05, 0.10)
BACKEND_SOURCES = {
    "osnet": "https://github.com/mikel-brostrom/boxmot",
    "clipreid": "https://github.com/Syliz517/CLIP-ReID",
    "siglip2": (
        "https://huggingface.co/MarketaJu/"
        "siglip2-person-description-reid"
    ),
    "differ": "https://github.com/xliangp/DIFFER",
}
BACKEND_ASSET_SETTINGS = {
    "osnet": ("reid_osnet_weights",),
    "clipreid": (
        "reid_clipreid_root",
        "reid_clipreid_config",
        "reid_clipreid_weights",
        "reid_clipreid_base_weights",
    ),
    "siglip2": ("reid_siglip2_model",),
    "differ": (
        "reid_differ_root",
        "reid_differ_config",
        "reid_differ_weights",
    ),
}
BACKEND_PREPROCESSING = {
    "osnet": "BoxMOT OSNet backend preprocessing, BGR person crop",
    "clipreid": "RGB 256x128, mean/std 0.5",
    "siglip2": "Pinned AutoImageProcessor, RGB 224x224",
    "differ": "RGB 224x224, ImageNet mean/std, neutral camera",
}
WEIGHT_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".onnx"}
ASSET_IGNORED_DIRS = {".git", ".cache", ".locks", "__pycache__"}
ASSET_IGNORED_FILES = {".DS_Store", "Thumbs.db"}
ASSET_IGNORED_SUFFIXES = {".pyc", ".pyo"}


def _path_hint(path: Path, model_root: Path) -> str:
    try:
        return path.resolve().relative_to(model_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _describe_asset(path: Path, model_root: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"模型资产不存在：{path}")
    if path.is_file():
        descriptor = {
            "path_hint": _path_hint(path, model_root),
            "kind": "file",
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        descriptor["weight_files"] = (
            [
                {
                    "path": path.name,
                    "sha256": descriptor["sha256"],
                    "size_bytes": descriptor["size_bytes"],
                }
            ]
            if path.suffix.lower() in WEIGHT_SUFFIXES
            else []
        )
        return descriptor

    files = []
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(path)
        if any(part in ASSET_IGNORED_DIRS for part in relative.parts):
            continue
        if (
            item.name in ASSET_IGNORED_FILES
            or item.suffix.lower() in ASSET_IGNORED_SUFFIXES
        ):
            continue
        files.append(item)
    files.sort()
    if not files:
        raise ValueError(f"模型目录为空：{path}")
    digest_parts = []
    total_bytes = 0
    weight_files = []
    for item in files:
        relative = item.relative_to(path).as_posix()
        item_hash = file_sha256(item)
        size = item.stat().st_size
        total_bytes += size
        digest_parts.append((relative, item_hash, size))
        if item.suffix.lower() in WEIGHT_SUFFIXES:
            weight_files.append(
                {
                    "path": relative,
                    "sha256": item_hash,
                    "size_bytes": size,
                }
            )
    marker = path / ".source-revision"
    return {
        "path_hint": _path_hint(path, model_root),
        "kind": "directory",
        "sha256": stable_hash(digest_parts),
        "size_bytes": total_bytes,
        "file_count": len(files),
        "source_revision": (
            marker.read_text(encoding="utf-8").strip()
            if marker.is_file()
            else None
        ),
        "weight_files": weight_files,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state() -> dict[str, Any]:
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = commit_result.stdout.strip()
    if not commit:
        raise RuntimeError("无法读取当前Git commit")
    diff_result = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return {
        "commit": commit,
        "dirty": bool(status_result.stdout),
        "status_sha256": (
            hashlib.sha256(status_result.stdout).hexdigest()
            if status_result.stdout
            else None
        ),
        "tracked_diff_sha256": (
            hashlib.sha256(diff_result.stdout).hexdigest()
            if diff_result.stdout
            else None
        ),
    }


def _code_state() -> dict[str, Any]:
    paths = [
        REPO_ROOT / "app" / "body_reid.py",
        REPO_ROOT / "app" / "core" / "config.py",
        REPO_ROOT / "app" / "identity" / "embedding_gallery.py",
        *sorted(
            (
                REPO_ROOT / "app" / "identity" / "body_reid_backends"
            ).glob("*.py")
        ),
        *sorted(PACKAGE_DIR.glob("*.py")),
    ]
    records = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in paths
    ]
    return {
        "sha256": stable_hash(records),
        "files": records,
    }


def _runtime_provenance(torch) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in (
                "numpy",
                "Pillow",
                "torch",
                "torchvision",
                "transformers",
                "boxmot",
                "timm",
            )
        },
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if cuda_available
            else None
        ),
        "git": _git_state(),
    }


def collect_backend_provenance(
    backend: str,
    settings,
) -> dict[str, Any]:
    import torch

    model_root = Path(settings.model_root).expanduser()
    assets = {}
    for setting_name in BACKEND_ASSET_SETTINGS[backend]:
        assets[setting_name] = _describe_asset(
            Path(getattr(settings, setting_name)),
            model_root,
        )
    source_revision = None
    if backend == "siglip2":
        source_revision = settings.reid_siglip2_revision
    else:
        for descriptor in assets.values():
            if descriptor.get("source_revision"):
                source_revision = descriptor["source_revision"]
                break
    weight_entries = [
        item
        for descriptor in assets.values()
        for item in descriptor["weight_files"]
    ]
    provenance = {
        "backend": backend,
        "source_url": BACKEND_SOURCES[backend],
        "source_revision": source_revision,
        "preprocessing": BACKEND_PREPROCESSING[backend],
        "inference_precision": "float32",
        "assets": assets,
        "model_weight_bytes": sum(
            int(item["size_bytes"]) for item in weight_entries
        ),
        "experiment_code": _code_state(),
        "runtime": _runtime_provenance(torch),
    }
    provenance["provenance_hash"] = stable_hash(provenance)
    return provenance


def _comparison_environment(
    provenance: dict[str, Any],
    *,
    device_request: str,
    actual_device: str,
) -> dict[str, Any]:
    runtime = provenance["runtime"]
    identity = {
        "experiment_code_sha256": provenance["experiment_code"]["sha256"],
        "device_request": device_request,
        "actual_device": actual_device,
        "runtime": {
            "python": runtime["python"],
            "platform": runtime["platform"],
            "packages": runtime["packages"],
            "cuda_available": runtime["cuda_available"],
            "cuda_version": runtime["cuda_version"],
            "gpu": runtime["gpu"],
            "git": runtime["git"],
        },
    }
    return {"sha256": stable_hash(identity), **identity}


def verify_frozen_sources(
    data_root: Path,
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> None:
    annotation_root = (
        data_root.resolve()
        / "annotation"
        / "mevid-v1-annotation-data"
    )
    for name, expected_hash in protocol["dataset"]["annotation_sha256"].items():
        path = annotation_root / name
        if not path.is_file():
            raise FileNotFoundError(f"MEVID标注文件不存在：{path}")
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"MEVID标注文件已变化：{name}; "
                f"expected={expected_hash}, actual={actual_hash}"
            )
    for index, row in enumerate(rows, start=1):
        source = resolve_source(data_root, row)
        if not source.is_file():
            raise FileNotFoundError(
                f"冻结样本不存在：{row['sample_id']} -> {source}"
            )
        actual_hash = file_sha256(source)
        if actual_hash != row["source_sha256"]:
            raise ValueError(
                f"冻结样本已变化：{row['sample_id']}; "
                f"expected={row['source_sha256']}, actual={actual_hash}"
            )
        if index % 500 == 0:
            print(f"    source hash {index}/{len(rows)}", flush=True)


def _embeddable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["role"] in QUERY_ROLES
        or (
            row["role"] in GALLERY_ROLES
            and parse_bool(row["gallery_allowed"])
        )
    ]


def _resolve_device_request(device: str, torch) -> str:
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"未知实验设备：{device!r}")
    cuda_available = bool(torch.cuda.is_available())
    if device == "cuda" and not cuda_available:
        raise RuntimeError("--device cuda，但当前PyTorch不可用CUDA")
    return "cuda" if cuda_available and device != "cpu" else "cpu"


def _cuda_active(device: str, torch) -> bool:
    return device == "cuda" and bool(torch.cuda.is_available())


def _cuda_sync(device: str, torch) -> None:
    if _cuda_active(device, torch):
        torch.cuda.synchronize()


def _performance_summary(
    timings_ms: list[float],
    peak_cuda_bytes: int,
    model_load_seconds: float,
    cuda_memory_after_load_bytes: int,
    actual_device: str,
) -> dict[str, Any]:
    values = np.asarray(timings_ms, dtype=np.float64)
    total_seconds = float(values.sum() / 1000.0)
    return {
        "images": len(timings_ms),
        "latency_ms_mean": round(float(values.mean()), 6),
        "latency_ms_p50": round(float(np.percentile(values, 50)), 6),
        "latency_ms_p95": round(float(np.percentile(values, 95)), 6),
        "throughput_images_per_second": (
            round(len(timings_ms) / total_seconds, 6)
            if total_seconds > 0
            else None
        ),
        "model_load_seconds": round(model_load_seconds, 6),
        "actual_device": actual_device,
        "cuda_memory_after_load_bytes": int(cuda_memory_after_load_bytes),
        "peak_cuda_memory_bytes": int(peak_cuda_bytes),
    }


def extract_embeddings(
    *,
    rows: list[dict[str, Any]],
    data_root: Path,
    backend: str,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], int]:
    from app import body_reid
    from app.core.config import settings
    import torch

    actual_device = _resolve_device_request(device, torch)
    metadata = body_reid.backend_metadata()[backend]
    if metadata["requires_cuda"] and actual_device == "cpu":
        raise ValueError(f"{backend}要求CUDA，但本次实际设备是CPU")

    timings_ms = []
    vectors: dict[str, np.ndarray] = {}
    peak_cuda_bytes = 0
    cuda_memory_after_load_bytes = 0
    model_load_seconds = 0.0
    body_reid.reset_backend()
    try:
        with settings.override(reid_backend=backend, reid_device=device):
            if _cuda_active(actual_device, torch):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            load_started = time.perf_counter()
            active = body_reid.active_backend()
            if active != backend:
                raise RuntimeError(
                    f"后端静默回退：requested={backend}, active={active}"
                )
            dimension = body_reid.embed_dim()
            if _cuda_active(actual_device, torch):
                torch.cuda.synchronize()
                cuda_memory_after_load_bytes = int(
                    torch.cuda.memory_allocated()
                )
            model_load_seconds = time.perf_counter() - load_started

            first = rows[0]
            first_path = resolve_source(data_root, first)
            with Image.open(first_path) as opened:
                first_image = opened.convert("RGB")
                first_image.load()
            first_a = body_reid.embed(first_image)
            first_b = body_reid.embed(first_image)
            if not np.allclose(first_a, first_b, rtol=1e-4, atol=1e-5):
                max_delta = float(np.max(np.abs(first_a - first_b)))
                raise RuntimeError(
                    f"{backend}同图重复推理不稳定，max_delta={max_delta}"
                )

            for index, row in enumerate(rows, start=1):
                source = resolve_source(data_root, row)
                with Image.open(source) as opened:
                    image = opened.convert("RGB")
                    image.load()
                _cuda_sync(actual_device, torch)
                started = time.perf_counter()
                vector = body_reid.embed(image)
                _cuda_sync(actual_device, torch)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if vector.size != dimension:
                    raise RuntimeError(
                        f"{backend}维度漂移：expected={dimension}, actual={vector.size}"
                    )
                norm = float(np.linalg.norm(vector))
                if not np.isclose(norm, 1.0, rtol=1e-4, atol=1e-5):
                    raise RuntimeError(
                        f"{backend}未返回L2归一化向量：norm={norm}"
                    )
                vectors[row["sample_id"]] = vector.astype(
                    np.float32,
                    copy=False,
                )
                timings_ms.append(elapsed_ms)
                if index % 100 == 0 or index == len(rows):
                    print(
                        f"    {backend} {index}/{len(rows)}",
                        flush=True,
                    )
            if _cuda_active(actual_device, torch):
                peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
    finally:
        body_reid.reset_backend()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        vectors,
        _performance_summary(
            timings_ms,
            peak_cuda_bytes,
            model_load_seconds,
            cuda_memory_after_load_bytes,
            actual_device,
        ),
        dimension,
    )


def _cache_paths(
    cache_dir: Path,
    backend: str,
    cache_key: str,
) -> tuple[Path, Path]:
    stem = f"{backend}_{cache_key[:16]}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def _load_cache(
    npz_path: Path,
    metadata_path: Path,
    cache_key: str,
    sample_ids: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any], int] | None:
    existing = (npz_path.exists(), metadata_path.exists())
    if existing == (False, False):
        return None
    if existing != (True, True):
        raise RuntimeError(
            f"embedding cache不完整：{npz_path}, {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("cache_key") != cache_key:
        raise RuntimeError("embedding cache key不匹配")
    actual_npz_hash = file_sha256(npz_path)
    if metadata.get("npz_sha256") != actual_npz_hash:
        raise RuntimeError("embedding cache文件哈希不匹配")
    with np.load(npz_path, allow_pickle=False) as payload:
        cached_ids = [str(value) for value in payload["sample_ids"].tolist()]
        matrix = np.asarray(payload["vectors"], dtype=np.float32)
    if cached_ids != sample_ids:
        raise RuntimeError("embedding cache样本顺序不匹配")
    if matrix.ndim != 2 or matrix.shape[0] != len(cached_ids):
        raise RuntimeError("embedding cache矩阵形状非法")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("embedding cache包含NaN或无穷值")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
        raise RuntimeError("embedding cache包含未归一化向量")
    vectors = {
        sample_id: matrix[index]
        for index, sample_id in enumerate(cached_ids)
    }
    return vectors, metadata["performance"], int(matrix.shape[1])


def _save_cache(
    npz_path: Path,
    metadata_path: Path,
    *,
    cache_key: str,
    backend: str,
    protocol_id: str,
    manifest_sha256: str,
    sample_ids: list[str],
    vectors: dict[str, np.ndarray],
    performance: dict[str, Any],
) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.stack([vectors[sample_id] for sample_id in sample_ids])
    temporary = npz_path.with_name(
        f".{npz_path.name}.{os.getpid()}.tmp"
    )
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=np.asarray(sample_ids, dtype=np.str_),
            vectors=matrix.astype(np.float32, copy=False),
        )
    os.replace(temporary, npz_path)
    atomic_write_json(
        metadata_path,
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "body_reid_embedding_cache",
            "cache_key": cache_key,
            "protocol_id": protocol_id,
            "manifest_sha256": manifest_sha256,
            "backend": backend,
            "sample_ids_sha256": stable_hash(sample_ids),
            "performance": performance,
            "npz_sha256": file_sha256(npz_path),
        },
    )


def _track_vectors(
    rows: list[dict[str, Any]],
    vectors: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[tuple[dict, np.ndarray]]] = (
        defaultdict(list)
    )
    for row in rows:
        vector = vectors.get(row["sample_id"])
        if vector is None:
            continue
        key = (
            row["split"],
            row["role"],
            int(row["official_track_index"]),
        )
        grouped[key].append((row, vector))

    result = []
    for (split, role, track), items in sorted(grouped.items()):
        person_ids = {row["person_id"] for row, _ in items}
        cameras = {int(row["camera_id"]) for row, _ in items}
        outfits = {int(row["outfit_id"]) for row, _ in items}
        if len(person_ids) != 1 or len(cameras) != 1 or len(outfits) != 1:
            raise ValueError(f"轨迹元数据不一致：split={split}, track={track}")
        result.append(
            {
                "query_id": f"{split}:t{track:05d}",
                "split": split,
                "role": role,
                "track": track,
                "person_id": next(iter(person_ids)),
                "camera_id": next(iter(cameras)),
                "outfit_id": next(iter(outfits)),
                "frames": len(items),
                "vector": l2norm(
                    np.mean(
                        np.stack([vector for _, vector in items]),
                        axis=0,
                    )
                ),
            }
        )
    return result


def _templates(
    tracks: list[dict[str, Any]],
    role: str,
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for track in tracks:
        if track["role"] == role:
            grouped[track["person_id"]].append(track["vector"])
    return {
        person_id: l2norm(np.mean(np.stack(vectors), axis=0))
        for person_id, vectors in grouped.items()
    }


def _rank_queries(
    tracks: list[dict[str, Any]],
    roles: tuple[str, str],
    templates: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    genuine_role, imposter_role = roles
    if not templates:
        raise RuntimeError("没有可用Gallery身份模板")
    rows = []
    for track in tracks:
        if track["role"] not in roles:
            continue
        scores = sorted(
            (
                (person_id, float(template @ track["vector"]))
                for person_id, template in templates.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        prediction, confidence = scores[0]
        genuine = track["role"] == genuine_role
        rank = None
        if genuine:
            for index, (person_id, _) in enumerate(scores, start=1):
                if person_id == track["person_id"]:
                    rank = index
                    break
        rows.append(
            {
                "query_id": track["query_id"],
                "track": track["track"],
                "person_id": track["person_id"],
                "role": track["role"],
                "genuine": genuine,
                "camera_id": track["camera_id"],
                "outfit_id": track["outfit_id"],
                "frames": track["frames"],
                "prediction": prediction,
                "confidence": round(confidence, 8),
                "rank": rank,
                "rank1_correct": bool(genuine and rank == 1),
                "rank5_correct": bool(genuine and rank is not None and rank <= 5),
                "average_precision": (
                    round(1.0 / rank, 8)
                    if genuine and rank is not None
                    else None
                ),
            }
        )
    return rows


def _closed_set_summary(
    query_rows: list[dict[str, Any]],
    *,
    subset_ids: set[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    selected = [
        row
        for row in query_rows
        if row["genuine"]
        and (subset_ids is None or row["query_id"] in subset_ids)
    ]
    rank1 = sum(row["rank1_correct"] for row in selected)
    rank5 = sum(row["rank5_correct"] for row in selected)
    average_precision = [
        float(row["average_precision"] or 0.0) for row in selected
    ]
    summary = {
        "queries": len(selected),
        "rank1_correct": rank1,
        "rank1_rate": round(rank1 / len(selected), 6) if selected else None,
        "rank5_correct": rank5,
        "rank5_rate": round(rank5 / len(selected), 6) if selected else None,
        "mAP_identity_template": (
            round(float(np.mean(average_precision)), 6)
            if average_precision
            else None
        ),
        "missing_truth_template": sum(
            row["rank"] is None for row in selected
        ),
    }
    if thresholds:
        summary["tpir_at_calibrated_threshold"] = {
            key: (
                round(
                    sum(
                        row["rank1_correct"]
                        and row["confidence"] >= threshold
                        for row in selected
                    )
                    / len(selected),
                    6,
                )
                if selected
                else None
            )
            for key, threshold in thresholds.items()
        }
    return summary


def calibrate_thresholds(
    rows: list[dict[str, Any]],
    targets: tuple[float, ...],
) -> tuple[dict[str, float], dict[str, Any]]:
    genuine = [row for row in rows if row["genuine"]]
    imposters = [row for row in rows if not row["genuine"]]
    if not genuine or not imposters:
        raise RuntimeError("校准集必须同时包含库内与陌生Query")
    scores = [float(row["confidence"]) for row in rows]
    thresholds = [
        float(np.nextafter(max(scores), np.inf)),
        *sorted(set(scores), reverse=True),
    ]
    frozen: dict[str, float] = {}
    operating = {}
    for target in targets:
        candidates = []
        for threshold in thresholds:
            correct = sum(
                row["rank1_correct"] and row["confidence"] >= threshold
                for row in genuine
            )
            false_matches = sum(
                row["confidence"] >= threshold for row in imposters
            )
            tpir = correct / len(genuine)
            fmr = false_matches / len(imposters)
            if fmr <= target:
                candidates.append((tpir, fmr, threshold, correct, false_matches))
        if not candidates:
            raise RuntimeError(f"无法满足校准FMR目标：{target}")
        tpir, fmr, threshold, correct, false_matches = max(
            candidates,
            key=lambda item: (item[0], -item[1], -item[2]),
        )
        key = f"fmr_{target:g}"
        frozen[key] = float(threshold)
        operating[key] = {
            "target_fmr": target,
            "threshold": float(threshold),
            "tpir": round(float(tpir), 6),
            "fmr": round(float(fmr), 6),
            "correct": correct,
            "false_matches": false_matches,
        }
    return frozen, {
        "genuine_queries": len(genuine),
        "imposter_queries": len(imposters),
        "operating_points": operating,
    }


def _open_set_summary(
    rows: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    genuine = [row for row in rows if row["genuine"]]
    imposters = [row for row in rows if not row["genuine"]]
    operating = {}
    for key, threshold in thresholds.items():
        correct = sum(
            row["rank1_correct"] and row["confidence"] >= threshold
            for row in genuine
        )
        false_matches = sum(
            row["confidence"] >= threshold for row in imposters
        )
        operating[key] = {
            "threshold": float(threshold),
            "tpir": round(correct / len(genuine), 6) if genuine else None,
            "fmr": (
                round(false_matches / len(imposters), 6)
                if imposters
                else None
            ),
            "correct": correct,
            "false_matches": false_matches,
            "genuine_rejected": sum(
                row["confidence"] < threshold for row in genuine
            ),
            "imposters_rejected": sum(
                row["confidence"] < threshold for row in imposters
            ),
        }
    return {
        "genuine_queries": len(genuine),
        "imposter_queries": len(imposters),
        "operating_points": operating,
    }


def evaluate_protocol(
    rows: list[dict[str, Any]],
    vectors: dict[str, np.ndarray],
    protocol: dict[str, Any],
    targets: tuple[float, ...] = DEFAULT_FMR_TARGETS,
) -> dict[str, Any]:
    expected_embedded = {
        row["sample_id"] for row in _embeddable_rows(rows)
    }
    if set(vectors) != expected_embedded:
        missing = sorted(expected_embedded - set(vectors))
        extra = sorted(set(vectors) - expected_embedded)
        raise ValueError(
            f"embedding样本集合不一致：missing={missing[:3]}, extra={extra[:3]}"
        )
    tracks = _track_vectors(rows, vectors)
    calibration_templates = _templates(tracks, "calibration_gallery")
    calibration_rows = _rank_queries(
        tracks,
        ("calibration_genuine_query", "calibration_imposter_query"),
        calibration_templates,
    )
    thresholds, calibration = calibrate_thresholds(
        calibration_rows,
        targets,
    )

    test_templates = _templates(tracks, "enroll_gallery")
    test_rows = _rank_queries(
        tracks,
        ("genuine_query", "imposter_query"),
        test_templates,
    )
    p2_ids = {
        f"test:t{int(track):05d}"
        for track in protocol["test"]["p2_cross_outfit_query_tracks"]
    }
    p3_ids = {
        f"test:t{int(track):05d}"
        for track in protocol["test"][
            "p3_same_outfit_cross_camera_query_tracks"
        ]
    }
    return {
        "calibration": calibration,
        "thresholds": {key: float(value) for key, value in thresholds.items()},
        "gallery": {
            "test_template_identities": len(test_templates),
            "calibration_template_identities": len(calibration_templates),
            "test_template_pids": sorted(test_templates),
        },
        "protocols": {
            "P1_overall": {
                "closed_set": _closed_set_summary(
                    test_rows,
                    thresholds=thresholds,
                ),
                "open_set": _open_set_summary(test_rows, thresholds),
            },
            "P2_cross_outfit": {
                "definition": (
                    "Query outfit is absent from every accepted Gallery "
                    "track for its identity"
                ),
                "query_ids": sorted(p2_ids),
                "closed_set": _closed_set_summary(
                    test_rows,
                    subset_ids=p2_ids,
                    thresholds=thresholds,
                ),
            },
            "P3_same_outfit_cross_camera": {
                "definition": (
                    "Query outfit exists in accepted Gallery and its camera "
                    "is absent from same-outfit Gallery tracks"
                ),
                "query_ids": sorted(p3_ids),
                "closed_set": _closed_set_summary(
                    test_rows,
                    subset_ids=p3_ids,
                    thresholds=thresholds,
                ),
            },
        },
        "queries": test_rows,
    }


def _embedding_cache_key(
    *,
    protocol_id: str,
    manifest_sha256: str,
    backend: str,
    device_request: str,
    actual_device: str,
    provenance_hash: str,
    sample_ids: list[str],
) -> str:
    return stable_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": protocol_id,
            "manifest_sha256": manifest_sha256,
            "backend": backend,
            "device_request": device_request,
            "actual_device": actual_device,
            "provenance_hash": provenance_hash,
            "sample_ids": sample_ids,
        }
    )


def _evaluation_config(targets: tuple[float, ...]) -> dict[str, Any]:
    identity = {
        "fmr_targets": list(targets),
        "calibration_split": "MEVID official train",
        "acceptance_rule": "confidence >= threshold",
    }
    return {"sha256": stable_hash(identity), **identity}


def _load_or_extract(
    *,
    rows: list[dict[str, Any]],
    data_root: Path,
    backend: str,
    device: str,
    cache_dir: Path,
    protocol_id: str,
    manifest_sha256: str,
    provenance: dict[str, Any],
    actual_device: str,
    force: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any], int, str, bool]:
    selected = _embeddable_rows(rows)
    sample_ids = [row["sample_id"] for row in selected]
    cache_key = _embedding_cache_key(
        protocol_id=protocol_id,
        manifest_sha256=manifest_sha256,
        backend=backend,
        device_request=device,
        actual_device=actual_device,
        provenance_hash=provenance["provenance_hash"],
        sample_ids=sample_ids,
    )
    npz_path, metadata_path = _cache_paths(cache_dir, backend, cache_key)
    if not force:
        cached = _load_cache(
            npz_path,
            metadata_path,
            cache_key,
            sample_ids,
        )
        if cached is not None:
            vectors, performance, dimension = cached
            return vectors, performance, dimension, cache_key, True

    vectors, performance, dimension = extract_embeddings(
        rows=selected,
        data_root=data_root,
        backend=backend,
        device=device,
    )
    _save_cache(
        npz_path,
        metadata_path,
        cache_key=cache_key,
        backend=backend,
        protocol_id=protocol_id,
        manifest_sha256=manifest_sha256,
        sample_ids=sample_ids,
        vectors=vectors,
        performance=performance,
    )
    return vectors, performance, dimension, cache_key, False


def run_backend(
    *,
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    data_root: Path,
    backend: str,
    device: str,
    output_dir: Path,
    cache_dir: Path,
    targets: tuple[float, ...],
    force: bool,
) -> dict[str, Any]:
    from app.core.config import settings
    import torch

    actual_device = _resolve_device_request(device, torch)
    provenance = collect_backend_provenance(backend, settings)
    sample_ids = [row["sample_id"] for row in _embeddable_rows(rows)]
    manifest_sha256 = protocol["sample_manifest"]["sha256"]
    prospective_cache_key = _embedding_cache_key(
        protocol_id=protocol["protocol_id"],
        manifest_sha256=manifest_sha256,
        backend=backend,
        device_request=device,
        actual_device=actual_device,
        provenance_hash=provenance["provenance_hash"],
        sample_ids=sample_ids,
    )
    evaluation_config = _evaluation_config(targets)
    comparison_environment = _comparison_environment(
        provenance,
        device_request=device,
        actual_device=actual_device,
    )
    result_path = output_dir / f"{backend}.json"
    if result_path.is_file() and not force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("kind") == "body_reid_backend_result"
            and existing.get("protocol_id") == protocol["protocol_id"]
            and existing.get("manifest_sha256") == manifest_sha256
            and existing.get("cache_key") == prospective_cache_key
            and existing.get("evaluation_config") == evaluation_config
            and existing.get("comparison_environment")
            == comparison_environment
        ):
            print(f"[reuse] {result_path}")
            return existing
        raise FileExistsError(
            f"结果已存在但与本次运行不一致；使用--force覆盖：{result_path}"
        )

    vectors, performance, dimension, cache_key, cache_reused = (
        _load_or_extract(
            rows=rows,
            data_root=data_root,
            backend=backend,
            device=device,
            cache_dir=cache_dir,
            protocol_id=protocol["protocol_id"],
            manifest_sha256=manifest_sha256,
            provenance=provenance,
            actual_device=actual_device,
            force=force,
        )
    )
    if performance.get("actual_device") != actual_device:
        raise RuntimeError(
            "embedding cache实际设备不一致："
            f"expected={actual_device}, actual={performance.get('actual_device')}"
        )
    evaluation = evaluate_protocol(rows, vectors, protocol, targets)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "body_reid_backend_result",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "manifest_sha256": manifest_sha256,
        "backend": backend,
        "active_backend": backend,
        "device_request": device,
        "actual_device": actual_device,
        "evaluation_config": evaluation_config,
        "comparison_environment": comparison_environment,
        "embedding_dimension": dimension,
        "provenance": provenance,
        "cache_key": cache_key,
        "cache_reused": cache_reused,
        "performance": performance,
        **evaluation,
    }
    atomic_write_json(result_path, result)
    print(f"[saved] {result_path}")
    return result


def _select_smoke_rows(
    rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    queries = [
        row
        for row in rows
        if row["role"] in {"genuine_query", "imposter_query"}
    ]
    selected = queries[:count]
    if len(selected) < count:
        raise ValueError(
            f"smoke需要{count}张test Query，实际只有{len(selected)}张"
        )
    return selected


def run_smoke(
    *,
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    data_root: Path,
    backends: tuple[str, ...],
    device: str,
    output_dir: Path,
    count: int,
    force: bool,
) -> dict[str, Any]:
    from app.core.config import settings
    import torch

    _resolve_device_request(device, torch)
    selected = _select_smoke_rows(rows, count)
    verify_frozen_sources(data_root, selected, protocol)
    expected_ids = [row["sample_id"] for row in selected]
    backend_results = {}
    for backend in backends:
        provenance = collect_backend_provenance(backend, settings)
        vectors, performance, dimension = extract_embeddings(
            rows=selected,
            data_root=data_root,
            backend=backend,
            device=device,
        )
        if list(vectors) != expected_ids:
            raise RuntimeError(f"{backend} smoke样本顺序不一致")
        backend_results[backend] = {
            "dimension": dimension,
            "actual_device": performance["actual_device"],
            "performance": performance,
            "provenance_hash": provenance["provenance_hash"],
            "norms": [
                round(float(np.linalg.norm(vectors[sample_id])), 8)
                for sample_id in expected_ids
            ],
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "body_reid_smoke_result",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "manifest_sha256": protocol["sample_manifest"]["sha256"],
        "sample_ids": expected_ids,
        "backends": backend_results,
    }
    output = output_dir / "smoke.json"
    if output.exists() and not force:
        raise FileExistsError(f"smoke结果已存在；使用--force覆盖：{output}")
    atomic_write_json(output, payload)
    print(f"[saved] {output}")
    return payload


def _parse_backends(value: str) -> tuple[str, ...]:
    from app import body_reid

    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise ValueError("--backends不能为空")
    canonical = []
    for item in requested:
        name = body_reid.validate_backend(item)
        if name == "auto":
            raise ValueError("正式实验禁止使用auto后端")
        if name not in BACKEND_SOURCES:
            raise ValueError(
                f"实验矩阵不支持后端{name}；可选{','.join(DEFAULT_BACKENDS)}"
            )
        if name not in canonical:
            canonical.append(name)
    return tuple(canonical)


def _parse_targets(value: str) -> tuple[float, ...]:
    targets = tuple(
        sorted(
            {
                float(item.strip())
                for item in value.split(",")
                if item.strip()
            }
        )
    )
    if not targets or any(item <= 0 or item >= 1 for item in targets):
        raise ValueError("--fmr-targets必须是0到1之间的逗号分隔值")
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="依次调用产品后端运行MEVID人形ReID矩阵"
    )
    parser.add_argument("--data", required=True, help="MEVID根目录")
    parser.add_argument(
        "--manifest",
        default=str(EXPERIMENT_DIR / "manifests" / "frozen_body_samples.csv"),
    )
    parser.add_argument(
        "--protocol",
        default=str(EXPERIMENT_DIR / "manifests" / "frozen_protocol.json"),
    )
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BACKENDS),
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--output-dir",
        default=str(EXPERIMENT_DIR / "results" / "runs"),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(EXPERIMENT_DIR / "results" / "cache"),
    )
    parser.add_argument(
        "--fmr-targets",
        default=",".join(str(item) for item in DEFAULT_FMR_TARGETS),
    )
    parser.add_argument(
        "--smoke-samples",
        type=int,
        default=0,
        help="大于0时只对同一批test Query做后端冒烟，不计算正式指标",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke_samples < 0:
        raise ValueError("--smoke-samples不能小于0")
    data_root = Path(args.data).resolve()
    manifest_path = Path(args.manifest).resolve()
    protocol_path = Path(args.protocol).resolve()
    rows, protocol = validate_manifest_pair(manifest_path, protocol_path)
    backends = _parse_backends(args.backends)
    targets = _parse_targets(args.fmr_targets)
    base_output = Path(args.output_dir).resolve()
    output_dir = base_output / protocol["protocol_id"]
    cache_dir = Path(args.cache_dir).resolve() / protocol["protocol_id"]
    if args.smoke_samples:
        run_smoke(
            rows=rows,
            protocol=protocol,
            data_root=data_root,
            backends=backends,
            device=args.device,
            output_dir=output_dir,
            count=args.smoke_samples,
            force=args.force,
        )
        return 0

    verify_frozen_sources(data_root, rows, protocol)
    for backend in backends:
        run_backend(
            rows=rows,
            protocol=protocol,
            data_root=data_root,
            backend=backend,
            device=args.device,
            output_dir=output_dir,
            cache_dir=cache_dir,
            targets=targets,
            force=args.force,
        )
    return 0
