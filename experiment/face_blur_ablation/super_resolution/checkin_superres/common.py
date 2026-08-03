"""Stable schema, hashing, path, and manifest helpers."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PACKAGE_DIR = Path(__file__).resolve().parent
SUPERRES_DIR = PACKAGE_DIR.parent
EXPERIMENT_DIR = SUPERRES_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]

SCHEMA_VERSION = 3
ARMS = ("A_original", "B_all_superres", "C_gated_superres")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHECKIN_RE = re.compile(r"^(?P<pid>\d+)-.+-(?P<view>[fb])$", re.IGNORECASE)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def manifest_identity(
    coverage: dict,
    config_hash: str,
    model_provenance_hash: str,
    gallery: list[dict],
    queries: list[dict],
) -> dict:
    """Complete immutable protocol input used to detect stale/tampered manifests."""
    return {
        "coverage": coverage,
        "config_hash": config_hash,
        "model_provenance_hash": model_provenance_hash,
        "gallery": gallery,
        "queries": queries,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checkin_image(path: Path) -> tuple[str, str] | None:
    match = CHECKIN_RE.match(path.stem)
    if not match:
        return None
    return f"{int(match.group('pid')):04d}", match.group("view").upper()


def load_checkin_front_images(root: Path) -> dict[str, list[Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"actor check-in目录不存在：{root}")
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(
        item for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    ):
        parsed = parse_checkin_image(path)
        if parsed and parsed[1] == "F":
            grouped[parsed[0]].append(path)
    if not grouped:
        raise RuntimeError(f"actor check-in目录没有可解析的-F正脸照片：{root}")
    return dict(grouped)


def audit_prefix_coverage(
    checkin_pids: set[str],
    train_pids: set[str],
    test_pids: set[str],
    query_pids: set[str],
) -> dict:
    coverage = {
        "checkin_prefixes": sorted(checkin_pids),
        "train_pids": sorted(train_pids),
        "test_pids": sorted(test_pids),
        "query_pids": sorted(query_pids),
        "missing_train_pids": sorted(train_pids - checkin_pids),
        "missing_test_pids": sorted(test_pids - checkin_pids),
        "missing_query_pids": sorted(query_pids - checkin_pids),
        "extra_checkin_prefixes": sorted(checkin_pids - train_pids - test_pids),
    }
    coverage["counts"] = {
        key: len(value)
        for key, value in coverage.items()
        if isinstance(value, list)
    }
    if coverage["missing_query_pids"]:
        raise RuntimeError(
            "actor check-in前缀缺少官方Query PID映射："
            + ",".join(coverage["missing_query_pids"])
        )
    return coverage


def annotation_pid_set(data_root: Path, filename: str) -> set[str]:
    path = (
        data_root
        / "annotation"
        / "mevid-v1-annotation-data"
        / filename
    )
    if not path.is_file():
        raise FileNotFoundError(f"MEVID annotation不存在：{path}")
    return {
        f"{int(float(parts[2])):04d}"
        for line in path.read_text(encoding="utf-8").splitlines()
        if len(parts := line.split()) >= 3
    }


def sample_evenly_indexed(items: list[Path], count: int) -> list[tuple[int, Path]]:
    if count <= 0 or len(items) <= count:
        return list(enumerate(items))
    if count == 1:
        indices = [0]
    else:
        indices = [
            int(round(index * (len(items) - 1) / (count - 1)))
            for index in range(count)
        ]
    return [(index, items[index]) for index in dict.fromkeys(indices)]


def build_image_manifest_records(
    queries: list[dict],
    artifacts_by_sample: dict[str, dict],
) -> list[dict]:
    rows = []
    for query in queries:
        sample_id = query["sample_id"]
        artifact = artifacts_by_sample.get(sample_id) or {}
        aligned = bool(query.get("aligned_path"))
        rows.append(
            {
                "sample_id": sample_id,
                "pid": query["pid"],
                "track": query["track"],
                "eligibility": query.get("eligibility", "none"),
                "source_frame_index": query.get("face_best_frame_index"),
                "status": artifact.get("status")
                or ("missing_artifact" if aligned else "non_processed"),
                "reason": artifact.get("reason")
                or (None if aligned else query.get("face_status", "not_detected")),
                "comparison_path": artifact.get("comparison_path"),
                "original_aligned_path": artifact.get("original_aligned_path"),
                "superres_aligned_path": artifact.get("superres_aligned_path"),
            }
        )
    return rows


def _relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve(value: str | None, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _save_bgr(path: Path, bgr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(bgr)[:, :, ::-1]).save(path)
    return file_sha256(path)
