from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PACKAGE_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
MEVID_COMMON_PATH = (
    REPO_ROOT
    / "experiment"
    / "face_blur_ablation"
    / "common"
    / "mevid_eval_common.py"
)

SAMPLE_FIELDS = (
    "split",
    "sample_id",
    "person_id",
    "role",
    "official_track_index",
    "camera_id",
    "outfit_id",
    "source_relpath",
    "source_sha256",
    "frame_position",
    "sample_position",
    "width",
    "height",
    "area",
    "blur_var",
    "aspect_ratio",
    "gallery_allowed",
    "gallery_rejection_reason",
)
GALLERY_ROLES = frozenset({"enroll_gallery", "calibration_gallery"})
QUERY_ROLES = frozenset(
    {
        "genuine_query",
        "imposter_query",
        "calibration_genuine_query",
        "calibration_imposter_query",
    }
)


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def l2norm(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)):
        raise ValueError("embedding包含NaN或无穷值")
    if norm <= 0:
        raise ValueError("embedding是零向量")
    return vector / norm


def load_mevid_common():
    module_name = "_event_monitor_mevid_eval_common"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, MEVID_COMMON_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载MEVID共享模块：{MEVID_COMMON_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def serialize_samples(rows: list[dict[str, Any]]) -> str:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=SAMPLE_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in SAMPLE_FIELDS})
    return output.getvalue()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SAMPLE_FIELDS:
            raise ValueError(
                "冻结样本CSV字段不匹配："
                f"expected={SAMPLE_FIELDS}, actual={tuple(reader.fieldnames or ())}"
            )
        rows = list(reader)
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("冻结样本CSV包含重复sample_id")
    return rows


def protocol_identity(payload: dict[str, Any]) -> dict[str, Any]:
    dataset = payload.get("dataset") or {}
    manifest = payload.get("sample_manifest") or {}
    test = payload.get("test") or {}
    calibration = payload.get("calibration") or {}
    return {
        "schema_version": payload.get("schema_version"),
        "dataset": {
            "name": dataset.get("name"),
            "annotation_sha256": dataset.get("annotation_sha256"),
        },
        "config": payload.get("config"),
        "quality_gate": payload.get("quality_gate"),
        "sample_manifest_sha256": manifest.get("sha256"),
        "test": {
            "candidate_pids": test.get("candidate_pids"),
            "enroll_pids": test.get("enroll_pids"),
            "imposter_pids": test.get("imposter_pids"),
            "p2_cross_outfit_query_tracks": test.get(
                "p2_cross_outfit_query_tracks"
            ),
            "p3_same_outfit_cross_camera_query_tracks": test.get(
                "p3_same_outfit_cross_camera_query_tracks"
            ),
        },
        "calibration": {
            "enroll_pids": calibration.get("enroll_pids"),
            "imposter_pids": calibration.get("imposter_pids"),
        },
    }


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"协议schema版本不支持：{payload.get('schema_version')}"
        )
    if payload.get("kind") != "mevid_body_reid_protocol":
        raise ValueError(f"不是人形ReID冻结协议：{path}")
    protocol_id = payload.get("protocol_id")
    if not protocol_id:
        raise ValueError("冻结协议缺少protocol_id")
    actual_id = stable_hash(protocol_identity(payload))
    if protocol_id != actual_id:
        raise ValueError(
            "冻结协议内容与protocol_id不一致："
            f"expected={protocol_id}, actual={actual_id}"
        )
    return payload


def validate_manifest_pair(
    manifest_path: Path,
    protocol_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol = load_protocol(protocol_path)
    actual_hash = file_sha256(manifest_path)
    expected = (protocol.get("sample_manifest") or {}).get("sha256")
    if actual_hash != expected:
        raise ValueError(
            "冻结样本CSV哈希与协议不一致："
            f"expected={expected}, actual={actual_hash}"
        )
    rows = load_samples(manifest_path)
    expected_rows = int((protocol.get("sample_manifest") or {}).get("rows", -1))
    if len(rows) != expected_rows:
        raise ValueError(
            f"冻结样本行数不一致：expected={expected_rows}, actual={len(rows)}"
        )
    return rows, protocol


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"非法布尔值：{value!r}")


def resolve_source(data_root: Path, row: dict[str, Any]) -> Path:
    root = data_root.resolve()
    source = (root / row["source_relpath"]).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"样本路径越出MEVID根目录：{row['source_relpath']}"
        ) from exc
    return source
