"""Fixed-manifest, multi-backend super-resolution evaluator."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from app.identity.face.quality import deep_fiqa_score, superres_quality_ok

from .common import SCHEMA_VERSION, canonical_json, file_sha256, stable_hash, _relative, _resolve
from .embeddings import _normalise, _pack_vectors, _unpack_vectors, _verify_manifest
from .metrics import (
    _score,
    _templates,
    paired_uncertainty,
    pid_cluster_bootstrap_rate,
    summarize_scores,
)
from .preparation import _model_provenance, _provenance_compatible
from .visualization import _median_rule_quality, render_matrix_figures

FROZEN_MANIFEST_ID = "83e51f3a5bf3879b5214557a9cbfe6df20e36f3cef57acb805c993c2a53b7921"
NORMALIZATION = {
    "name": "normalize112",
    "input_mode": "RGB",
    "input_dtype": "uint8",
    "target_size": [112, 112],
    "same_size": "byte_preserving",
    "resize": "Pillow.LANCZOS",
}
BACKEND_SPECS: tuple[dict[str, Any], ...] = (
    {
        "backend": "gfpgan",
        "label": "GFPGAN",
        "b_arm": "B1_all_gfpgan",
        "c_arm": "C1_gated_gfpgan",
        "fidelity": None,
        "transform": "product_gfpgan_aligned",
    },
    {
        "backend": "codeformer",
        "label": "CodeFormer w=1",
        "b_arm": "B2_all_codeformer_w1",
        "c_arm": "C2_gated_codeformer_w1",
        "fidelity": 1.0,
        "transform": "product_codeformer_aligned",
    },
    {
        "backend": "realesrgan_x2plus",
        "label": "RealESRGAN x2",
        "b_arm": "B3_all_realesrgan_x2plus",
        "c_arm": "C3_gated_realesrgan_x2plus",
        "fidelity": None,
        "transform": "raw_x2_no_face_enhancement",
    },
)
CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "backend": "resize512",
        "label": "R resize512",
        "b_arm": "R_resize512",
        "c_arm": None,
        "fidelity": None,
        "transform": "112_to_512_bilinear_then_normalize112",
    },
    {
        "backend": "resize_x2",
        "label": "R resize x2",
        "b_arm": "R_resize_x2",
        "c_arm": None,
        "fidelity": None,
        "transform": "112_to_224_bicubic_then_normalize112",
    },
)
EXPECTED_NATIVE_SIZES = {
    "gfpgan": (512, 512),
    "codeformer": (512, 512),
    "realesrgan_x2plus": (224, 224),
    "resize512": (512, 512),
    "resize_x2": (224, 224),
}
MAIN_ARMS = (
    "A_original",
    "C1_gated_gfpgan",
    "C2_gated_codeformer_w1",
    "C3_gated_realesrgan_x2plus",
)
ALL_ARMS = (
    "A_original",
    "P_off_original",
    *(spec["b_arm"] for spec in BACKEND_SPECS),
    *(spec["c_arm"] for spec in BACKEND_SPECS),
    *(spec["b_arm"] for spec in CONTROL_SPECS),
)


def normalize112(image: Image.Image) -> Image.Image:
    """Apply the one explicit pixel contract used by ArcFace and FIQA."""
    rgb = image.convert("RGB")
    if rgb.size == (112, 112):
        return Image.fromarray(np.asarray(rgb, dtype=np.uint8).copy(), mode="RGB")
    return rgb.resize((112, 112), Image.Resampling.LANCZOS)


def _load_frozen_rgb(path: Path) -> Image.Image:
    if path.suffix.lower() != ".png":
        raise ValueError(f"fixed aligned source must be PNG: {path}")
    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"fixed aligned source must be RGB uint8: {path}")
    return Image.fromarray(array.copy(), mode="RGB")


def _assert_normalized112(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"))
    if image.size != (112, 112) or array.shape != (112, 112, 3):
        raise AssertionError(f"embedding/FIQA input is not normalized112: {array.shape}")
    if array.dtype != np.uint8:
        raise AssertionError(f"embedding/FIQA input is not uint8: {array.dtype}")
    return array


def embed_normalized112(image: Image.Image, face_mod) -> np.ndarray | None:
    rgb = _assert_normalized112(image)
    return _normalise(
        face_mod.embed_aligned_face(rgb[:, :, ::-1].copy(), "arcface")
    )


def fiqa_normalized112(
    image: Image.Image,
    score_fn: Callable[[np.ndarray], float | None] = deep_fiqa_score,
) -> float | None:
    rgb = _assert_normalized112(image)
    value = score_fn(rgb[:, :, ::-1].copy())
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def derive_backend_arms(
    original: np.ndarray | None,
    restored: np.ndarray | None,
    *,
    eligibility: str,
    transform_succeeded: bool,
    b_arm: str,
    c_arm: str,
) -> dict[str, np.ndarray | None]:
    b_vector = restored if transform_succeeded and restored is not None else None
    if eligibility == "direct":
        c_vector = original
    elif eligibility == "recoverable":
        c_vector = b_vector
    else:
        c_vector = None
    return {b_arm: b_vector, c_arm: c_vector}


def p_off_original(
    original: np.ndarray | None,
    eligibility: str,
) -> np.ndarray | None:
    return original if eligibility in {"direct", "recoverable"} else None


def _package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _configured_weight(settings, backend: str) -> tuple[str | None, str | None]:
    attributes = {
        "gfpgan": "face_gfpgan_weights",
        "codeformer": "face_codeformer_weights",
        "realesrgan_x2plus": "face_realesrgan_x2plus_weights",
    }
    attribute = attributes.get(backend)
    value = str(getattr(settings, attribute, "") or "").strip() if attribute else ""
    candidates: list[Path] = []
    if value and not value.startswith(("http://", "https://")):
        candidates.append(Path(value).expanduser())
    elif backend == "gfpgan":
        name = Path(urlparse(value).path).name if value else "GFPGANv1.3.pth"
        candidates.append(Path.home() / ".cache" / "gfpgan" / name)
    elif backend == "codeformer":
        name = Path(urlparse(value).path).name if value else "codeformer.pth"
        if value:
            name = f"{hashlib.sha256(value.encode()).hexdigest()[:12]}-{name}"
        candidates.append(
            Path.home()
            / ".cache"
            / "event-monitor"
            / "superres"
            / "codeformer-v0.1.0"
            / name
        )
    elif backend == "realesrgan_x2plus":
        name = Path(urlparse(value).path).name if value else "RealESRGAN_x2plus.pth"
        if value:
            name = f"{hashlib.sha256(value.encode()).hexdigest()[:12]}-{name}"
        candidates.append(
            Path.home()
            / ".cache"
            / "event-monitor"
            / "superres"
            / "realesrgan-x2plus-v0.2.1"
            / name
        )
    resolved = next((path for path in candidates if path.is_file()), None)
    return (
        str(resolved.resolve()) if resolved else (value or None),
        file_sha256(resolved) if resolved else None,
    )


def backend_provenance(settings, backend: str) -> dict:
    configured, weight_hash = _configured_weight(settings, backend)
    packages = {
        "gfpgan": ("gfpgan", "basicsr", "torch"),
        "codeformer": ("spandrel", "spandrel-extra-arches", "torch"),
        "realesrgan_x2plus": ("spandrel", "torch"),
    }.get(backend, ("Pillow",))
    return {
        "backend": backend,
        "weight": {"configured_or_resolved": configured, "sha256": weight_hash},
        "packages": _package_versions(packages),
    }


def evaluator_runtime_provenance(settings, model_provenance: dict) -> dict:
    return {
        "model_provenance": model_provenance,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(
            (
                "numpy",
                "Pillow",
                "insightface",
                "onnxruntime",
                "onnxruntime-gpu",
                "torch",
                "torchvision",
            )
        ),
        "normalization": NORMALIZATION,
        "face_model": getattr(settings, "face_model", None),
        "face_rec_backend": "arcface",
        "face_fiqa_backend": getattr(settings, "face_fiqa_backend", None),
        "face_fiqa_arch": getattr(settings, "face_fiqa_arch", None),
    }


def build_matrix_run_spec(
    *,
    manifest_id: str,
    source_hash: str,
    backend: str,
    fidelity: float | None,
    transform: str,
    model_provenance: dict,
) -> dict:
    body = {
        "schema_version": 3,
        "kind": "checkin_superres_matrix_run_spec",
        "manifest_id": manifest_id,
        "source_hash": source_hash,
        "normalization": NORMALIZATION,
        "backend": backend,
        "fidelity": fidelity,
        "transform": transform,
        "model_provenance": model_provenance,
        "model_provenance_hash": stable_hash(model_provenance),
    }
    return {**body, "run_spec_hash": stable_hash(body)}


def matrix_cache_key(run_spec: dict) -> str:
    expected = stable_hash(
        {key: value for key, value in run_spec.items() if key != "run_spec_hash"}
    )
    if run_spec.get("run_spec_hash") != expected:
        raise ValueError("matrix run spec hash mismatch")
    return expected


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _write_png(path: Path, image: Image.Image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG")
    return file_sha256(path)


def _source_set_hash(payload: dict) -> str:
    return stable_hash(
        [
            {
                "sample_id": row["sample_id"],
                "aligned_sha256": row.get("aligned_sha256"),
            }
            for row in [*payload["gallery"], *payload["queries"]]
        ]
    )


def _write_run_spec(artifact_root: Path, run_spec: dict) -> Path:
    path = artifact_root / "run_specs" / f"{matrix_cache_key(run_spec)}.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != run_spec:
            raise RuntimeError(f"canonical run spec collision: {path}")
    else:
        _atomic_json(path, run_spec)
    return path


def _cache_record_payload(cache: dict) -> dict:
    if cache.get("cache_role") == "shared_gallery_and_a":
        return {
            "gallery_records": cache.get("gallery_records", []),
            "query_records": cache.get("query_records", []),
        }
    return {"records": cache.get("records", [])}


def _validate_cache(cache: dict, run_spec: dict, cache_dir: Path) -> dict[str, np.ndarray]:
    if cache.get("schema_version") != 3:
        raise RuntimeError("legacy GFPGAN/embedding cache rejected for formal matrix run")
    if cache.get("kind") != "checkin_superres_matrix_cache":
        raise RuntimeError("not a generic matrix cache")
    if cache.get("run_spec_hash") != matrix_cache_key(run_spec):
        raise RuntimeError("matrix cache is isolated to a different run spec")
    if cache.get("records_hash") != stable_hash(_cache_record_payload(cache)):
        raise RuntimeError("matrix cache record metadata checksum mismatch")
    npz_path = cache_dir / cache["npz_path"]
    if not npz_path.is_file() or file_sha256(npz_path) != cache["npz_sha256"]:
        raise RuntimeError("matrix cache NPZ missing or checksum mismatch")
    with np.load(npz_path, allow_pickle=False) as loaded:
        return {name: loaded[name].copy() for name in loaded.files}


def _shared_cache(
    payload: dict,
    manifest_path: Path,
    artifact_root: Path,
    face_mod,
    settings,
    *,
    fiqa_fn: Callable[[np.ndarray], float | None],
    force: bool,
    runtime_provenance: dict,
) -> tuple[list[dict], list[dict], list[np.ndarray | None], list[np.ndarray | None], dict]:
    source_hash = _source_set_hash(payload)
    provenance = {
        "runtime": runtime_provenance,
        "recognition": "arcface",
        "fiqa": runtime_provenance.get("model_provenance", {}).get("fiqa", {}),
    }
    run_spec = build_matrix_run_spec(
        manifest_id=payload["manifest_id"],
        source_hash=source_hash,
        backend="shared_gallery_and_a",
        fidelity=None,
        transform="frozen_aligned_png_to_normalize112_to_arcface",
        model_provenance=provenance,
    )
    _write_run_spec(artifact_root, run_spec)
    cache_dir = artifact_root / "cache_v3" / "shared" / matrix_cache_key(run_spec)
    cache_path = cache_dir / "cache.json"
    if cache_path.is_file() and not force:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        arrays = _validate_cache(cache, run_spec, cache_dir)
        expected_gallery_ids = [row["sample_id"] for row in payload["gallery"]]
        expected_query_ids = [row["sample_id"] for row in payload["queries"]]
        if (
            arrays["gallery_sample_ids"].tolist() != expected_gallery_ids
            or arrays["query_sample_ids"].tolist() != expected_query_ids
            or [row["sample_id"] for row in cache["gallery_records"]]
            != expected_gallery_ids
            or [row["sample_id"] for row in cache["query_records"]]
            != expected_query_ids
        ):
            raise RuntimeError("shared cache sample order mismatch")
        for record in [*cache["gallery_records"], *cache["query_records"]]:
            normalized_path = _resolve(record.get("normalized_path"), artifact_root)
            if normalized_path is not None and (
                not normalized_path.is_file()
                or file_sha256(normalized_path) != record.get("normalized_sha256")
            ):
                raise RuntimeError("shared normalized112 image checksum mismatch")
        gallery = _unpack_vectors(arrays["gallery"], arrays["gallery_valid"])
        query_a = _unpack_vectors(arrays["query_a"], arrays["query_a_valid"])
        cache = {**cache, "runtime": {**cache["runtime"], "cache_reused": True}}
        return cache["gallery_records"], cache["query_records"], gallery, query_a, cache
    if force:
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree(artifact_root / "images" / "shared", ignore_errors=True)

    records: dict[str, list[dict]] = {"gallery": [], "query": []}
    vectors: dict[str, list[np.ndarray | None]] = {"gallery": [], "query": []}
    for cohort, rows in (("gallery", payload["gallery"]), ("query", payload["queries"])):
        for row in rows:
            record = dict(row)
            path = _resolve(row.get("aligned_path"), manifest_path.parent)
            if path is None:
                record["normalization_status"] = "unavailable"
                vectors[cohort].append(None)
                records[cohort].append(record)
                continue
            source = _load_frozen_rgb(path)
            source_hash_actual = file_sha256(path)
            if row.get("aligned_sha256") and source_hash_actual != row["aligned_sha256"]:
                raise RuntimeError(f"fixed source checksum changed: {path}")
            normalized = normalize112(source)
            normalized_path = (
                artifact_root
                / "images"
                / "shared"
                / cohort
                / f"{row['sample_id']}.png"
            )
            normalized_hash = _write_png(normalized_path, normalized)
            vector = embed_normalized112(normalized, face_mod)
            record.update(
                {
                    "source_native_dimensions": list(source.size),
                    "source_sha256_actual": source_hash_actual,
                    "normalized_dimensions": list(normalized.size),
                    "normalized_sha256": normalized_hash,
                    "normalized_path": _relative(normalized_path, artifact_root),
                    "normalization_status": "ok",
                    "embedding_available": vector is not None,
                    "fiqa_normalized112": (
                        fiqa_normalized112(normalized, fiqa_fn)
                        if cohort == "query"
                        else None
                    ),
                }
            )
            vectors[cohort].append(vector)
            records[cohort].append(record)

    gallery_matrix, gallery_valid = _pack_vectors(vectors["gallery"])
    query_matrix, query_valid = _pack_vectors(vectors["query"])
    npz_path = cache_dir / "embeddings.npz"
    _atomic_npz(
        npz_path,
        gallery_sample_ids=np.asarray(
            [row["sample_id"] for row in records["gallery"]], dtype="U128"
        ),
        gallery=gallery_matrix,
        gallery_valid=gallery_valid,
        query_sample_ids=np.asarray(
            [row["sample_id"] for row in records["query"]], dtype="U128"
        ),
        query_a=query_matrix,
        query_a_valid=query_valid,
    )
    cache = {
        "schema_version": 3,
        "kind": "checkin_superres_matrix_cache",
        "cache_role": "shared_gallery_and_a",
        "run_spec_hash": matrix_cache_key(run_spec),
        "run_spec": run_spec,
        "npz_path": npz_path.name,
        "npz_sha256": file_sha256(npz_path),
        "gallery_records": records["gallery"],
        "query_records": records["query"],
        "runtime": {"cache_reused": False},
    }
    cache["records_hash"] = stable_hash(_cache_record_payload(cache))
    _atomic_json(cache_path, cache)
    return (
        records["gallery"],
        records["query"],
        vectors["gallery"],
        vectors["query"],
        cache,
    )


def _control_transform(image: Image.Image, backend: str) -> Image.Image:
    if backend == "resize512":
        return image.resize((512, 512), Image.Resampling.BILINEAR)
    if backend == "resize_x2":
        return image.resize((224, 224), Image.Resampling.BICUBIC)
    raise ValueError(f"unknown control: {backend}")


def _backend_cache(
    payload: dict,
    artifact_root: Path,
    query_records: list[dict],
    face_mod,
    settings,
    spec: dict,
    *,
    fiqa_fn: Callable[[np.ndarray], float | None],
    force: bool,
    runtime_provenance: dict,
) -> tuple[list[dict], list[np.ndarray | None], dict]:
    backend = spec["backend"]
    threshold = float(payload["prepare_config"]["face_fiqa_poor_thresh"])
    provenance = {
        "transform": backend_provenance(settings, backend),
        "runtime": runtime_provenance,
        "fiqa_diagnostic_threshold": payload["prepare_config"][
            "face_fiqa_poor_thresh"
        ],
    }
    run_spec = build_matrix_run_spec(
        manifest_id=payload["manifest_id"],
        source_hash=_source_set_hash(payload),
        backend=backend,
        fidelity=spec["fidelity"],
        transform=spec["transform"],
        model_provenance=provenance,
    )
    _write_run_spec(artifact_root, run_spec)
    cache_dir = artifact_root / "cache_v3" / "runs" / matrix_cache_key(run_spec)
    cache_path = cache_dir / "cache.json"
    if cache_path.is_file() and not force:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        arrays = _validate_cache(cache, run_spec, cache_dir)
        expected_ids = [row["sample_id"] for row in payload["queries"]]
        if (
            arrays["sample_ids"].tolist() != expected_ids
            or [row["sample_id"] for row in cache["records"]] != expected_ids
        ):
            raise RuntimeError("backend cache sample order mismatch")
        for record in cache["records"]:
            fiqa_after = record.get("fiqa_after")
            if (
                record.get("status") == "ok"
                and fiqa_after is not None
                and math.isfinite(float(fiqa_after))
            ):
                expected_accepted, expected_reason = superres_quality_ok(
                    float(fiqa_after),
                    poor_threshold=threshold,
                )
            else:
                expected_accepted, expected_reason = (
                    False,
                    "fiqa_unavailable_or_nonfinite",
                )
            if bool(record.get("fiqa_diagnostic_passed")) != bool(
                expected_accepted
            ):
                raise RuntimeError(
                    f"{backend} cached FIQA diagnostic does not match threshold"
                )
            if (
                record.get("status") == "ok"
                and record.get("fiqa_diagnostic_reason") != expected_reason
            ):
                raise RuntimeError(
                    f"{backend} cached FIQA diagnostic reason does not match threshold"
                )
            for path_key, hash_key in (
                ("native_path", "native_sha256"),
                ("normalized_path", "normalized_sha256"),
            ):
                output_path = _resolve(record.get(path_key), artifact_root)
                if output_path is not None and (
                    not output_path.is_file()
                    or file_sha256(output_path) != record.get(hash_key)
                ):
                    raise RuntimeError(
                        f"{backend} cached image checksum mismatch: {path_key}"
                    )
        vectors = _unpack_vectors(arrays["vectors"], arrays["valid"])
        return cache["records"], vectors, {**cache, "runtime": {**cache["runtime"], "cache_reused": True}}
    if force:
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree(artifact_root / "images" / backend, ignore_errors=True)

    if backend == "codeformer":
        settings.face_codeformer_fidelity = 1.0
    startup_error = None
    load_seconds = 0.0
    if backend not in {"resize512", "resize_x2"}:
        load_started = time.perf_counter()
        face_mod._ensure_superres(backend)
        load_seconds = time.perf_counter() - load_started
        startup_error = face_mod.superres_error(backend)

    records: list[dict] = []
    vectors: list[np.ndarray | None] = []
    total_seconds = 0.0
    invocations = 0
    successes = 0
    failures = 0
    for frozen, shared in zip(payload["queries"], query_records):
        record = {
            "sample_id": frozen["sample_id"],
            "pid": frozen["pid"],
            "track": frozen["track"],
            "eligibility": frozen.get("eligibility", "none"),
            "category": (frozen.get("quality") or {}).get("category", "none"),
            "rule_quality": (frozen.get("quality") or {}).get("rule_quality"),
            "fiqa_before": shared.get("fiqa_normalized112"),
            "backend": backend,
            "status": "unavailable",
            "failure_reason": None,
            "fiqa_diagnostic_passed": False,
            "fiqa_diagnostic_reason": None,
        }
        normalized_path = _resolve(shared.get("normalized_path"), artifact_root)
        if normalized_path is None:
            record["failure_reason"] = frozen.get("face_status", "not_aligned")
            records.append(record)
            vectors.append(None)
            continue
        source = _load_frozen_rgb(normalized_path)
        if startup_error:
            record["failure_reason"] = f"startup_error:{startup_error}"
            failures += 1
            records.append(record)
            vectors.append(None)
            continue
        invocations += 1
        started = time.perf_counter()
        try:
            if backend in {"resize512", "resize_x2"}:
                native = _control_transform(source, backend)
            else:
                native = face_mod.enhance(
                    source,
                    aligned=True,
                    backend=backend,
                )
                if native is source:
                    error = face_mod.superres_error(backend)
                    raise RuntimeError(error or f"{backend}_no_output")
                if not isinstance(native, Image.Image):
                    raise TypeError("backend returned non-image output")
            elapsed = time.perf_counter() - started
            native = native.convert("RGB")
            expected_size = EXPECTED_NATIVE_SIZES[backend]
            if native.size != expected_size:
                raise ValueError(
                    f"{backend} output size {native.size} != {expected_size}"
                )
            normalized = normalize112(native)
            native_path = (
                artifact_root
                / "images"
                / backend
                / "native"
                / f"{frozen['sample_id']}.png"
            )
            normalized_output_path = (
                artifact_root
                / "images"
                / backend
                / "normalized112"
                / f"{frozen['sample_id']}.png"
            )
            native_hash = _write_png(native_path, native)
            normalized_hash = _write_png(normalized_output_path, normalized)
            vector = embed_normalized112(normalized, face_mod)
            fiqa_after = fiqa_normalized112(normalized, fiqa_fn)
            if fiqa_after is None:
                accepted, post_reason = False, "fiqa_unavailable_or_nonfinite"
            else:
                accepted, post_reason = superres_quality_ok(
                    fiqa_after,
                    poor_threshold=threshold,
                )
            succeeded = vector is not None
            if not succeeded:
                raise RuntimeError("normalized_embedding_failed")
            record.update(
                {
                    "status": "ok",
                    "transform_seconds": round(elapsed, 6),
                    "native_path": _relative(native_path, artifact_root),
                    "native_dimensions": list(native.size),
                    "native_sha256": native_hash,
                    "normalized_path": _relative(normalized_output_path, artifact_root),
                    "normalized_dimensions": list(normalized.size),
                    "normalized_sha256": normalized_hash,
                    "embedding_available": True,
                    "fiqa_after": fiqa_after,
                    "fiqa_diagnostic_passed": bool(accepted),
                    "fiqa_diagnostic_reason": post_reason,
                }
            )
            total_elapsed = time.perf_counter() - started
            total_seconds += total_elapsed
            record["latency_seconds"] = round(total_elapsed, 6)
            successes += 1
            vectors.append(vector)
        except Exception as exc:  # noqa: BLE001
            total_elapsed = time.perf_counter() - started
            total_seconds += total_elapsed
            failures += 1
            record.update(
                {
                    "status": "failed",
                    "latency_seconds": round(total_elapsed, 6),
                    "failure_reason": f"{type(exc).__name__}:{exc}",
                    "embedding_available": False,
                }
            )
            vectors.append(None)
        records.append(record)

    matrix, valid = _pack_vectors(vectors)
    npz_path = cache_dir / "embeddings.npz"
    _atomic_npz(
        npz_path,
        sample_ids=np.asarray(
            [row["sample_id"] for row in records], dtype="U128"
        ),
        vectors=matrix,
        valid=valid,
    )
    cache = {
        "schema_version": 3,
        "kind": "checkin_superres_matrix_cache",
        "cache_role": "backend_or_control",
        "run_spec_hash": matrix_cache_key(run_spec),
        "run_spec": run_spec,
        "npz_path": npz_path.name,
        "npz_sha256": file_sha256(npz_path),
        "records": records,
        "runtime": {
            "cache_reused": False,
            "startup_error": startup_error,
            "load_seconds": round(load_seconds, 6),
            "invocations": invocations,
            "successes": successes,
            "failures": failures,
            "total_seconds": round(total_seconds, 6),
            "mean_seconds": (
                round(total_seconds / invocations, 6) if invocations else None
            ),
        },
    }
    cache["records_hash"] = stable_hash(_cache_record_payload(cache))
    _atomic_json(cache_path, cache)
    return records, vectors, cache


def _score_rows(
    payload: dict,
    query_records: list[dict],
    vectors_by_arm: dict[str, list[np.ndarray | None]],
    templates: dict[str, np.ndarray],
) -> dict[str, list[dict]]:
    rows_by_arm: dict[str, list[dict]] = {}
    for arm, vectors in vectors_by_arm.items():
        rows = []
        for frozen, record, vector in zip(payload["queries"], query_records, vectors):
            rows.append(
                {
                    "sample_id": frozen["sample_id"],
                    "pid": frozen["pid"],
                    "track": frozen["track"],
                    "arm": arm,
                    "eligibility": frozen.get("eligibility", "none"),
                    "category": (frozen.get("quality") or {}).get("category", "none"),
                    "rule_quality": (frozen.get("quality") or {}).get("rule_quality"),
                    "fiqa_before": record.get("fiqa_normalized112"),
                    **_score(vector, templates, frozen["pid"]),
                }
            )
        rows_by_arm[arm] = rows
    return rows_by_arm


def _summaries(rows_by_arm: dict[str, list[dict]], threshold: float) -> dict:
    return {
        arm: {
            "full316": summarize_scores(rows, threshold),
            "aligned156": summarize_scores(
                [row for row in rows if row["eligibility"] != "none"], threshold
            ),
            "by_eligibility": {
                value: summarize_scores(
                    [row for row in rows if row["eligibility"] == value], threshold
                )
                for value in ("direct", "recoverable", "unusable", "none")
            },
            "by_category": {
                value: summarize_scores(
                    [row for row in rows if row["category"] == value], threshold
                )
                for value in ("clear", "marginal", "poor", "none")
            },
        }
        for arm, rows in rows_by_arm.items()
    }


def _write_long_rows(
    artifact_root: Path,
    rows_by_arm: dict[str, list[dict]],
    backend_records: dict[str, list[dict]],
) -> tuple[Path, Path, list[dict]]:
    lookup = {
        (record["sample_id"], backend): record
        for backend, records in backend_records.items()
        for record in records
    }
    rows: list[dict] = []
    arm_backend = {
        spec["b_arm"]: spec["backend"] for spec in (*BACKEND_SPECS, *CONTROL_SPECS)
    } | {spec["c_arm"]: spec["backend"] for spec in BACKEND_SPECS}
    for arm, scores in rows_by_arm.items():
        backend = arm_backend.get(arm)
        for score in scores:
            diagnostic = lookup.get((score["sample_id"], backend), {})
            rows.append(
                {
                    **score,
                    "backend": backend,
                    "transform_status": diagnostic.get("status"),
                    "failure_reason": diagnostic.get("failure_reason"),
                    "fiqa_diagnostic_passed": diagnostic.get(
                        "fiqa_diagnostic_passed"
                    ),
                    "fiqa_after": diagnostic.get("fiqa_after"),
                    "latency_seconds": diagnostic.get("latency_seconds"),
                    "native_dimensions": diagnostic.get("native_dimensions"),
                    "native_sha256": diagnostic.get("native_sha256"),
                    "normalized_dimensions": diagnostic.get(
                        "normalized_dimensions"
                    ),
                    "normalized_sha256": diagnostic.get("normalized_sha256"),
                }
            )
    json_path = artifact_root / "per_sample_long.json"
    csv_path = artifact_root / "per_sample_long.csv"
    _atomic_json(json_path, rows)
    fieldnames = sorted({key for row in rows for key in row})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: canonical_json(value)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    os.replace(temporary, csv_path)
    return json_path, csv_path, rows


def _artifact_checksums(artifact_root: Path, include: set[Path]) -> list[dict]:
    return [
        {
            "path": _relative(path, artifact_root),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted({path.resolve() for path in include})
        if path.is_file()
    ]


def _cache_artifact_paths(
    artifact_root: Path,
    caches: dict[str, dict],
) -> set[Path]:
    paths: set[Path] = set()
    for cache in caches.values():
        run_spec_hash = cache["run_spec_hash"]
        paths.add(artifact_root / "run_specs" / f"{run_spec_hash}.json")
        role = cache.get("cache_role")
        cache_dir = (
            artifact_root / "cache_v3" / "shared" / run_spec_hash
            if role == "shared_gallery_and_a"
            else artifact_root / "cache_v3" / "runs" / run_spec_hash
        )
        paths.update(path for path in cache_dir.rglob("*") if path.is_file())
        record_groups = (
            [
                cache.get("gallery_records", []),
                cache.get("query_records", []),
            ]
            if role == "shared_gallery_and_a"
            else [cache.get("records", [])]
        )
        for records in record_groups:
            for record in records:
                for key in (
                    "normalized_path",
                    "native_path",
                ):
                    path = _resolve(record.get(key), artifact_root)
                    if path is not None:
                        paths.add(path)
    return paths


def evaluate_matrix_payload(
    payload: dict,
    manifest_path: Path,
    artifact_root: Path,
    *,
    face_mod,
    settings,
    seed: int,
    bootstrap_samples: int,
    permutation_samples: int,
    force_recompute: bool,
    fiqa_fn: Callable[[np.ndarray], float | None] = deep_fiqa_score,
    runtime_provenance: dict | None = None,
) -> dict:
    """Run the matrix; the CLI performs the frozen-ID/cohort checks before this."""
    artifact_root.mkdir(parents=True, exist_ok=True)
    if (artifact_root / "embedding_cache.json").exists():
        raise RuntimeError(
            "legacy GFPGAN embedding cache rejected; use a clean matrix artifact root"
        )
    settings.face_rec_backend = "arcface"
    active_runtime_provenance = runtime_provenance or evaluator_runtime_provenance(
        settings,
        payload.get("model_provenance", {}),
    )
    gallery_records, query_records, gallery_vectors, query_a, shared_cache = _shared_cache(
        payload,
        manifest_path,
        artifact_root,
        face_mod,
        settings,
        fiqa_fn=fiqa_fn,
        force=force_recompute,
        runtime_provenance=active_runtime_provenance,
    )
    templates = _templates(gallery_records, gallery_vectors)
    vectors_by_arm: dict[str, list[np.ndarray | None]] = {
        "A_original": query_a,
        "P_off_original": [
            p_off_original(vector, row.get("eligibility", "none"))
            for vector, row in zip(query_a, payload["queries"])
        ],
    }
    records_by_backend: dict[str, list[dict]] = {}
    caches: dict[str, dict] = {"shared": shared_cache}
    for spec in (*BACKEND_SPECS, *CONTROL_SPECS):
        records, vectors, cache = _backend_cache(
            payload,
            artifact_root,
            query_records,
            face_mod,
            settings,
            spec,
            fiqa_fn=fiqa_fn,
            force=force_recompute,
            runtime_provenance=active_runtime_provenance,
        )
        records_by_backend[spec["backend"]] = records
        caches[spec["backend"]] = cache
        vectors_by_arm[spec["b_arm"]] = vectors
        if spec["c_arm"]:
            c_vectors = []
            for frozen, original, restored, diagnostic in zip(
                payload["queries"], query_a, vectors, records
            ):
                arms = derive_backend_arms(
                    original,
                    restored,
                    eligibility=frozen.get("eligibility", "none"),
                    transform_succeeded=diagnostic.get("status") == "ok",
                    b_arm=spec["b_arm"],
                    c_arm=spec["c_arm"],
                )
                c_vectors.append(arms[spec["c_arm"]])
            vectors_by_arm[spec["c_arm"]] = c_vectors

    threshold = float(payload["prepare_config"]["face_hit_thresh"])
    rows_by_arm = _score_rows(payload, query_records, vectors_by_arm, templates)
    summaries = _summaries(rows_by_arm, threshold)
    combined_arrays: dict[str, np.ndarray] = {
        "sample_ids": np.asarray(
            [row["sample_id"] for row in payload["queries"]], dtype="U128"
        )
    }
    for arm, vectors in vectors_by_arm.items():
        matrix, valid = _pack_vectors(vectors)
        combined_arrays[arm] = matrix
        combined_arrays[f"{arm}_valid"] = valid
    combined_npz = artifact_root / "matrix_embeddings.npz"
    _atomic_npz(combined_npz, **combined_arrays)
    recoverable_indices = [
        index
        for index, row in enumerate(payload["queries"])
        if row.get("eligibility") == "recoverable"
    ]
    recoverable_main = {
        arm: {
            **summarize_scores(
                [rows_by_arm[arm][index] for index in recoverable_indices],
                threshold,
            ),
            "rank1_uncertainty": pid_cluster_bootstrap_rate(
                [rows_by_arm[arm][index] for index in recoverable_indices],
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            ),
        }
        for arm in MAIN_ARMS
    }
    paired = {}
    aligned_indices = [
        index
        for index, row in enumerate(payload["queries"])
        if row.get("eligibility") != "none"
    ]
    for offset, spec in enumerate(BACKEND_SPECS):
        for candidate in (spec["b_arm"], spec["c_arm"]):
            before = [rows_by_arm["A_original"][index] for index in recoverable_indices]
            after = [rows_by_arm[candidate][index] for index in recoverable_indices]
            paired[f"recoverable_A_vs_{candidate}"] = paired_uncertainty(
                before,
                after,
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
                seed=seed + offset,
            )
        for cohort_name, indices in (
            ("recoverable", recoverable_indices),
            ("aligned", aligned_indices),
        ):
            paired[
                f"{cohort_name}_{spec['b_arm']}_vs_{spec['c_arm']}"
            ] = paired_uncertainty(
                [rows_by_arm[spec["b_arm"]][index] for index in indices],
                [rows_by_arm[spec["c_arm"]][index] for index in indices],
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
                seed=seed + offset,
            )

    json_path, csv_path, long_rows = _write_long_rows(
        artifact_root, rows_by_arm, records_by_backend
    )
    figure_paths = render_matrix_figures(
        artifact_root,
        payload,
        manifest_path,
        query_records,
        gallery_records,
        rows_by_arm,
        records_by_backend,
        recoverable_main,
        paired,
    )
    template_pids = set(templates)
    query_pids = {row["pid"] for row in payload["queries"]}
    backend_diagnostics = {}
    for spec in BACKEND_SPECS:
        records = records_by_backend[spec["backend"]]
        drift = [
            float(original @ restored)
            for original, restored in zip(
                query_a, vectors_by_arm[spec["b_arm"]]
            )
            if original is not None and restored is not None
        ]
        fiqa_deltas = [
            float(row["fiqa_after"]) - float(row["fiqa_before"])
            for row in records
            if row.get("fiqa_before") is not None
            and row.get("fiqa_after") is not None
        ]
        aligned_records = [
            row for row in records if row["eligibility"] != "none"
        ]
        backend_diagnostics[spec["backend"]] = {
            "embedding_drift_cosine": {
                "count": len(drift),
                "mean": round(float(np.mean(drift)), 6) if drift else None,
                "median": round(float(np.median(drift)), 6) if drift else None,
            },
            "fiqa_delta": {
                "count": len(fiqa_deltas),
                "mean": (
                    round(float(np.mean(fiqa_deltas)), 6)
                    if fiqa_deltas
                    else None
                ),
                "median": (
                    round(float(np.median(fiqa_deltas)), 6)
                    if fiqa_deltas
                    else None
                ),
            },
            "fiqa_diagnostic_aligned": {
                "denominator": len(aligned_records),
                "above_threshold": sum(
                    bool(row.get("fiqa_diagnostic_passed"))
                    for row in aligned_records
                ),
                "below_or_unavailable": sum(
                    not bool(row.get("fiqa_diagnostic_passed"))
                    for row in aligned_records
                ),
                "note": "diagnostic only; FIQA does not route B or C",
            },
            "failures": {
                "count": sum(
                    row.get("status") != "ok" for row in aligned_records
                ),
                "reasons": sorted(
                    {
                        str(row["failure_reason"])
                        for row in aligned_records
                        if row.get("failure_reason")
                    }
                ),
            },
        }
    current_run_spec_paths = sorted(
        artifact_root / "run_specs" / f"{cache['run_spec_hash']}.json"
        for cache in caches.values()
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "checkin_superres_multi_backend_matrix_result",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "manifest_id": payload["manifest_id"],
        "run_semantics": {
            "normalization": NORMALIZATION,
            "gallery": "shared original check-in; recomputed under normalize112",
            "A_original": "shared frozen aligned Query recomputed under normalize112",
            "P_off_original": (
                "descriptive only: direct+recoverable=A; unusable/none unavailable"
            ),
            "C": (
                "direct=A; recoverable=backend B only on transform, normalized "
                "embedding success; unusable/none unavailable; FIQA diagnostic "
                "does not route C; no fallback"
            ),
            "primary": "exact frozen recoverable cohort: A vs C1/C2/C3",
            "controls": [spec["b_arm"] for spec in CONTROL_SPECS],
            "genuine_wrong_identity_is_not_fmr": True,
        },
        "cohorts": {
            "full": len(payload["queries"]),
            "aligned": sum(bool(row.get("aligned_path")) for row in payload["queries"]),
            "recoverable": len(recoverable_indices),
            "eligibility": {
                value: sum(
                    row.get("eligibility", "none") == value
                    for row in payload["queries"]
                )
                for value in ("direct", "recoverable", "unusable", "none")
            },
            "category": {
                value: sum(
                    (row.get("quality") or {}).get("category", "none") == value
                    for row in payload["queries"]
                )
                for value in ("clear", "marginal", "poor", "none")
            },
        },
        "template_coverage": {
            "gallery_records": len(gallery_records),
            "template_pids": len(template_pids),
            "template_pid_list": sorted(template_pids),
            "query_pids": len(query_pids),
            "missing_query_pids": sorted(query_pids - template_pids),
            "missing_pid_policy": "end-to-end failure; no template synthesis",
        },
        "recoverable_main": recoverable_main,
        "b_c_aligned_and_fiqa_diagnostic": {
            spec["backend"]: {
                "B_aligned156": summaries[spec["b_arm"]]["aligned156"],
                "C_aligned156": summaries[spec["c_arm"]]["aligned156"],
                "fiqa_diagnostic": backend_diagnostics[spec["backend"]][
                    "fiqa_diagnostic_aligned"
                ],
            }
            for spec in BACKEND_SPECS
        },
        "results": summaries,
        "paired_transitions": paired,
        "diagnostics": backend_diagnostics,
        "qualitative_selection": {
            category: (
                selected["sample_id"]
                if (
                    selected := _median_rule_quality(
                        [
                            row
                            for row in payload["queries"]
                            if (row.get("quality") or {}).get("category")
                            == category
                        ]
                    )
                )
                else None
            )
            for category in ("clear", "marginal", "poor")
        },
        "runtime": {
            backend: cache["runtime"] for backend, cache in caches.items()
        },
        "artifacts": {
            "per_sample_json": _relative(json_path, artifact_root),
            "per_sample_csv": _relative(csv_path, artifact_root),
            "embeddings_npz": _relative(combined_npz, artifact_root),
            "embeddings_npz_sha256": file_sha256(combined_npz),
            "long_rows": len(long_rows),
            "figures": [_relative(path, artifact_root) for path in figure_paths],
            "run_specs": [
                _relative(path, artifact_root)
                for path in current_run_spec_paths
            ],
            "cache_schema": 3,
        },
    }
    result_path = artifact_root / "matrix_result.json"
    _atomic_json(result_path, result)
    checksum_path = artifact_root / "checksums.json"
    checksums = {
        "schema_version": 1,
        "manifest_id": payload["manifest_id"],
        "files": _artifact_checksums(
            artifact_root,
            include=(
                _cache_artifact_paths(artifact_root, caches)
                | {
                    combined_npz,
                    json_path,
                    csv_path,
                    *figure_paths,
                }
            ),
        ),
    }
    _atomic_json(checksum_path, checksums)
    result["artifacts"]["checksums"] = _relative(checksum_path, artifact_root)
    _atomic_json(result_path, result)
    return result


def _validate_frozen_cohort(payload: dict) -> None:
    counts = {
        "full": len(payload["queries"]),
        "aligned": sum(bool(row.get("aligned_path")) for row in payload["queries"]),
        "recoverable": sum(
            row.get("eligibility") == "recoverable" for row in payload["queries"]
        ),
    }
    expected = {"full": 316, "aligned": 156, "recoverable": 40}
    if counts != expected:
        raise RuntimeError(f"frozen cohort mismatch: expected {expected}, got {counts}")


def evaluate_matrix(args) -> int:
    from app import face as face_mod
    from app.core.config import settings

    manifest_path = Path(args.manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_manifest(payload, manifest_path)
    if payload.get("manifest_id") != FROZEN_MANIFEST_ID:
        raise RuntimeError(
            f"evaluate-matrix requires frozen manifest {FROZEN_MANIFEST_ID}"
        )
    _validate_frozen_cohort(payload)
    live_model_provenance = _model_provenance(settings)
    if not _provenance_compatible(
        payload.get("model_provenance", {}),
        live_model_provenance,
    ):
        raise RuntimeError(
            "runtime ArcFace/FIQA provenance does not match frozen manifest"
        )
    runtime_provenance = evaluator_runtime_provenance(
        settings,
        live_model_provenance,
    )
    missing_weight_hashes = [
        spec["backend"]
        for spec in BACKEND_SPECS
        if backend_provenance(settings, spec["backend"])["weight"]["sha256"] is None
    ]
    if missing_weight_hashes:
        raise RuntimeError(
            "formal matrix run requires pre-provisioned, hashable weights for: "
            + ", ".join(missing_weight_hashes)
        )
    artifact_root = Path(args.artifact_root).resolve()
    result = evaluate_matrix_payload(
        payload,
        manifest_path,
        artifact_root,
        face_mod=face_mod,
        settings=settings,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        force_recompute=args.force_recompute,
        runtime_provenance=runtime_provenance,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = artifact_root / "matrix_result.json"
    if output != source:
        shutil.copyfile(source, output)
    print(f"[saved] {output}")
    return 0


__all__ = [
    "ALL_ARMS",
    "BACKEND_SPECS",
    "CONTROL_SPECS",
    "FROZEN_MANIFEST_ID",
    "MAIN_ARMS",
    "NORMALIZATION",
    "backend_provenance",
    "build_matrix_run_spec",
    "derive_backend_arms",
    "embed_normalized112",
    "evaluate_matrix",
    "evaluate_matrix_payload",
    "fiqa_normalized112",
    "matrix_cache_key",
    "normalize112",
    "p_off_original",
]
