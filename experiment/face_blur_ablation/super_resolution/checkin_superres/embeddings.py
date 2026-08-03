"""Embedding backend execution and validated A/B/C cache handling."""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.identity.face.quality import deep_fiqa_score, superres_quality_ok

from .common import (
    ARMS,
    SCHEMA_VERSION,
    file_sha256,
    manifest_identity,
    stable_hash,
    _relative,
    _resolve,
    _save_bgr,
)
from .preparation import _model_provenance


def select_arm_embeddings(
    original: np.ndarray | None,
    superres: np.ndarray | None,
    *,
    eligibility: str,
    superres_succeeded: bool,
    post_superres_accepted: bool,
) -> dict[str, np.ndarray | None]:
    """Derive A/B/C from frozen eligibility without invoking GFPGAN."""
    b_vector = superres if superres_succeeded and superres is not None else None
    if eligibility == "direct":
        c_vector = original
    elif eligibility == "recoverable":
        c_vector = b_vector
    else:
        c_vector = None
    return {
        "A_original": original,
        "B_all_superres": b_vector,
        "C_gated_superres": c_vector,
    }


def _verify_manifest(payload: dict, manifest_path: Path) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("evaluate仅接受schema-v3 manifest")
    if payload.get("kind") != "checkin_superres_abc_manifest":
        raise ValueError("不是checkin_superres_abc_manifest")
    if stable_hash(payload["prepare_config"]) != payload.get("prepare_config_hash"):
        raise RuntimeError("prepare配置快照hash不匹配")
    if stable_hash(payload["model_provenance"]) != payload.get(
        "model_provenance_hash"
    ):
        raise RuntimeError("模型provenance hash不匹配")
    fixed_identity = manifest_identity(
        payload["coverage"],
        payload["prepare_config_hash"],
        payload["model_provenance_hash"],
        payload["gallery"],
        payload["queries"],
    )
    if stable_hash(fixed_identity) != payload.get("manifest_id"):
        raise RuntimeError("manifest固定输入identity hash不匹配")
    for row in [*payload["gallery"], *payload["queries"]]:
        for path_key, hash_key in (
            ("source_path", "source_sha256"),
            ("aligned_path", "aligned_sha256"),
        ):
            path = _resolve(row.get(path_key), manifest_path.parent)
            expected = row.get(hash_key)
            if path is None:
                continue
            if not path.is_file():
                raise FileNotFoundError(f"固定输入不存在：{path}")
            if expected and file_sha256(path) != expected:
                raise RuntimeError(f"固定输入hash不匹配：{path}")


def _normalise(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(value)):
        return None
    norm = float(np.linalg.norm(value))
    return value / norm if math.isfinite(norm) and norm > 0 else None


def _pack_vectors(vectors: list[np.ndarray | None], dim: int = 512) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(vectors), dim), dtype=np.float32)
    valid = np.zeros(len(vectors), dtype=np.bool_)
    for index, vector in enumerate(vectors):
        value = _normalise(vector)
        if value is not None:
            if value.size != dim:
                raise ValueError(f"embedding维度错误：{value.size} != {dim}")
            matrix[index] = value
            valid[index] = True
    return matrix, valid


def _unpack_vectors(matrix: np.ndarray, valid: np.ndarray) -> list[np.ndarray | None]:
    return [
        np.asarray(matrix[index], dtype=np.float32) if bool(valid[index]) else None
        for index in range(len(valid))
    ]


def _compute_embedding_cache(
    payload: dict,
    manifest_path: Path,
    artifact_dir: Path,
    face_mod,
    settings,
) -> tuple[list[dict], list[dict], dict]:
    original_dir = artifact_dir / "aligned_original"
    superres_dir = artifact_dir / "aligned_superres"
    original_dir.mkdir(parents=True, exist_ok=True)
    superres_dir.mkdir(parents=True, exist_ok=True)

    gallery_records = []
    gallery_vectors = []
    for row in payload["gallery"]:
        path = _resolve(row["aligned_path"], manifest_path.parent)
        assert path is not None
        vector = _normalise(
            face_mod.embed_aligned_face(
                np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy(),
                "arcface",
            )
        )
        gallery_records.append(dict(row))
        gallery_vectors.append(vector)

    settings.face_superres = "gfpgan"
    face_mod._ensure_superres()
    startup_error = face_mod.superres_error()
    frozen_poor_threshold = float(
        payload["prepare_config"]["face_fiqa_poor_thresh"]
    )
    query_records = []
    query_a = []
    query_b = []
    query_c = []
    gfpgan_seconds = 0.0
    superres_requests = 0
    gfpgan_invocations = 0
    gfpgan_outputs = 0
    enhanced_embedding_success = 0
    startup_blocked = 0
    gfpgan_no_output = 0
    enhanced_embedding_failure = 0
    for index, frozen in enumerate(payload["queries"], start=1):
        record = dict(frozen)
        original = enhanced = None
        sr_success = False
        sr_reason = None
        fiqa_after = None
        post_accepted = False
        restoration_output = False
        aligned_path = _resolve(frozen.get("aligned_path"), manifest_path.parent)
        if aligned_path is not None:
            original_path = original_dir / f"{frozen['sample_id']}.png"
            shutil.copyfile(aligned_path, original_path)
            original_hash = file_sha256(original_path)
            original_bgr = np.asarray(
                Image.open(original_path).convert("RGB")
            )[:, :, ::-1].copy()
            original = _normalise(
                face_mod.embed_aligned_face(original_bgr, "arcface")
            )
            record["original_aligned_path"] = _relative(original_path, artifact_dir)
            record["original_aligned_sha256"] = original_hash
            superres_requests += 1
            if startup_error:
                startup_blocked += 1
                sr_reason = f"startup_error:{startup_error}"
            else:
                gfpgan_invocations += 1
                original_rgb = Image.fromarray(original_bgr[:, :, ::-1])
                started = time.perf_counter()
                restored = face_mod.enhance(original_rgb, aligned=True)
                elapsed = time.perf_counter() - started
                gfpgan_seconds += elapsed
                record["gfpgan_seconds"] = round(elapsed, 6)
                if restored is original_rgb:
                    gfpgan_no_output += 1
                    sr_reason = "gfpgan_no_output"
                else:
                    restoration_output = True
                    gfpgan_outputs += 1
                    enhanced_bgr = np.asarray(
                        restored.convert("RGB")
                    )[:, :, ::-1].copy()
                    sr_path = superres_dir / f"{frozen['sample_id']}.png"
                    sr_hash = _save_bgr(sr_path, enhanced_bgr)
                    enhanced = _normalise(
                        face_mod.embed_aligned_face(enhanced_bgr, "arcface")
                    )
                    if enhanced is None:
                        enhanced_embedding_failure += 1
                        sr_reason = "superres_embedding_failed"
                    else:
                        sr_success = True
                        enhanced_embedding_success += 1
                    fiqa_after = deep_fiqa_score(enhanced_bgr)
                    post_accepted, post_reason = superres_quality_ok(
                        fiqa_after,
                        poor_threshold=frozen_poor_threshold,
                    )
                    record["superres_aligned_path"] = _relative(
                        sr_path, artifact_dir
                    )
                    record["superres_aligned_sha256"] = sr_hash
                    record["post_superres_reason"] = post_reason
        else:
            sr_reason = frozen.get("face_status", "not_detected")

        arms = select_arm_embeddings(
            original,
            enhanced,
            eligibility=frozen.get("eligibility", "none"),
            superres_succeeded=sr_success,
            post_superres_accepted=post_accepted,
        )
        query_a.append(arms["A_original"])
        query_b.append(arms["B_all_superres"])
        query_c.append(arms["C_gated_superres"])
        record.update(
            {
                "superres_attempted": aligned_path is not None,
                "restoration_output": restoration_output,
                "superres_succeeded": sr_success,
                "superres_failure_reason": sr_reason,
                "fiqa_before": (frozen.get("quality") or {}).get("fiqa"),
                "fiqa_after": fiqa_after,
                "post_superres_accepted": post_accepted,
                "embedding_cosine_original_superres": (
                    round(float(original @ enhanced), 6)
                    if original is not None and enhanced is not None
                    else None
                ),
            }
        )
        query_records.append(record)
        if index % 20 == 0 or index == len(payload["queries"]):
            print(f"    evaluate A/B {index}/{len(payload['queries'])}", flush=True)

    gallery_matrix, gallery_valid = _pack_vectors(gallery_vectors)
    a_matrix, a_valid = _pack_vectors(query_a)
    b_matrix, b_valid = _pack_vectors(query_b)
    c_matrix, c_valid = _pack_vectors(query_c)
    npz_path = artifact_dir / "embeddings.npz"
    np.savez_compressed(
        npz_path,
        gallery_sample_ids=np.asarray(
            [row["sample_id"] for row in gallery_records], dtype="U128"
        ),
        gallery_vectors=gallery_matrix,
        gallery_valid=gallery_valid,
        query_sample_ids=np.asarray(
            [row["sample_id"] for row in query_records], dtype="U128"
        ),
        A_original=a_matrix,
        A_original_valid=a_valid,
        B_all_superres=b_matrix,
        B_all_superres_valid=b_valid,
        C_gated_superres=c_matrix,
        C_gated_superres_valid=c_valid,
    )
    cache = {
        "schema_version": 2,
        "kind": "checkin_superres_embedding_cache",
        "manifest_id": payload["manifest_id"],
        "prepare_config_hash": payload["prepare_config_hash"],
        "model_provenance_hash": payload["model_provenance_hash"],
        "evaluation_model_provenance": _model_provenance(settings),
        "npz_path": _relative(npz_path, artifact_dir),
        "npz_sha256": file_sha256(npz_path),
        "gallery_records": gallery_records,
        "query_records": query_records,
        "runtime": {
            "superres_requests": superres_requests,
            "gfpgan_calls": gfpgan_invocations,
            "gfpgan_outputs": gfpgan_outputs,
            "gfpgan_no_output": gfpgan_no_output,
            "startup_blocked": startup_blocked,
            "enhanced_embedding_success": enhanced_embedding_success,
            "enhanced_embedding_failure": enhanced_embedding_failure,
            "gfpgan_seconds": round(gfpgan_seconds, 6),
            "gfpgan_mean_seconds": (
                round(gfpgan_seconds / gfpgan_invocations, 6)
                if gfpgan_invocations
                else None
            ),
            "superres_startup_error": startup_error,
        },
    }
    cache["evaluation_model_provenance_hash"] = stable_hash(
        cache["evaluation_model_provenance"]
    )
    cache_path = artifact_dir / "embedding_cache.json"
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return gallery_records, query_records, cache


def _load_embedding_cache(
    payload: dict,
    artifact_dir: Path,
    settings,
) -> tuple[list[dict], list[dict], dict]:
    cache_path = artifact_dir / "embedding_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("schema_version") != 2:
        raise RuntimeError(
            "embedding cache schema已过期；请使用--force-recompute重建"
        )
    for key in ("manifest_id", "prepare_config_hash", "model_provenance_hash"):
        if cache.get(key) != payload.get(key):
            raise RuntimeError(f"embedding cache provenance不匹配：{key}")
    if stable_hash(cache["evaluation_model_provenance"]) != cache.get(
        "evaluation_model_provenance_hash"
    ):
        raise RuntimeError("evaluation model provenance hash不匹配")
    if _model_provenance(settings) != cache["evaluation_model_provenance"]:
        raise RuntimeError("运行时模型文件与embedding cache provenance不匹配")
    npz_path = _resolve(cache["npz_path"], artifact_dir)
    assert npz_path is not None
    if file_sha256(npz_path) != cache["npz_sha256"]:
        raise RuntimeError("embedding NPZ hash不匹配")
    arrays = np.load(npz_path, allow_pickle=False)
    expected_gallery = [row["sample_id"] for row in cache["gallery_records"]]
    expected_query = [row["sample_id"] for row in cache["query_records"]]
    manifest_gallery = [row["sample_id"] for row in payload["gallery"]]
    manifest_queries = [row["sample_id"] for row in payload["queries"]]
    if expected_gallery != manifest_gallery or expected_query != manifest_queries:
        raise RuntimeError("embedding cache样本顺序与manifest不匹配")
    for cached, frozen in zip(cache["gallery_records"], payload["gallery"]):
        for key in ("sample_id", "pid", "aligned_sha256", "source_sha256"):
            if cached.get(key) != frozen.get(key):
                raise RuntimeError(f"Gallery cache字段与manifest不匹配：{key}")
    for cached, frozen in zip(cache["query_records"], payload["queries"]):
        for key in (
            "sample_id",
            "pid",
            "track",
            "eligibility",
            "face_best_frame_index",
            "aligned_sha256",
            "source_sha256",
            "quality",
        ):
            if cached.get(key) != frozen.get(key):
                raise RuntimeError(f"Query cache字段与manifest不匹配：{key}")
    if arrays["gallery_sample_ids"].tolist() != expected_gallery:
        raise RuntimeError("Gallery embedding顺序不匹配")
    if arrays["query_sample_ids"].tolist() != expected_query:
        raise RuntimeError("Query embedding顺序不匹配")
    unpacked_gallery = _unpack_vectors(
        arrays["gallery_vectors"], arrays["gallery_valid"]
    )
    for index, row in enumerate(cache["gallery_records"]):
        row["vectors"] = {
            "A_original": unpacked_gallery[index]
        }
    unpacked = {
        arm: _unpack_vectors(arrays[arm], arrays[f"{arm}_valid"])
        for arm in ARMS
    }
    frozen_poor_threshold = float(
        payload["prepare_config"]["face_fiqa_poor_thresh"]
    )
    for index, row in enumerate(cache["query_records"]):
        original = unpacked["A_original"][index]
        superres = unpacked["B_all_superres"][index]
        if row.get("restoration_output"):
            expected_post_accepted, expected_post_reason = superres_quality_ok(
                row.get("fiqa_after"),
                poor_threshold=frozen_poor_threshold,
            )
        else:
            expected_post_accepted, expected_post_reason = False, None
        if bool(row.get("post_superres_accepted")) != expected_post_accepted:
            raise RuntimeError(
                "cache FIQA诊断不等于manifest冻结阈值的派生结果"
            )
        if row.get("restoration_output") and row.get(
            "post_superres_reason"
        ) != expected_post_reason:
            raise RuntimeError("cache FIQA诊断原因与冻结阈值不匹配")
        derived = select_arm_embeddings(
            original,
            superres,
            eligibility=payload["queries"][index].get("eligibility", "none"),
            superres_succeeded=bool(row.get("superres_succeeded")),
            post_superres_accepted=expected_post_accepted,
        )
        persisted_c = unpacked["C_gated_superres"][index]
        derived_c = derived["C_gated_superres"]
        if (persisted_c is None) != (derived_c is None) or (
            persisted_c is not None
            and derived_c is not None
            and not np.array_equal(persisted_c, derived_c)
        ):
            raise RuntimeError("持久化C向量不等于由缓存A/B及固定门控派生的C")
        row["vectors"] = derived
        for path_key, hash_key in (
            ("original_aligned_path", "original_aligned_sha256"),
            ("superres_aligned_path", "superres_aligned_sha256"),
        ):
            path = _resolve(row.get(path_key), artifact_dir)
            if path is not None and file_sha256(path) != row.get(hash_key):
                raise RuntimeError(f"cache图像hash不匹配：{path}")
    return cache["gallery_records"], cache["query_records"], cache
