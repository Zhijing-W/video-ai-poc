from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from app.identity.evidence_selection import face_evidence_rank
from app.identity.face.quality import superres_quality_ok


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiment"
    / "糊脸消融实验"
    / "超分实验"
    / "scripts"
    / "run_checkin_superres_abc.py"
)
SPEC = importlib.util.spec_from_file_location("run_checkin_superres_abc", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_checkin_prefix_and_front_parsing_are_case_insensitive() -> None:
    assert MODULE.parse_checkin_image(
        Path("205-20180514-521-001-F.jpg")
    ) == ("0205", "F")
    assert MODULE.parse_checkin_image(
        Path("7-anything-f.PNG")
    ) == ("0007", "F")
    assert MODULE.parse_checkin_image(
        Path("205-20180514-521-001-B.jpg")
    ) == ("0205", "B")
    assert MODULE.parse_checkin_image(Path("not-a-checkin.jpg")) is None


def test_prefix_coverage_audits_all_splits_and_requires_query() -> None:
    coverage = MODULE.audit_prefix_coverage(
        {"0001", "0002", "9999"},
        {"0001", "0003"},
        {"0001", "0002"},
        {"0002"},
    )
    assert coverage["missing_train_pids"] == ["0003"]
    assert coverage["missing_test_pids"] == []
    assert coverage["extra_checkin_prefixes"] == ["9999"]

    with pytest.raises(RuntimeError, match="Query PID"):
        MODULE.audit_prefix_coverage(
            {"0001"},
            {"0001"},
            {"0001", "0002"},
            {"0002"},
        )


def test_product_face_rank_is_independent_of_body_best_and_proxy() -> None:
    body_best = {
        "frame_index": 0,
        "proxy_score": 9999.0,
        "_face": {
            "det_score": 0.99,
            "quality": {
                "eligibility": "recoverable",
                "category": "poor",
                "quality": 0.2,
            },
        },
    }
    face_best = {
        "frame_index": 3,
        "proxy_score": 1.0,
        "_face": {
            "det_score": 0.8,
            "quality": {
                "eligibility": "direct",
                "category": "clear",
                "quality": 0.8,
            },
        },
    }
    assert max([body_best, face_best], key=face_evidence_rank) is face_best


def test_arm_semantics_have_no_b_fallback_and_c_reuses_cache() -> None:
    original = np.asarray([1.0, 0.0], dtype=np.float32)
    restored = np.asarray([0.0, 1.0], dtype=np.float32)

    failed = MODULE.select_arm_embeddings(
        original,
        None,
        eligibility="recoverable",
        superres_succeeded=False,
        post_superres_accepted=False,
    )
    assert failed["A_original"] is original
    assert failed["B_all_superres"] is None
    assert failed["C_gated_superres"] is None

    direct = MODULE.select_arm_embeddings(
        original,
        restored,
        eligibility="direct",
        superres_succeeded=True,
        post_superres_accepted=True,
    )
    assert direct["B_all_superres"] is restored
    assert direct["C_gated_superres"] is original

    recoverable = MODULE.select_arm_embeddings(
        original,
        restored,
        eligibility="recoverable",
        superres_succeeded=True,
        post_superres_accepted=True,
    )
    assert recoverable["B_all_superres"] is restored
    assert recoverable["C_gated_superres"] is restored

    rejected = MODULE.select_arm_embeddings(
        original,
        restored,
        eligibility="recoverable",
        superres_succeeded=True,
        post_superres_accepted=False,
    )
    assert rejected["B_all_superres"] is restored
    assert rejected["C_gated_superres"] is restored

    unusable = MODULE.select_arm_embeddings(
        original,
        restored,
        eligibility="unusable",
        superres_succeeded=True,
        post_superres_accepted=True,
    )
    assert unusable["B_all_superres"] is restored
    assert unusable["C_gated_superres"] is None


def test_image_manifest_retains_every_query_and_non_processed_rows() -> None:
    queries = [
        {
            "sample_id": "aligned",
            "pid": "0001",
            "track": 1,
            "eligibility": "recoverable",
            "face_best_frame_index": 7,
            "face_status": "detected",
            "aligned_path": "aligned.png",
        },
        {
            "sample_id": "none",
            "pid": "0002",
            "track": 2,
            "eligibility": "none",
            "face_best_frame_index": None,
            "face_status": "none",
            "aligned_path": None,
        },
    ]
    rows = MODULE.build_image_manifest_records(
        queries,
        {
            "aligned": {
                "status": "processed",
                "comparison_path": "comparisons/aligned.jpg",
                "original_aligned_path": "aligned_original/aligned.png",
                "superres_aligned_path": "aligned_superres/aligned.png",
            }
        },
    )
    assert [row["sample_id"] for row in rows] == ["aligned", "none"]
    assert rows[0]["status"] == "processed"
    assert rows[0]["comparison_path"] == "comparisons/aligned.jpg"
    assert rows[1]["status"] == "non_processed"
    assert rows[1]["superres_aligned_path"] is None


def test_manifest_and_config_hashing_is_deterministic() -> None:
    left = {"z": [3, 2, 1], "a": {"threshold": 0.45, "enabled": True}}
    right = {"a": {"enabled": True, "threshold": 0.45}, "z": [3, 2, 1]}
    assert MODULE.canonical_json(left) == MODULE.canonical_json(right)
    assert MODULE.stable_hash(left) == MODULE.stable_hash(right)
    assert MODULE.stable_hash(left) != MODULE.stable_hash({**left, "new": 1})


def test_manifest_identity_hash_covers_quality_and_candidate_provenance() -> None:
    gallery = [{"sample_id": "g", "pid": "0001", "quality": {"category": "clear"}}]
    queries = [
        {
            "sample_id": "q",
            "pid": "0001",
            "quality": {"category": "clear", "eligibility": "direct"},
            "candidate_provenance": {"bounded": [{"frame_index": 3}]},
        }
    ]
    original = MODULE.manifest_identity({}, "config", "models", gallery, queries)
    changed_quality = MODULE.manifest_identity(
        {},
        "config",
        "models",
        gallery,
        [{**queries[0], "quality": {"category": "poor", "eligibility": "unusable"}}],
    )
    changed_candidates = MODULE.manifest_identity(
        {},
        "config",
        "models",
        gallery,
        [{**queries[0], "candidate_provenance": {"bounded": [{"frame_index": 9}]}}],
    )

    assert MODULE.stable_hash(original) != MODULE.stable_hash(changed_quality)
    assert MODULE.stable_hash(original) != MODULE.stable_hash(changed_candidates)


def test_post_superres_fiqa_is_diagnostic_only() -> None:
    assert superres_quality_ok(None, poor_threshold=0.3) == (
        False,
        "fiqa_unavailable_or_nonfinite",
    )
    assert superres_quality_ok(float("nan"), poor_threshold=0.3) == (
        False,
        "fiqa_unavailable_or_nonfinite",
    )
    assert superres_quality_ok(0.4, poor_threshold=0.3) == (True, None)
    assert superres_quality_ok(0.4, poor_threshold=0.5) == (
        False,
        "fiqa_below_poor_threshold",
    )


def test_recoverable_only_successful_accepted_cohort_has_identical_b_and_c() -> None:
    original = np.asarray([1.0, 0.0], dtype=np.float32)
    restored = np.asarray([0.8, 0.2], dtype=np.float32)
    rows = [
        MODULE.select_arm_embeddings(
            original,
            restored,
            eligibility="recoverable",
            superres_succeeded=True,
            post_superres_accepted=True,
        )
        for _ in range(4)
    ]
    assert all(
        np.array_equal(row["B_all_superres"], row["C_gated_superres"])
        for row in rows
    )


def test_compatibility_facade_reexports_and_cli_help(capsys) -> None:
    required_symbols = (
        "SCHEMA_VERSION",
        "ARMS",
        "canonical_json",
        "stable_hash",
        "manifest_identity",
        "parse_checkin_image",
        "load_checkin_front_images",
        "audit_prefix_coverage",
        "sample_evenly_indexed",
        "select_arm_embeddings",
        "build_image_manifest_records",
        "paired_uncertainty",
        "build_parser",
        "main",
        "prepare",
        "evaluate",
    )
    assert all(hasattr(MODULE, name) for name in required_symbols)

    from checkin_superres.common import stable_hash as package_stable_hash
    from checkin_superres.orchestration import (
        evaluate as package_evaluate,
        prepare as package_prepare,
    )

    assert MODULE.stable_hash is package_stable_hash
    assert MODULE.prepare is package_prepare
    assert MODULE.evaluate is package_evaluate

    with pytest.raises(SystemExit) as exit_info:
        MODULE.main(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "prepare" in help_text
    assert "evaluate" in help_text
