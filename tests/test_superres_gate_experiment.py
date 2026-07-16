from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiment"
    / "糊脸消融实验"
    / "超分实验"
    / "scripts"
    / "run_superres_gate.py"
)
SPEC = importlib.util.spec_from_file_location("run_superres_gate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_superres_variants_share_fixed_inputs() -> None:
    original = np.asarray([1.0, 0.0], dtype=np.float32)
    enhanced = np.asarray([0.0, 1.0], dtype=np.float32)

    gated = MODULE.select_variant_embeddings(original, enhanced, True)
    not_gated = MODULE.select_variant_embeddings(original, enhanced, False)
    failed = MODULE.select_variant_embeddings(original, None, True)

    assert gated["A_original"] is original
    assert gated["B_all_superres"] is enhanced
    assert gated["C_gated_superres"] is enhanced
    assert not_gated["C_gated_superres"] is original
    assert failed["B_all_superres"] is original
    assert failed["C_gated_superres"] is original


def test_paired_transition_stats_reports_both_directions() -> None:
    baseline = [
        {"sample_id": "improve", "gt": "1", "pred": "2", "genuine": True, "confidence": 0.8},
        {"sample_id": "degrade", "gt": "2", "pred": "2", "genuine": True, "confidence": 0.8},
        {"sample_id": "imposter", "gt": "9", "pred": None, "genuine": False, "confidence": 0.2},
    ]
    candidate = [
        {"sample_id": "improve", "gt": "1", "pred": "1", "genuine": True, "confidence": 0.8},
        {"sample_id": "degrade", "gt": "2", "pred": "3", "genuine": True, "confidence": 0.8},
        {"sample_id": "imposter", "gt": "9", "pred": "1", "genuine": False, "confidence": 0.7},
    ]

    result = MODULE.paired_transition_stats(
        baseline,
        candidate,
        {"0.050": 0.5},
    )

    assert result["prediction_changed"] == 3
    assert result["rank1_improved"] == 1
    assert result["rank1_degraded"] == 1
    assert result["imposter_false_accept_transitions"]["0.050"]["increased"] == 1
