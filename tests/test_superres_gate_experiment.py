from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

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


def test_select_product_best_frame_uses_body_quality_score() -> None:
    tracklet = MODULE.common.Tracklet(
        pid="0001",
        cam=1,
        outfit=1,
        track=7,
        frames=[Path("frame_a.png"), Path("frame_b.png"), Path("frame_c.png")],
    )
    quality_by_path = {
        Path("frame_a.png"): {"blur_var": 100.0, "area": 1000.0, "quality": 0.99},
        Path("frame_b.png"): {"blur_var": 50.0, "area": 20000.0, "quality": 0.01},
        Path("frame_c.png"): {"blur_var": 90.0, "area": 5000.0, "quality": 0.50},
    }

    chosen = MODULE.select_product_best_frame(
        tracklet,
        frames_per_track=3,
        image_loader=lambda path: path,
        quality_fn=lambda image: quality_by_path[image],
    )

    assert chosen["path"] == Path("frame_b.png")
    assert chosen["best_idx"] == 1
    assert chosen["candidate_count"] == 3
    assert chosen["score"] == pytest.approx(50.0)


def test_select_product_best_frame_scans_full_track_by_default() -> None:
    frames = [Path(f"frame_{index}.png") for index in range(9)]
    tracklet = MODULE.common.Tracklet(
        pid="0001",
        cam=1,
        outfit=1,
        track=8,
        frames=frames,
    )

    chosen = MODULE.select_product_best_frame(
        tracklet,
        frames_per_track=0,
        image_loader=lambda path: path,
        quality_fn=lambda image: {
            "blur_var": 1000.0 if image == frames[-1] else 1.0,
            "area": 20000.0,
        },
    )

    assert chosen["path"] == frames[-1]
    assert chosen["best_idx"] == 8
    assert chosen["candidate_count"] == 9


def test_stratified_train_selection_limits_identities_and_tracks() -> None:
    tracklets = [
        MODULE.common.Tracklet(
            pid=f"{pid:04d}",
            cam=track % 3,
            outfit=track % 2,
            track=track,
            frames=[Path(f"{pid}_{track}.png")],
        )
        for pid in range(10)
        for track in range(6)
    ]

    selected = MODULE.select_stratified_train_tracklets(
        tracklets,
        max_identities=4,
        tracks_per_identity=3,
        seed=7,
    )

    grouped = {}
    for tracklet in selected:
        grouped.setdefault(tracklet.pid, []).append(tracklet)
    assert len(grouped) == 4
    assert all(len(rows) == 3 for rows in grouped.values())
    assert selected == MODULE.select_stratified_train_tracklets(
        tracklets,
        max_identities=4,
        tracks_per_identity=3,
        seed=7,
    )


def test_calibrate_fiqa_thresholds_picks_max_coverage_thresholds() -> None:
    rows = [
        {"pid": "0001", "fiqa": 0.10, "usable": False},
        {"pid": "0001", "fiqa": 0.20, "usable": False},
        {"pid": "0002", "fiqa": 0.30, "usable": True},
        {"pid": "0002", "fiqa": 0.70, "usable": True},
        {"pid": "0003", "fiqa": 0.80, "usable": True},
        {"pid": "0003", "fiqa": 0.90, "usable": True},
    ]

    result = MODULE.calibrate_fiqa_thresholds(rows)

    assert result["poor"]["threshold"] == pytest.approx(0.25)
    assert result["poor"]["precision"] == pytest.approx(1.0)
    assert result["poor"]["coverage"] == pytest.approx(2 / 6)
    assert result["clear"]["threshold"] == pytest.approx(0.3)
    assert result["clear"]["precision"] == pytest.approx(1.0)
    assert result["clear"]["coverage"] == pytest.approx(4 / 6)


def test_calibrate_fiqa_thresholds_raises_when_poor_and_clear_cannot_separate() -> None:
    rows = [
        {"pid": "0001", "fiqa": 0.10, "usable": False},
        {"pid": "0002", "fiqa": 0.10, "usable": False},
        {"pid": "0003", "fiqa": 0.10, "usable": True},
        {"pid": "0004", "fiqa": 0.90, "usable": True},
    ]

    with pytest.raises(ValueError, match="无法分开"):
        MODULE.calibrate_fiqa_thresholds(rows, poor_precision=0.60, clear_precision=0.50)


def test_fiqa_diagnostics_are_saved_without_a_valid_90_percent_threshold() -> None:
    rows = [
        {"pid": "0001", "track": 1, "fiqa": 0.1, "usable": False},
        {"pid": "0002", "track": 2, "fiqa": 0.2, "usable": True},
        {"pid": "0003", "track": 3, "fiqa": 0.8, "usable": False},
        {"pid": "0004", "track": 4, "fiqa": 0.9, "usable": True},
    ]

    result = MODULE.summarize_fiqa_calibration(rows)

    assert result["total_rows"] == 4
    assert result["overall_usable_rate"] == pytest.approx(0.5)
    assert result["clear_frontier"]
    assert result["reject_curve"][0]["kept"] == 4
    assert len(result["rows"]) == 4


def test_prepare_rejects_calibration_from_different_fiqa_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.config import settings

    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        """
        {
          "kind": "fiqa_calibration",
          "calibration_config": {
            "face_fiqa_backend": "cr_fiqa",
            "face_fiqa_arch": "iresnet50"
          },
          "thresholds": {
            "poor": {"threshold": 0.2},
            "clear": {"threshold": 0.8}
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "face_fiqa_backend", "off")
    args = type(
        "Args",
        (),
        {
            "data": str(tmp_path),
            "manifest": str(tmp_path / "manifest.json"),
            "calibration": str(calibration),
            "cache": str(tmp_path / "cache"),
            "max_tracklets": 0,
            "frames_per_track": 0,
        },
    )()

    with pytest.raises(ValueError, match="FIQA后端与Train校准不一致"):
        MODULE.prepare_manifest(args)


def test_evaluate_no_longer_uses_track_aggregation_helpers() -> None:
    assert not hasattr(MODULE, "_aggregate_track")
    assert not hasattr(MODULE, "_best_quality")


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


@pytest.mark.parametrize("subcommand", ["calibrate", "prepare", "evaluate"])
def test_cli_help_subcommands_load(subcommand: str, capsys: pytest.CaptureFixture[str]) -> None:
    parser = MODULE.build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args([subcommand, "--help"])

    assert excinfo.value.code == 0
    assert subcommand in capsys.readouterr().out
