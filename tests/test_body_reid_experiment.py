from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiment" / "body_reid"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from body_reid_experiment.common import (  # noqa: E402
    GALLERY_ROLES,
    QUERY_ROLES,
    load_protocol,
    load_samples,
    parse_bool,
    stable_hash,
)
from body_reid_experiment.protocol import (  # noqa: E402
    FreezeConfig,
    freeze_protocol,
)
from body_reid_experiment import runner  # noqa: E402
from body_reid_experiment.runner import (  # noqa: E402
    _describe_asset,
    _embedding_cache_key,
    _evaluation_config,
    _resolve_device_request,
    calibrate_thresholds,
    evaluate_protocol,
)
from body_reid_experiment.summary import build_summary  # noqa: E402


def _image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 256, size=(128, 64, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values, mode="RGB").save(path, quality=95)


def _write_split(
    root: Path,
    *,
    split: str,
    person_ids: tuple[int, ...],
) -> tuple[list[str], list[str], list[int]]:
    names = []
    info = []
    query_indices = []
    global_index = 0
    track_index = 0
    for person_id in person_ids:
        for local_track in range(2):
            outfit = (
                2
                if split == "test"
                and local_track == 1
                and person_id in {203, 204}
                else 1
            )
            camera = local_track + 1
            start = global_index
            for frame in range(3):
                name = (
                    f"{person_id:04d}O{outfit}C{camera}"
                    f"T{track_index:03d}F{frame:03d}.jpg"
                )
                names.append(name)
                _image(
                    root
                    / f"bbox_{split}"
                    / f"{person_id:04d}"
                    / name,
                    seed=person_id * 100 + local_track * 10 + frame,
                )
                global_index += 1
            info.append(
                f"{start} {global_index - 1} "
                f"{person_id} {outfit} {camera}"
            )
            if split == "test" and local_track == 1:
                query_indices.append(track_index)
            track_index += 1
    return names, info, query_indices


def _dataset(root: Path) -> Path:
    annotation = root / "annotation" / "mevid-v1-annotation-data"
    annotation.mkdir(parents=True)
    test_names, test_info, query_indices = _write_split(
        root,
        split="test",
        person_ids=(201, 202, 203, 204),
    )
    train_names, train_info, _ = _write_split(
        root,
        split="train",
        person_ids=(1, 2, 3, 4),
    )
    (annotation / "test_name.txt").write_text(
        "\n".join(test_names) + "\n",
        encoding="utf-8",
    )
    (annotation / "track_test_info.txt").write_text(
        "\n".join(test_info) + "\n",
        encoding="utf-8",
    )
    (annotation / "query_IDX.txt").write_text(
        "\n".join(str(value) for value in query_indices) + "\n",
        encoding="utf-8",
    )
    (annotation / "train_name.txt").write_text(
        "\n".join(train_names) + "\n",
        encoding="utf-8",
    )
    (annotation / "track_train_info.txt").write_text(
        "\n".join(train_info) + "\n",
        encoding="utf-8",
    )
    return root


def _freeze(tmp_path: Path, name: str = "one"):
    data_root = _dataset(tmp_path / "mevid")
    output = tmp_path / name
    config = FreezeConfig(
        seed=0,
        enroll_subjects=2,
        imposter_subjects=2,
        frames_per_track=2,
        max_gallery_tracks=1,
        max_query_tracks=1,
        calibration_enroll_subjects=2,
        calibration_imposter_subjects=2,
        calibration_gallery_tracks=1,
        calibration_query_tracks=1,
        expected_test_identities=4,
    )
    manifest = output / "frozen_body_samples.csv"
    protocol_path = output / "frozen_protocol.json"
    protocol = freeze_protocol(
        data_root,
        manifest,
        protocol_path,
        config,
    )
    return data_root, manifest, protocol_path, protocol


def test_freeze_protocol_is_deterministic_and_leak_free(tmp_path: Path) -> None:
    data_root, manifest, _, first = _freeze(tmp_path)
    second_manifest = tmp_path / "two" / "frozen_body_samples.csv"
    second_protocol = tmp_path / "two" / "frozen_protocol.json"
    config = FreezeConfig(
        seed=0,
        enroll_subjects=2,
        imposter_subjects=2,
        frames_per_track=2,
        max_gallery_tracks=1,
        max_query_tracks=1,
        calibration_enroll_subjects=2,
        calibration_imposter_subjects=2,
        calibration_gallery_tracks=1,
        calibration_query_tracks=1,
        expected_test_identities=4,
    )
    second = freeze_protocol(
        data_root,
        second_manifest,
        second_protocol,
        config,
    )

    assert first["protocol_id"] == second["protocol_id"]
    assert manifest.read_bytes() == second_manifest.read_bytes()
    assert first["test"]["enroll_pids"] == ["0203", "0201"]
    assert first["counts"]["p2_query_tracks"] == 1
    assert first["counts"]["p3_query_tracks"] == 1

    rows = load_samples(manifest)
    gallery_paths = {
        row["source_relpath"] for row in rows if row["role"] in GALLERY_ROLES
    }
    query_paths = {
        row["source_relpath"] for row in rows if row["role"] in QUERY_ROLES
    }
    assert gallery_paths.isdisjoint(query_paths)
    assert all(
        parse_bool(row["gallery_allowed"])
        for row in rows
        if row["role"] in GALLERY_ROLES
    )
    assert all("\\" not in row["source_relpath"] for row in rows)


def test_freeze_protocol_refuses_accidental_overwrite(tmp_path: Path) -> None:
    data_root, manifest, protocol_path, _ = _freeze(tmp_path)
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        freeze_protocol(
            data_root,
            manifest,
            protocol_path,
            FreezeConfig(
                enroll_subjects=2,
                imposter_subjects=2,
                frames_per_track=2,
                max_gallery_tracks=1,
                max_query_tracks=1,
                calibration_enroll_subjects=2,
                calibration_imposter_subjects=2,
                calibration_gallery_tracks=1,
                calibration_query_tracks=1,
                expected_test_identities=4,
            ),
        )


def test_load_protocol_rejects_tampered_identity(tmp_path: Path) -> None:
    _, _, protocol_path, _ = _freeze(tmp_path)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    payload["test"]["p2_cross_outfit_query_tracks"].append(999)
    protocol_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="protocol_id"):
        load_protocol(protocol_path)


def _vector_map(rows: list[dict], protocol: dict) -> dict[str, np.ndarray]:
    test_enroll = {
        person_id: index
        for index, person_id in enumerate(protocol["test"]["enroll_pids"])
    }
    calibration_enroll = {
        person_id: index
        for index, person_id in enumerate(
            protocol["calibration"]["enroll_pids"]
        )
    }
    vectors = {}
    for row in rows:
        role = row["role"]
        if role in GALLERY_ROLES and not parse_bool(row["gallery_allowed"]):
            continue
        if role not in GALLERY_ROLES | QUERY_ROLES:
            continue
        if role == "enroll_gallery":
            index = test_enroll[row["person_id"]]
            vector = np.asarray([1.0, 0.0] if index == 0 else [0.0, 1.0])
        elif role == "genuine_query":
            index = test_enroll[row["person_id"]]
            vector = np.asarray([1.0, 0.0] if index == 0 else [0.0, 1.0])
        elif role == "imposter_query":
            vector = np.asarray([0.95, np.sqrt(1.0 - 0.95**2)])
        elif role == "calibration_gallery":
            index = calibration_enroll[row["person_id"]]
            vector = np.asarray([1.0, 0.0] if index == 0 else [0.0, 1.0])
        elif role == "calibration_genuine_query":
            index = calibration_enroll[row["person_id"]]
            vector = np.asarray(
                [0.9, np.sqrt(1.0 - 0.9**2)]
                if index == 0
                else [np.sqrt(1.0 - 0.9**2), 0.9]
            )
        else:
            vector = np.asarray([0.8, 0.6])
        vectors[row["sample_id"]] = vector.astype(np.float32)
    return vectors


def test_evaluation_uses_train_threshold_without_test_retuning(
    tmp_path: Path,
) -> None:
    _, manifest, _, protocol = _freeze(tmp_path)
    rows = load_samples(manifest)
    result = evaluate_protocol(
        rows,
        _vector_map(rows, protocol),
        protocol,
        targets=(0.1,),
    )

    assert result["thresholds"]["fmr_0.1"] == pytest.approx(0.9)
    test_point = result["protocols"]["P1_overall"]["open_set"][
        "operating_points"
    ]["fmr_0.1"]
    assert test_point["fmr"] == 1.0
    assert (
        result["protocols"]["P1_overall"]["closed_set"]["rank1_rate"]
        == 1.0
    )
    assert (
        result["protocols"]["P2_cross_outfit"]["closed_set"]["queries"]
        == 1
    )
    assert (
        result["protocols"]["P3_same_outfit_cross_camera"]["closed_set"][
            "queries"
        ]
        == 1
    )


def test_reject_all_threshold_survives_json_roundtrip() -> None:
    rows = [
        {"genuine": True, "rank1_correct": True, "confidence": 0.5},
        {"genuine": False, "rank1_correct": False, "confidence": 0.9},
    ]
    thresholds, calibration = calibrate_thresholds(rows, targets=(0.1,))
    serialized = json.loads(json.dumps(thresholds))

    assert serialized["fmr_0.1"] > 0.9
    assert (
        calibration["operating_points"]["fmr_0.1"]["threshold"] > 0.9
    )


def test_cache_and_evaluation_identities_cover_their_inputs() -> None:
    common = {
        "protocol_id": "protocol",
        "backend": "osnet",
        "device_request": "cpu",
        "actual_device": "cpu",
        "provenance_hash": "model",
        "sample_ids": ["sample"],
    }
    first = _embedding_cache_key(manifest_sha256="one", **common)
    second = _embedding_cache_key(manifest_sha256="two", **common)

    assert first != second
    assert _evaluation_config((0.01,))["sha256"] != _evaluation_config(
        (0.05,)
    )["sha256"]


def test_explicit_cuda_request_fails_when_cuda_is_unavailable() -> None:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Torch:
        cuda = _Cuda()

    with pytest.raises(RuntimeError, match="不可用CUDA"):
        _resolve_device_request("cuda", _Torch())
    assert _resolve_device_request("auto", _Torch()) == "cpu"


def test_asset_hash_ignores_generated_bytecode(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    first = _describe_asset(source, tmp_path)
    bytecode = source / "__pycache__" / "model.cpython-312.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"first")
    second = _describe_asset(source, tmp_path)
    bytecode.write_bytes(b"second")
    third = _describe_asset(source, tmp_path)

    assert first["sha256"] == second["sha256"] == third["sha256"]
    assert first["file_count"] == second["file_count"] == 1


def _identity_payload(identity: dict) -> dict:
    return {"sha256": stable_hash(identity), **identity}


def _backend_result(backend: str, evaluation: dict, protocol: dict) -> dict:
    environment = _identity_payload(
        {
            "experiment_code_sha256": "code",
            "device_request": "cpu",
            "actual_device": "cpu",
            "runtime": {"environment": "test"},
        }
    )
    return {
        "schema_version": 1,
        "kind": "body_reid_backend_result",
        "protocol_id": protocol["protocol_id"],
        "manifest_sha256": protocol["sample_manifest"]["sha256"],
        "backend": backend,
        "active_backend": backend,
        "device_request": "cpu",
        "actual_device": "cpu",
        "evaluation_config": _evaluation_config((0.1,)),
        "comparison_environment": environment,
        "embedding_dimension": 2,
        "performance": {
            "latency_ms_mean": 1.0,
            "latency_ms_p95": 2.0,
            "throughput_images_per_second": 100.0,
            "model_load_seconds": 0.5,
            "cuda_memory_after_load_bytes": 0,
            "peak_cuda_memory_bytes": 0,
        },
        "provenance": {"model_weight_bytes": 10},
        **copy.deepcopy(evaluation),
    }


def test_summary_reports_paired_rescue_without_changing_query_universe(
    tmp_path: Path,
) -> None:
    _, manifest, _, protocol = _freeze(tmp_path)
    rows = load_samples(manifest)
    evaluation = evaluate_protocol(
        rows,
        _vector_map(rows, protocol),
        protocol,
        targets=(0.1,),
    )
    baseline = _backend_result("osnet", evaluation, protocol)
    candidate = _backend_result("differ", evaluation, protocol)
    genuine = next(row for row in baseline["queries"] if row["genuine"])
    genuine["rank1_correct"] = False
    genuine["rank5_correct"] = False
    genuine["rank"] = None
    genuine["average_precision"] = None

    summary = build_summary(
        protocol,
        {"osnet": baseline, "differ": candidate},
    )

    comparison = summary["paired_vs_baseline"]["differ"]["P1_overall"]
    assert comparison["rescued_count"] == 1
    assert comparison["harmed_count"] == 0
    assert summary["query_universe"]["all"] == len(evaluation["queries"])


def test_summary_rejects_incomparable_environments(tmp_path: Path) -> None:
    _, manifest, _, protocol = _freeze(tmp_path)
    rows = load_samples(manifest)
    evaluation = evaluate_protocol(
        rows,
        _vector_map(rows, protocol),
        protocol,
        targets=(0.1,),
    )
    baseline = _backend_result("osnet", evaluation, protocol)
    candidate = _backend_result("differ", evaluation, protocol)
    candidate["comparison_environment"] = _identity_payload(
        {
            "experiment_code_sha256": "different-code",
            "device_request": "cpu",
            "actual_device": "cpu",
            "runtime": {"environment": "test"},
        }
    )

    with pytest.raises(ValueError, match="运行环境或实验代码"):
        build_summary(protocol, {"osnet": baseline, "differ": candidate})


def test_existing_result_is_not_reused_for_new_fmr_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, manifest, _, protocol = _freeze(tmp_path)
    rows = load_samples(manifest)
    provenance = {
        "provenance_hash": "model",
        "model_weight_bytes": 0,
        "experiment_code": {"sha256": "code"},
        "runtime": {
            "python": "test",
            "platform": "test",
            "packages": {},
            "cuda_available": False,
            "cuda_version": None,
            "gpu": None,
            "git": {},
        },
    }
    sample_ids = [
        row["sample_id"]
        for row in rows
        if row["role"] in QUERY_ROLES
        or (
            row["role"] in GALLERY_ROLES
            and parse_bool(row["gallery_allowed"])
        )
    ]
    cache_key = _embedding_cache_key(
        protocol_id=protocol["protocol_id"],
        manifest_sha256=protocol["sample_manifest"]["sha256"],
        backend="osnet",
        device_request="cpu",
        actual_device="cpu",
        provenance_hash="model",
        sample_ids=sample_ids,
    )
    performance = {
        "actual_device": "cpu",
        "latency_ms_mean": 1.0,
        "latency_ms_p50": 1.0,
        "latency_ms_p95": 1.0,
        "throughput_images_per_second": 1.0,
        "model_load_seconds": 1.0,
        "cuda_memory_after_load_bytes": 0,
        "peak_cuda_memory_bytes": 0,
    }
    monkeypatch.setattr(
        runner,
        "collect_backend_provenance",
        lambda backend, settings: copy.deepcopy(provenance),
    )
    monkeypatch.setattr(
        runner,
        "_load_or_extract",
        lambda **kwargs: ({}, performance, 2, cache_key, False),
    )
    monkeypatch.setattr(
        runner,
        "evaluate_protocol",
        lambda rows, vectors, protocol, targets: {},
    )
    output = tmp_path / "results"
    cache = tmp_path / "cache"
    runner.run_backend(
        rows=rows,
        protocol=protocol,
        data_root=data_root,
        backend="osnet",
        device="cpu",
        output_dir=output,
        cache_dir=cache,
        targets=(0.01,),
        force=False,
    )

    with pytest.raises(FileExistsError, match="结果已存在"):
        runner.run_backend(
            rows=rows,
            protocol=protocol,
            data_root=data_root,
            backend="osnet",
            device="cpu",
            output_dir=output,
            cache_dir=cache,
            targets=(0.05,),
            force=False,
        )
