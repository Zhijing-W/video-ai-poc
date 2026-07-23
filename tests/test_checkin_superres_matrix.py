from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SUPERRES_DIR = (
    ROOT / "experiment" / "糊脸消融实验" / "超分实验"
)
EXPERIMENT_DIR = SUPERRES_DIR.parent
for import_root in (ROOT, EXPERIMENT_DIR, SUPERRES_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from checkin_superres.common import file_sha256
from checkin_superres.matrix import (
    BACKEND_SPECS,
    FROZEN_MANIFEST_ID,
    _configured_weight,
    build_matrix_run_spec,
    derive_backend_arms,
    evaluate_matrix_payload,
    matrix_cache_key,
    normalize112,
)
from checkin_superres.embeddings import _normalise
from checkin_superres.metrics import _score, paired_uncertainty
from checkin_superres.orchestration import build_parser


class FakeFace:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    def _ensure_superres(self, backend: str):
        self.loaded.append(backend)
        return object()

    def superres_error(self, backend: str):
        return None

    def enhance(self, image: Image.Image, *, aligned: bool, backend: str):
        assert aligned and image.size == (112, 112)
        if backend == "codeformer":
            return image.resize((512, 512), Image.Resampling.BILINEAR)
        if backend == "realesrgan_x2plus":
            return image.resize((224, 224), Image.Resampling.BICUBIC)
        array = np.asarray(
            image.resize((512, 512), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        ).copy()
        array[0, 0, 0] = (int(array[0, 0, 0]) + 1) % 256
        return Image.fromarray(array)

    def embed_aligned_face(self, bgr: np.ndarray, backend: str):
        assert backend == "arcface"
        assert bgr.shape == (112, 112, 3) and bgr.dtype == np.uint8
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0
        vector[1] = float(bgr.mean()) / 255.0
        return vector


def _settings(**overrides):
    values = {
        "face_model": "buffalo_l",
        "face_rec_backend": "arcface",
        "face_fiqa_backend": "off",
        "face_fiqa_arch": "iresnet50",
        "face_gfpgan_weights": "",
        "face_codeformer_weights": "",
        "face_codeformer_fidelity": 1.0,
        "face_realesrgan_x2plus_weights": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_rgb(path: Path, size: tuple[int, int], value: int) -> str:
    Image.new("RGB", size, (value, value, value)).save(path)
    return file_sha256(path)


def _fake_payload(root: Path) -> tuple[dict, Path]:
    gallery = root / "gallery.png"
    recoverable = root / "recoverable.png"
    direct = root / "direct.png"
    gallery_hash = _write_rgb(gallery, (112, 112), 60)
    recoverable_hash = _write_rgb(recoverable, (112, 112), 80)
    direct_hash = _write_rgb(direct, (96, 120), 70)
    payload = {
        "schema_version": 3,
        "manifest_id": "fake-manifest",
        "prepare_config": {
            "face_fiqa_poor_thresh": 0.3,
            "face_hit_thresh": 0.45,
        },
        "gallery": [
            {
                "sample_id": "g1",
                "pid": "0001",
                "aligned_path": gallery.name,
                "aligned_sha256": gallery_hash,
            }
        ],
        "queries": [
            {
                "sample_id": "q-recoverable",
                "pid": "0001",
                "track": 1,
                "aligned_path": recoverable.name,
                "aligned_sha256": recoverable_hash,
                "eligibility": "recoverable",
                "quality": {
                    "category": "poor",
                    "rule_quality": 0.2,
                    "fiqa": 0.2,
                },
            },
            {
                "sample_id": "q-direct",
                "pid": "0001",
                "track": 2,
                "aligned_path": direct.name,
                "aligned_sha256": direct_hash,
                "eligibility": "direct",
                "quality": {
                    "category": "clear",
                    "rule_quality": 0.8,
                    "fiqa": 0.8,
                },
            },
            {
                "sample_id": "q-none",
                "pid": "0002",
                "track": 3,
                "aligned_path": None,
                "aligned_sha256": None,
                "eligibility": "none",
                "face_status": "not_detected",
                "quality": {"category": "none"},
            },
        ],
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return payload, manifest


def test_normalize112_is_byte_preserving_at_112_and_lanczos_otherwise() -> None:
    array = np.arange(112 * 112 * 3, dtype=np.uint8).reshape(112, 112, 3)
    source = Image.fromarray(array)
    normalized = normalize112(source)
    assert normalized.size == (112, 112)
    assert np.array_equal(np.asarray(normalized), array)
    resized = normalize112(Image.new("RGB", (224, 224), "red"))
    assert resized.size == (112, 112)
    assert np.asarray(resized).dtype == np.uint8


def test_nonfinite_embeddings_are_rejected() -> None:
    assert _normalise(np.asarray([np.inf, 0.0], dtype=np.float32)) is None
    assert _normalise(np.asarray([np.nan, 1.0], dtype=np.float32)) is None


def test_run_spec_hash_is_canonical_and_backend_isolated() -> None:
    base = dict(
        manifest_id=FROZEN_MANIFEST_ID,
        source_hash="source",
        fidelity=None,
        transform="aligned",
        model_provenance={"weight": {"sha256": "abc"}},
    )
    gfpgan = build_matrix_run_spec(backend="gfpgan", **base)
    gfpgan_again = build_matrix_run_spec(backend="gfpgan", **base)
    codeformer = build_matrix_run_spec(
        backend="codeformer",
        **{**base, "fidelity": 1.0},
    )
    assert matrix_cache_key(gfpgan) == matrix_cache_key(gfpgan_again)
    assert matrix_cache_key(gfpgan) != matrix_cache_key(codeformer)
    assert gfpgan["normalization"]["name"] == "normalize112"


def test_matrix_arm_semantics_are_closed_and_have_no_fallback() -> None:
    original = np.asarray([1.0, 0.0])
    restored = np.asarray([0.0, 1.0])
    failed = derive_backend_arms(
        original,
        restored,
        eligibility="recoverable",
        transform_succeeded=False,
        b_arm="B",
        c_arm="C",
    )
    assert failed == {"B": None, "C": None}
    direct = derive_backend_arms(
        original,
        None,
        eligibility="direct",
        transform_succeeded=False,
        b_arm="B",
        c_arm="C",
    )
    assert direct["B"] is None and direct["C"] is original
    recoverable = derive_backend_arms(
        original,
        restored,
        eligibility="recoverable",
        transform_succeeded=True,
        b_arm="B",
        c_arm="C",
    )
    assert recoverable["B"] is restored and recoverable["C"] is restored


def test_template_missing_is_end_to_end_failure_and_score_has_diagnostics() -> None:
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    score = _score(vector, {"0001": vector}, "0002")
    assert not score["gt_template_available"]
    assert score["rank"] is None
    assert score["gt_cosine"] is None
    assert score["max_other_cosine"] == 1.0
    assert score["margin"] is None


def test_pid_cluster_uncertainty_uses_pid_as_resampling_unit() -> None:
    before = [
        {"sample_id": "a", "pid": "1", "rank1_correct": False, "pred": "x"},
        {"sample_id": "b", "pid": "1", "rank1_correct": False, "pred": "x"},
        {"sample_id": "c", "pid": "2", "rank1_correct": True, "pred": "2"},
    ]
    after = [
        {"sample_id": "a", "pid": "1", "rank1_correct": True, "pred": "1"},
        {"sample_id": "b", "pid": "1", "rank1_correct": True, "pred": "1"},
        {"sample_id": "c", "pid": "2", "rank1_correct": True, "pred": "2"},
    ]
    result = paired_uncertainty(
        before,
        after,
        bootstrap_samples=50,
        permutation_samples=100,
        seed=7,
    )
    assert result["bootstrap_unit"] == "pid_cluster"
    assert result["pid_clusters"] == 2
    assert result["pid_sign_flip_two_sided_p"] is not None
    assert result["inference_label"] == "exploratory"


def test_fake_matrix_writes_complete_artifacts_and_is_cacheable(tmp_path: Path) -> None:
    payload, manifest = _fake_payload(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "stale-from-other-run.txt").write_text(
        "must not enter this run checksum graph",
        encoding="utf-8",
    )
    face = FakeFace()
    settings = SimpleNamespace(
        face_model="buffalo_l",
        face_rec_backend="arcface",
        face_gfpgan_weights="",
        face_codeformer_weights="",
        face_codeformer_fidelity=0.5,
        face_realesrgan_x2plus_weights="",
    )

    result = evaluate_matrix_payload(
        payload,
        manifest,
        artifacts,
        face_mod=face,
        settings=settings,
        seed=0,
        bootstrap_samples=20,
        permutation_samples=20,
        force_recompute=False,
        fiqa_fn=lambda bgr: None,
    )

    assert tuple(result["recoverable_main"]) == (
        "A_original",
        "C1_gated_gfpgan",
        "C2_gated_codeformer_w1",
        "C3_gated_realesrgan_x2plus",
    )
    required = {
        "recoverable_rank1_ci.png",
        "rescue_harm_transitions.png",
        "qualitative_grid.png",
        "fiqa_delta_margin_scatter.png",
        "accuracy_vs_latency.png",
        "b_vs_c_eligibility.png",
    }
    assert required <= {path.name for path in (artifacts / "figures").glob("*.png")}
    assert len(list((artifacts / "figures" / "all40_recoverable_panels").glob("*.png"))) == 1
    assert (artifacts / "per_sample_long.json").is_file()
    assert (artifacts / "per_sample_long.csv").is_file()
    assert (artifacts / "checksums.json").is_file()
    checksums = json.loads(
        (artifacts / "checksums.json").read_text(encoding="utf-8")
    )
    assert "stale-from-other-run.txt" not in {
        row["path"] for row in checksums["files"]
    }
    assert len(list((artifacts / "run_specs").glob("*.json"))) == 6
    assert len(list((artifacts / "cache_v3" / "runs").glob("*/cache.json"))) == 5
    assert settings.face_codeformer_fidelity == 1.0
    assert {spec["backend"] for spec in BACKEND_SPECS} <= set(face.loaded)
    for arm in (
        "C1_gated_gfpgan",
        "C2_gated_codeformer_w1",
        "C3_gated_realesrgan_x2plus",
    ):
        assert result["results"][arm]["by_eligibility"]["recoverable"][
            "vector_available"
        ] == 1
    assert result["diagnostics"]["gfpgan"]["fiqa_diagnostic_aligned"] == {
        "denominator": 2,
        "above_threshold": 0,
        "below_or_unavailable": 2,
        "note": "diagnostic only; FIQA does not route B or C",
    }

    second = evaluate_matrix_payload(
        payload,
        manifest,
        artifacts,
        face_mod=face,
        settings=settings,
        seed=0,
        bootstrap_samples=20,
        permutation_samples=20,
        force_recompute=False,
        fiqa_fn=lambda bgr: 0.9,
    )
    assert all(value["cache_reused"] for value in second["runtime"].values())
    for spec in BACKEND_SPECS:
        assert (
            f"aligned_{spec['b_arm']}_vs_{spec['c_arm']}"
            in second["paired_transitions"]
        )


def test_custom_url_weight_hash_uses_backend_download_name(
    monkeypatch,
    tmp_path: Path,
) -> None:
    url = "https://example.test/custom-codeformer.pth"
    settings = _settings(face_codeformer_weights=url)
    filename = (
        f"{__import__('hashlib').sha256(url.encode()).hexdigest()[:12]}-"
        "custom-codeformer.pth"
    )
    cached = (
        tmp_path
        / ".cache"
        / "event-monitor"
        / "superres"
        / "codeformer-v0.1.0"
        / filename
    )
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"custom-weights")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    resolved, digest = _configured_weight(settings, "codeformer")

    assert resolved == str(cached.resolve())
    assert digest == file_sha256(cached)


def test_backend_cache_rejects_tampered_metadata(tmp_path: Path) -> None:
    payload, manifest = _fake_payload(tmp_path)
    artifacts = tmp_path / "artifacts"
    face = FakeFace()
    settings = _settings()
    evaluate_matrix_payload(
        payload,
        manifest,
        artifacts,
        face_mod=face,
        settings=settings,
        seed=0,
        bootstrap_samples=5,
        permutation_samples=5,
        force_recompute=False,
        fiqa_fn=lambda bgr: 0.9,
    )
    cache_paths = list((artifacts / "cache_v3" / "runs").glob("*/cache.json"))
    codeformer_cache = next(
        path
        for path in cache_paths
        if json.loads(path.read_text(encoding="utf-8"))["run_spec"]["backend"]
        == "codeformer"
    )
    cache = json.loads(codeformer_cache.read_text(encoding="utf-8"))
    cache["records"][0]["fiqa_diagnostic_passed"] = not bool(
        cache["records"][0]["fiqa_diagnostic_passed"]
    )
    codeformer_cache.write_text(json.dumps(cache), encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata checksum"):
        evaluate_matrix_payload(
            payload,
            manifest,
            artifacts,
            face_mod=face,
            settings=settings,
            seed=0,
            bootstrap_samples=5,
            permutation_samples=5,
            force_recompute=False,
            fiqa_fn=lambda bgr: 0.9,
        )


def test_unexpected_native_backend_size_fails_without_fallback(
    tmp_path: Path,
) -> None:
    class WrongSizeFace(FakeFace):
        def enhance(self, image: Image.Image, *, aligned: bool, backend: str):
            if backend == "codeformer":
                return image.resize((511, 512), Image.Resampling.BILINEAR)
            return super().enhance(image, aligned=aligned, backend=backend)

    payload, manifest = _fake_payload(tmp_path)
    result = evaluate_matrix_payload(
        payload,
        manifest,
        tmp_path / "artifacts",
        face_mod=WrongSizeFace(),
        settings=_settings(),
        seed=0,
        bootstrap_samples=5,
        permutation_samples=5,
        force_recompute=False,
        fiqa_fn=lambda bgr: 0.9,
    )

    codeformer = result["diagnostics"]["codeformer"]
    assert codeformer["failures"]["count"] == 2
    assert result["results"]["B2_all_codeformer_w1"]["aligned156"][
        "vector_available"
    ] == 0


def test_cli_preserves_old_commands_and_accepts_exact_matrix_shape() -> None:
    parser = build_parser()
    assert parser.parse_args(["prepare", "--data", "d"]).command == "prepare"
    assert parser.parse_args(
        ["evaluate", "--manifest", "m"]
    ).command == "evaluate"
    matrix = parser.parse_args(
        [
            "evaluate-matrix",
            "--manifest",
            "m",
            "--output",
            "o",
            "--artifact-root",
            "a",
            "--seed",
            "4",
            "--bootstrap-samples",
            "10",
            "--permutation-samples",
            "20",
            "--force-recompute",
        ]
    )
    assert matrix.command == "evaluate-matrix"
    assert matrix.artifact_root == "a" and matrix.force_recompute
