from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from app import body_reid
from app.config import Settings, settings
from app.identity.body_reid_backends import differ


ROOT = Path(__file__).resolve().parents[1]


def test_unset_configuration_selects_differ_and_explicit_env_wins(
    monkeypatch,
) -> None:
    monkeypatch.delenv("REID_BACKEND", raising=False)
    assert Settings().reid_backend == "differ"

    monkeypatch.setenv("REID_BACKEND", "clipreid")
    assert Settings().reid_backend == "clipreid"


def test_clean_process_default_startup_selects_and_wires_differ(tmp_path) -> None:
    code = """
import json
import os

os.environ.pop("REID_BACKEND", None)
from app.core.config import Settings, settings
from app import body_reid

fresh = Settings()
result = {"configured": fresh.reid_backend}
with settings.override(reid_backend=fresh.reid_backend):
    try:
        body_reid.active_backend()
    except RuntimeError as exc:
        result["startup_error"] = str(exc)
    else:
        result["active"] = body_reid.active_backend()
        result["dim"] = body_reid.embed_dim()
print(json.dumps(result))
"""
    env = os.environ.copy()
    env.pop("REID_BACKEND", None)
    env["MODEL_ROOT"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    assert result["configured"] == "differ"
    assert result.get("active") == "differ" or (
        "人形ReID后端 differ 加载失败" in result["startup_error"]
        and "DIFFER" in result["startup_error"]
    )


@pytest.mark.skipif(
    os.getenv("RUN_DIFFER_INTEGRATION") != "1",
    reason="set RUN_DIFFER_INTEGRATION=1 when local DIFFER assets are provisioned",
)
def test_default_differ_real_asset_initialization() -> None:
    with settings.override(reid_backend="differ"):
        body_reid.reset_backend()
        try:
            assert body_reid.active_backend() == "differ"
            assert body_reid.embed_dim() == 1024
        finally:
            body_reid.reset_backend()


def test_invalid_backend_still_fails_clearly() -> None:
    with pytest.raises(ValueError, match="未知人形ReID后端.*not-a-backend"):
        body_reid.validate_backend("not-a-backend")


def test_missing_default_assets_report_provisioning_action(tmp_path) -> None:
    with settings.override(
        model_root=str(tmp_path),
        reid_differ_root=str(tmp_path / "source"),
        reid_differ_config=str(tmp_path / "config.yml"),
        reid_differ_weights=str(tmp_path / "weights.pth"),
    ):
        with pytest.raises(
            FileNotFoundError,
            match=r"download_models\.py --reid.*REID_DIFFER_ROOT",
        ):
            differ.load(settings)


def test_deployment_surfaces_pass_differ_and_require_its_assets() -> None:
    values = (ROOT / "charts" / "video-poc" / "values.yaml").read_text(
        encoding="utf-8"
    )
    assert "reidBackend: differ" in values

    for relative_path in (
        "charts/video-poc/templates/deployment-webapi.yaml",
        "charts/video-poc/templates/deployment-gpu.yaml",
    ):
        deployment = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "- name: REID_BACKEND" in deployment
        assert 'value: "{{ .Values.config.reidBackend }}"' in deployment
        assert "- name: MODEL_ROOT" in deployment
        assert "value: /models" in deployment

    cpu_dockerfile = (ROOT / "Dockerfile.cpu").read_text(encoding="utf-8")
    assert "MODEL_ROOT=/models" in cpu_dockerfile
    for dockerfile in (
        cpu_dockerfile,
        (ROOT / "Dockerfile.gpu.base").read_text(encoding="utf-8"),
        (ROOT / "Dockerfile.gpu").read_text(encoding="utf-8"),
    ):
        assert "POSE_MODEL=/models/detection/yolo/yolov8n-pose.pt" in dockerfile
        assert "GAIT_SEG_MODEL=/models/detection/yolo/yolov8m-seg.pt" in dockerfile

    gpu_workflow = (
        ROOT / ".github" / "workflows" / "build-gpu-image.yml"
    ).read_text(encoding="utf-8")
    assert '--build-arg "GPU_BASE_IMAGE=${GPU_BASE_IMAGE}"' in gpu_workflow

    gpu_base = (ROOT / "Dockerfile.gpu.base").read_text(encoding="utf-8")
    assert "python3 -m pip install -r requirements.txt" in gpu_base
    assert "ln -sf /usr/bin/python3.11 /usr/local/bin/python" not in gpu_base

    manifest = json.loads(
        (ROOT / "models" / "manifest.json").read_text(encoding="utf-8")
    )
    assert any(
        entry["environment"] == "REID_DIFFER_WEIGHTS"
        for entry in manifest["required"]
    )


def _pwsh() -> str:
    executable = shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell 7 is not installed")
    return executable


def test_model_upload_fails_before_azure_when_differ_assets_are_missing(
    tmp_path,
) -> None:
    completed = subprocess.run(
        [
            _pwsh(),
            "-NoProfile",
            "-File",
            str(ROOT / "infra" / "upload-models.ps1"),
            "-Storage",
            "unit-test",
            "-ModelRoot",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "DIFFER" in output
    assert "config.yml" in output
    assert "eva02_l_bio_best.pth" in output
    assert "download_models.py --reid" in output


def test_model_upload_provisions_complete_reid_tree(tmp_path) -> None:
    model_root = tmp_path / "models"
    differ_root = model_root / "reid" / "differ"
    (differ_root / "source").mkdir(parents=True)
    (differ_root / "config.yml").write_text("MODEL: {}", encoding="utf-8")
    (differ_root / "eva02_l_bio_best.pth").write_bytes(b"checkpoint")
    for backend in ("osnet", "resnet50", "clipreid", "siglip2"):
        path = model_root / "reid" / backend
        path.mkdir(parents=True)
        (path / "asset").write_text(backend, encoding="utf-8")

    calls_path = tmp_path / "az-calls.jsonl"
    fake_az = tmp_path / "fake_az.py"
    fake_az.write_text(
        "import json, os, sys\n"
        "with open(os.environ['FAKE_AZ_CALLS'], 'a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    if os.name == "nt":
        launcher = tmp_path / "az.cmd"
        launcher.write_text(
            f'@"{sys.executable}" "{fake_az}" %*\n',
            encoding="utf-8",
        )
    else:
        launcher = tmp_path / "az"
        launcher.write_text(
            f"#!{sys.executable}\n"
            f"exec(compile(open({str(fake_az)!r}).read(), {str(fake_az)!r}, 'exec'))\n",
            encoding="utf-8",
        )
        launcher.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + os.pathsep + env.get("PATH", "")
    env["FAKE_AZ_CALLS"] = str(calls_path)
    completed = subprocess.run(
        [
            _pwsh(),
            "-NoProfile",
            "-File",
            str(ROOT / "infra" / "upload-models.ps1"),
            "-Storage",
            "unit-test",
            "-ModelRoot",
            str(model_root),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    calls = [
        json.loads(line)
        for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
    reid_upload = next(
        call
        for call in calls
        if "upload-batch" in call and str(model_root / "reid") in call
    )
    assert reid_upload[reid_upload.index("--destination-path") + 1] == "reid"
    assert "完整 ReID 树已上传" in completed.stdout
