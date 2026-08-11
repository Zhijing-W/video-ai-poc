from __future__ import annotations

import os
import subprocess
import sys

import pytest

from app.core.config import BASE_DIR, Settings


def test_product_identity_defaults_are_calibrated_and_separated() -> None:
    assert Settings().face_codeformer_fidelity == 1.0
    assert Settings().face_superres == "off"
    assert Settings().face_rec_backend == "arcface"
    assert Settings().gallery_candidate_top_k == 5
    assert Settings().enrollment_evidence_frames == 5


def test_legacy_reid_top_k_migrates_to_gallery_depth_only() -> None:
    env = os.environ.copy()
    env.pop("GALLERY_CANDIDATE_TOP_K", None)
    env.pop("ENROLLMENT_EVIDENCE_FRAMES", None)
    env["REID_DECISION_TOP_K"] = "30"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from app.core.config import settings; "
                "print(settings.gallery_candidate_top_k, "
                "settings.enrollment_evidence_frames)"
            ),
        ],
        cwd=BASE_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "30 5"


@pytest.mark.parametrize("fidelity", [-0.01, 1.01, float("nan")])
def test_codeformer_fidelity_config_rejects_out_of_range(fidelity: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Settings(face_codeformer_fidelity=fidelity)


def test_event_monitor_ui_uses_registered_dropdown_and_conditional_fidelity() -> None:
    html = (BASE_DIR / "templates" / "event-monitor.html").read_text(encoding="utf-8")
    settings_js = (
        BASE_DIR / "static" / "js" / "event-monitor" / "settings.js"
    ).read_text(encoding="utf-8")

    assert '<select id="faceSuperres"' in html
    assert 'id="faceSuperresOptions"' in html
    assert 'id="enrollmentEvidenceFrames"' in html
    assert "Rank-1" not in html
    assert "Rank-5" not in html
    assert "默认（ArcFace）" in html
    assert '"enrollment_evidence_frames"' in settings_js
    assert '"face_superres_options"' in settings_js
    assert "data-superres-option" in settings_js
    assert "catalog.backends" in settings_js
    assert "catalog.metadata" in settings_js
