from __future__ import annotations

import pytest

from app.core.config import BASE_DIR, Settings


def test_codeformer_fidelity_defaults_to_identity_first() -> None:
    assert Settings().face_codeformer_fidelity == 1.0


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
    assert 'value="codeformer"' in html
    assert 'value="realesrgan_x2plus"' in html
    assert 'id="faceCodeformerFidelity"' in html
    assert 'formData, "face_codeformer_fidelity"' in settings_js
    assert "if (isCodeFormerSelected())" in settings_js
    assert "catalog.backends" in settings_js
