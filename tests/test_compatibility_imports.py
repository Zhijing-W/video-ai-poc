from __future__ import annotations


def test_legacy_module_exports_remain_available() -> None:
    from app import body_gallery, face
    from app.services.event_understanding import (
        EVENT_SYSTEM,
        WINDOW_SUMMARY_SCHEMA,
        WINDOW_SUMMARY_SYSTEM,
        format_identity_grounding,
    )

    assert hasattr(body_gallery, "SessionGallery")
    assert callable(face._ensure_superres)
    assert EVENT_SYSTEM
    assert WINDOW_SUMMARY_SYSTEM
    assert WINDOW_SUMMARY_SCHEMA
    assert callable(format_identity_grounding)
