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


def test_face_facade_preserves_public_and_private_compatibility_seams() -> None:
    from app import face

    public_exports = {
        "FACE_DIM",
        "active_backend",
        "align_face",
        "adaface_error",
        "assess_quality",
        "associate_to_persons",
        "available_superres_backends",
        "best_face",
        "detect",
        "finalize_identity",
        "embed_aligned_face",
        "enhance",
        "fuse_embeddings",
        "geometry_descriptor",
        "register_superres_backend",
        "superres_error",
        "validate_superres_backend",
    }
    private_seams = {
        "_state",
        "_ensure_backend",
        "_to_bgr",
        "_detect_face_candidates",
        "_align_face",
        "_attach_optional_geometry",
    }

    assert public_exports == set(face.__all__)
    assert all(hasattr(face, name) for name in private_seams)


def test_face_facade_and_runtime_share_one_insightface_state(monkeypatch) -> None:
    from app import face
    from app.identity.face import association, geometry, recognition, runtime

    assert face._state is runtime._state
    replacement = {"backend": "test-backend", "model": {"app": object()}}
    monkeypatch.setattr(face, "_state", replacement)

    assert face.active_backend() == "test-backend"
    assert callable(runtime.detect_face_candidates)
    assert callable(recognition.embed_aligned_face)
    assert callable(geometry.geometry_descriptor)
    assert callable(association.associate_to_persons)
