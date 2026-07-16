from __future__ import annotations

from app.identity.identity_confidence import score_identity_confidence


def test_score_identity_confidence_preserves_output_contract() -> None:
    ident = {
        "score": 0.72,
        "decision": "hit",
        "face": {"matched": True, "quality": "clear", "quality_score": 0.8, "match_score": 0.88},
        "gait": {"score": 0.67},
    }

    fused = score_identity_confidence(ident)

    assert set(fused) == {"confidence", "resolved", "multi_source", "agreed", "primary", "sources"}
    assert fused["resolved"] is True
    assert fused["multi_source"] is True
    assert fused["agreed"] is True
    assert fused["primary"] == "face"
    assert [item["cue"] for item in fused["sources"]] == ["face", "body", "gait"]
    assert ident["fused"] == fused
