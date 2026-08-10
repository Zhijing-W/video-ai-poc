from __future__ import annotations

import pytest

from app.llm_client import parse_json
from app.subject_language import (
    decorate_named_subject_references,
    normalize_subject_references,
)


@pytest.mark.parametrize(
    ("language", "label"),
    [("zh-CN", "主体"), ("en", "subject")],
)
def test_normalize_subject_references_recursively_for_report_language(
    language: str,
    label: str,
) -> None:
    payload = {
        "event": {
            "subject": "subject#01 与 主体 #2",
            "action": "subject #3 leaves",
            "summary": "主体#4 appears",
            "notification": "subject#5 alert",
        },
        "overall": {
            "story": [{"subject": "主体#6", "action": "subject#7 waits"}],
            "subjects": ["subject#8: observed"],
        },
        "chat": {
            "answer": "主体#9",
            "evidence": [{"reason": "subject#10 is visible"}],
            "limitations": "主体 #11 is obscured",
        },
    }

    normalized = normalize_subject_references(payload, language)

    assert normalized["event"]["subject"] == f"{label}#01 与 {label}#2"
    assert normalized["event"]["action"] == f"{label}#3 leaves"
    assert normalized["event"]["summary"] == f"{label}#4 appears"
    assert normalized["event"]["notification"] == f"{label}#5 alert"
    assert normalized["overall"]["story"][0] == {
        "subject": f"{label}#6",
        "action": f"{label}#7 waits",
    }
    assert normalized["overall"]["subjects"] == [f"{label}#8: observed"]
    assert normalized["chat"]["answer"] == f"{label}#9"
    assert normalized["chat"]["evidence"][0]["reason"] == f"{label}#10 is visible"
    assert normalized["chat"]["limitations"] == f"{label}#11 is obscured"
    assert payload["event"]["subject"] == "subject#01 与 主体 #2"


def test_normalize_subject_references_does_not_change_unrelated_words() -> None:
    value = "subjective subject#1x foo_subject#2 主体#3号 subject#4"

    assert normalize_subject_references(value, "zh-CN") == (
        "subjective subject#1x foo_subject#2 主体#3号 主体#4"
    )


def test_parse_json_normalizes_subject_references_after_parsing() -> None:
    parsed = parse_json(
        '{"answer":"主体 #12","evidence":[{"reason":"subject#13"}]}',
        language="en",
    )

    assert parsed == {
        "answer": "subject#12",
        "evidence": [{"reason": "subject#13"}],
    }


def test_decorate_named_subject_references_maps_alias_subject_and_track() -> None:
    windows = [
        {
            "window_index": 0,
            "people": [
                {
                    "track_id": 7,
                    "source_track_ids": [7, 9],
                    "subject_id": 1,
                    "db_identity": "Alice",
                },
                {
                    "track_id": 8,
                    "subject_id": 2,
                    "db_identity": None,
                },
            ],
        }
    ]
    payload = {
        "subject": "S1",
        "summary": "subject#1 与 主体#2",
        "story": ["track#7 leaves", "Alice (主体#1) waits"],
    }

    decorated = decorate_named_subject_references(
        payload,
        windows,
        "zh-CN",
    )

    assert decorated["subject"] == "Alice (主体#1)"
    assert decorated["summary"] == "Alice (主体#1) 与 主体#2"
    assert decorated["story"] == [
        "Alice (主体#1) leaves",
        "Alice (主体#1) waits",
    ]


def test_named_decorator_does_not_rewrite_scene_s_tokens() -> None:
    windows = [
        {
            "window_index": 0,
            "people": [
                {
                    "track_id": 7,
                    "subject_id": 1,
                    "db_identity": "Alice",
                }
            ],
        }
    ]

    decorated = decorate_named_subject_references(
        {
            "subject": "S1",
            "summary": "Gate S1 opens while subject#1 waits",
            "notification": "Package S1 moved; Alice, subject#1, leaves",
        },
        windows,
        "zh-CN",
    )

    assert decorated["subject"] == "Alice (主体#1)"
    assert decorated["summary"] == (
        "Gate S1 opens while Alice (主体#1) waits"
    )
    assert "Package S1 moved" in decorated["notification"]
    assert "subject#1" not in decorated["notification"]
