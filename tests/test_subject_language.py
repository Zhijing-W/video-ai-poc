from __future__ import annotations

import pytest

from app.llm_client import parse_json
from app.subject_language import normalize_subject_references


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
