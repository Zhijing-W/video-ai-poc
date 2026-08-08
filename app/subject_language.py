"""Normalize stable subject references in parsed model output."""
from __future__ import annotations

import re

from .event_monitor_i18n import normalize_report_language


_SUBJECT_REFERENCE = re.compile(
    r"(?<!\w)(?:subject|主体)\s*#\s*(\d+)(?!\w)",
    re.IGNORECASE,
)


def normalize_subject_references(payload: object, language: str | None) -> object:
    """Recursively render model-produced subject IDs in the report language.

    Only complete ``subject#<id>`` / ``主体#<id>`` references are changed, so
    prose and identifiers that merely contain a similar substring remain intact.
    """
    target_language = normalize_report_language(language)
    if target_language not in {"en", "zh-CN"}:
        return payload

    label = "subject" if target_language == "en" else "主体"

    if isinstance(payload, str):
        return _SUBJECT_REFERENCE.sub(lambda match: f"{label}#{match.group(1)}", payload)
    if isinstance(payload, dict):
        return {
            key: normalize_subject_references(value, target_language)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [normalize_subject_references(value, target_language) for value in payload]
    if isinstance(payload, tuple):
        return tuple(normalize_subject_references(value, target_language) for value in payload)
    return payload


__all__ = ["normalize_subject_references"]
