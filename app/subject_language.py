"""Normalize stable subject references in parsed model output."""
from __future__ import annotations

import re

from .event_monitor_i18n import normalize_report_language


_SUBJECT_REFERENCE = re.compile(
    r"(?<!\w)(?:subject|主体)\s*#\s*(\d+)(?!\w)",
    re.IGNORECASE,
)
_SID_REFERENCE = re.compile(r"(?<!\w)S(\d+)(?!\w)")
_TRACK_REFERENCE = re.compile(
    r"(?<!\w)track\s*#\s*(\d+)(?!\w)",
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


def _subject_key(person: dict) -> str:
    subject_id = person.get("subject_id")
    if subject_id is not None:
        return f"subject:{subject_id}"
    tracks = person.get("source_track_ids") or [person.get("track_id")]
    return "track:" + ",".join(
        str(track) for track in tracks if track is not None
    )


def _display_name(person: dict, language: str) -> str | None:
    name = person.get("db_identity")
    if not name:
        return None
    subject_id = person.get("subject_id")
    if subject_id is None:
        return str(name)
    label = "subject" if language == "en" else "主体"
    return f"{name} ({label}#{subject_id})"


def _replace_named_reference(
    text: str,
    pattern: re.Pattern,
    mapping: dict[str, str],
) -> str:
    def replace(match: re.Match) -> str:
        display = mapping.get(match.group(1))
        if not display:
            return match.group(0)
        name = display.split(" (", 1)[0]
        prefix = text[
            max(0, match.start() - len(name) - 3):match.start()
        ]
        if prefix.endswith(f"{name} ("):
            return match.group(0)
        return display

    return pattern.sub(replace, text)


def decorate_named_subject_references(
    payload: object,
    windows: list[dict],
    language: str | None,
    *,
    alias_narrative_fields: set[str] | None = None,
) -> object:
    """Decorate model IDs with trusted CV names while keeping stable IDs."""
    target_language = normalize_report_language(language)
    if target_language not in {"en", "zh-CN"}:
        return payload

    aliases: dict[str, str] = {}
    subjects: dict[str, str] = {}
    tracks: dict[str, str] = {}
    seen_keys: dict[str, int] = {}
    ordered_windows = sorted(
        (window for window in windows if isinstance(window, dict)),
        key=lambda window: (
            int(window.get("window_index"))
            if str(window.get("window_index", "")).isdigit()
            else 10**9
        ),
    )
    for window in ordered_windows:
        people = sorted(
            (
                person
                for person in window.get("people") or []
                if isinstance(person, dict)
            ),
            key=_subject_key,
        )
        for person in people:
            key = _subject_key(person)
            alias = seen_keys.setdefault(key, len(seen_keys) + 1)
            display = _display_name(person, target_language)
            if not display:
                continue
            aliases[str(alias)] = display
            if person.get("subject_id") is not None:
                subjects[str(person["subject_id"])] = display
            for track_id in (
                person.get("source_track_ids")
                or [person.get("track_id")]
            ):
                if track_id is not None:
                    tracks[str(track_id)] = display

    alias_fields = {
        "subject",
        "subjects",
        "subjects_involved",
    }
    alias_fields.update(alias_narrative_fields or set())

    def decorate(value: object, field: str | None = None) -> object:
        if isinstance(value, str):
            output = value
            if field in alias_fields:
                output = _replace_named_reference(
                    output,
                    _SID_REFERENCE,
                    aliases,
                )
            output = _replace_named_reference(
                output,
                _SUBJECT_REFERENCE,
                subjects,
            )
            return _replace_named_reference(
                output,
                _TRACK_REFERENCE,
                tracks,
            )
        if isinstance(value, dict):
            return {
                key: decorate(item, str(key))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [decorate(item, field) for item in value]
        if isinstance(value, tuple):
            return tuple(decorate(item, field) for item in value)
        return value

    return decorate(payload)


def named_subject_displays(
    windows: list[dict],
    language: str | None,
) -> list[str]:
    target_language = normalize_report_language(language)
    if target_language not in {"en", "zh-CN"}:
        return []
    displays: dict[str, str] = {}
    for window in windows:
        for person in window.get("people") or []:
            if not isinstance(person, dict):
                continue
            display = _display_name(person, target_language)
            if display:
                displays[_subject_key(person)] = display
    return [
        displays[key]
        for key in sorted(
            displays,
            key=lambda item: (
                int(item.split(":", 1)[1])
                if item.startswith("subject:")
                and item.split(":", 1)[1].isdigit()
                else 10**9,
                item,
            ),
        )
    ]


def attach_named_subject_roster(
    payload: dict,
    windows: list[dict],
    language: str | None,
    *,
    overall: bool = False,
) -> dict:
    """Attach deterministic CV names so model prose cannot omit them."""
    roster = named_subject_displays(windows, language)
    if not roster:
        return payload
    target_language = normalize_report_language(language)
    payload["named_subjects"] = roster
    roster_text = "、".join(roster) if target_language == "zh-CN" else ", ".join(roster)
    prefix = (
        f"已识别建档人员：{roster_text}。"
        if target_language == "zh-CN"
        else f"Recognized enrolled people: {roster_text}."
    )
    summary_key = "overall_summary" if overall else "summary"
    summary = str(payload.get(summary_key) or "").strip()
    payload[summary_key] = f"{prefix}{summary}" if summary else prefix

    list_key = "subjects" if overall else "subjects_involved"
    existing = list(payload.get(list_key) or [])
    existing_text = "\n".join(str(item) for item in existing)
    for display in roster:
        if display in existing_text:
            continue
        existing.append(
            (
                f"{display}：CV 已确认在本视频出现"
                if target_language == "zh-CN" and overall
                else (
                    f"{display}: confirmed present by CV"
                    if overall
                    else display
                )
            )
        )
    payload[list_key] = existing
    return payload


__all__ = [
    "normalize_subject_references",
    "decorate_named_subject_references",
    "named_subject_displays",
    "attach_named_subject_roster",
]
