from __future__ import annotations

from collections.abc import Callable


def containment(face_box, person_box) -> float:
    """Return intersection area divided by face area."""
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box
    ix1, iy1 = max(fx1, px1), max(fy1, py1)
    ix2, iy2 = min(fx2, px2), min(fy2, py2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    face_area = max(1e-6, (fx2 - fx1) * (fy2 - fy1))
    return intersection / face_area


def associate_to_persons(
    faces: list[dict],
    person_dets: list[dict],
    *,
    min_contain: float,
    ambiguity_margin: float,
    max_head_y_ratio: float,
    containment_fn: Callable[[object, object], float] = containment,
) -> dict[int, dict]:
    """Greedily produce an unambiguous one-to-one face/person assignment."""
    persons = [
        detection
        for detection in person_dets
        if detection.get("label", "person") == "person"
        and detection.get("track_id") is not None
    ]
    pairs: list[tuple[float, int, int]] = []
    ambiguous_faces: set[int] = set()
    for face_index, face in enumerate(faces):
        face_box = face["bbox"]
        face_cx = (float(face_box[0]) + float(face_box[2])) / 2.0
        face_cy = (float(face_box[1]) + float(face_box[3])) / 2.0
        face_pairs = []
        for person_index, person in enumerate(persons):
            px1, py1, px2, py2 = [
                float(value) for value in person["box"][:4]
            ]
            person_height = max(1e-6, py2 - py1)
            head_y_ratio = (face_cy - py1) / person_height
            score = containment_fn(face_box, person["box"])
            if not (px1 <= face_cx <= px2):
                continue
            if head_y_ratio < -0.05 or head_y_ratio > max_head_y_ratio:
                continue
            if score >= min_contain:
                face_pairs.append(
                    (score, face_index, person_index)
                )
        face_pairs.sort(reverse=True)
        if (
            len(face_pairs) > 1
            and face_pairs[0][0] - face_pairs[1][0] < ambiguity_margin
        ):
            ambiguous_faces.add(face_index)
            continue
        pairs.extend(face_pairs)

    result: dict[int, dict] = {}
    used_faces: set[int] = set()
    used_persons: set[int] = set()
    for score, face_index, person_index in sorted(pairs, reverse=True):
        if (
            face_index in ambiguous_faces
            or face_index in used_faces
            or person_index in used_persons
        ):
            continue
        track_id = int(persons[person_index]["track_id"])
        associated = dict(faces[face_index])
        associated["association_score"] = round(float(score), 4)
        result[track_id] = associated
        used_faces.add(face_index)
        used_persons.add(person_index)
    return result


__all__ = ["associate_to_persons", "containment"]
