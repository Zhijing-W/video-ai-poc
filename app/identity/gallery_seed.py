from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .face.quality import face_gallery_quality_ok


class GallerySeedError(ValueError):
    """Raised when a sample gallery manifest or one of its assets is invalid."""


@dataclass(frozen=True)
class GallerySeedSubject:
    label: str
    source_id: str
    body_images: tuple[Path, ...]
    face_images: tuple[Path, ...]
    attributes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GallerySeed:
    manifest_path: Path
    dataset: dict[str, str]
    subjects: tuple[GallerySeedSubject, ...]

    def public_dict(self) -> dict:
        return {
            "version": 1,
            "manifest": self.manifest_path.name,
            "dataset": dict(self.dataset),
            "subject_count": len(self.subjects),
            "subjects": [
                {
                    "label": subject.label,
                    "source_id": subject.source_id,
                    "body_shots": len(subject.body_images),
                    "face_shots": len(subject.face_images),
                }
                for subject in self.subjects
            ],
        }


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GallerySeedError(f"{field} must be a non-empty string")
    return value.strip()


def _resolve_images(root: Path, values: object, field: str) -> tuple[Path, ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise GallerySeedError(f"{field} must be a list")
    images = []
    for index, value in enumerate(values):
        relative = Path(_require_text(value, f"{field}[{index}]"))
        if relative.is_absolute():
            raise GallerySeedError(f"{field}[{index}] must be relative to the manifest")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise GallerySeedError(f"{field}[{index}] escapes the sample directory") from exc
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            raise GallerySeedError(f"{field}[{index}] is not a supported image")
        if not path.is_file():
            raise GallerySeedError(f"{field}[{index}] does not exist: {relative}")
        images.append(path)
    return tuple(images)


def load_gallery_seed(video_path: str | Path) -> GallerySeed | None:
    video_path = Path(video_path)
    manifest_path = video_path.with_suffix(".gallery.json")
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GallerySeedError(f"invalid gallery manifest {manifest_path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GallerySeedError("gallery manifest root must be an object")
    if payload.get("version") != 1:
        raise GallerySeedError("gallery manifest version must be 1")

    raw_dataset = payload.get("dataset")
    if not isinstance(raw_dataset, dict):
        raise GallerySeedError("gallery manifest dataset must be an object")
    dataset = {
        str(key): _require_text(value, f"dataset.{key}")
        for key, value in raw_dataset.items()
    }
    if "name" not in dataset:
        raise GallerySeedError("gallery manifest dataset.name is required")

    raw_subjects = payload.get("subjects")
    if not isinstance(raw_subjects, list) or not raw_subjects:
        raise GallerySeedError("gallery manifest subjects must be a non-empty list")
    root = manifest_path.parent.resolve()
    labels = set()
    subjects = []
    for index, item in enumerate(raw_subjects):
        if not isinstance(item, dict):
            raise GallerySeedError(f"subjects[{index}] must be an object")
        label = _require_text(item.get("label"), f"subjects[{index}].label")
        if label in labels:
            raise GallerySeedError(f"duplicate gallery label: {label}")
        labels.add(label)
        source_id = _require_text(
            item.get("source_id"),
            f"subjects[{index}].source_id",
        )
        body_images = _resolve_images(
            root,
            item.get("body_images"),
            f"subjects[{index}].body_images",
        )
        face_images = _resolve_images(
            root,
            item.get("face_images"),
            f"subjects[{index}].face_images",
        )
        if not body_images and not face_images:
            raise GallerySeedError(
                f"subjects[{index}] must provide body_images or face_images"
            )
        raw_attributes = item.get("attributes") or []
        if not isinstance(raw_attributes, list):
            raise GallerySeedError(f"subjects[{index}].attributes must be a list")
        attributes = tuple(
            _require_text(value, f"subjects[{index}].attributes")
            for value in raw_attributes
        )
        subjects.append(
            GallerySeedSubject(
                label=label,
                source_id=source_id,
                body_images=body_images,
                face_images=face_images,
                attributes=attributes,
            )
        )
    return GallerySeed(
        manifest_path=manifest_path.resolve(),
        dataset=dataset,
        subjects=tuple(subjects),
    )


def _subject_attributes(seed: GallerySeed, subject: GallerySeedSubject) -> list[str]:
    return [
        *subject.attributes,
        f"dataset:{seed.dataset['name']}",
        f"source_id:{subject.source_id}",
    ]


def seed_body_gallery(
    seed: GallerySeed,
    session_id: str,
    dim: int,
    *,
    reid_module,
    gallery_module,
) -> dict:
    subjects = []
    total_shots = 0
    for subject in seed.subjects:
        if not subject.body_images:
            continue
        subject_id = None
        shots = 0
        for image_path in subject.body_images:
            try:
                with Image.open(image_path) as source_image:
                    image = source_image.convert("RGB")
                quality = reid_module.assess_quality(image)
                vector = reid_module.embed(image, purpose="identity_gallery")
                result = gallery_module.with_gallery_locked(
                    session_id,
                    dim,
                    lambda gallery: gallery.enroll_labeled(
                        vector,
                        quality,
                        label=subject.label,
                        attributes=_subject_attributes(seed, subject),
                    ),
                )
            except Exception as exc:
                raise GallerySeedError(
                    f"failed to seed body image for {subject.label}: "
                    f"{image_path.name}: {exc}"
                ) from exc
            if not result.get("enrolled"):
                raise GallerySeedError(
                    f"body image rejected for {subject.label}: "
                    f"{image_path.name}: {result.get('quality_reason') or 'unknown'}"
                )
            subject_id = result["subject_id"]
            shots += 1
            total_shots += 1
        subjects.append(
            {
                "label": subject.label,
                "source_id": subject.source_id,
                "subject_id": subject_id,
                "shots": shots,
            }
        )
    return {"subjects": subjects, "total_shots": total_shots}


def seed_face_gallery(
    seed: GallerySeed,
    session_id: str,
    *,
    face_module,
    gallery_module,
) -> dict:
    subjects = []
    total_shots = 0
    for subject in seed.subjects:
        if not subject.face_images:
            continue
        subject_id = None
        shots = 0
        for image_path in subject.face_images:
            try:
                with Image.open(image_path) as source_image:
                    image = source_image.convert("RGB")
                candidates = face_module.detect(
                    image,
                    with_quality=True,
                    enhance_blurry=False,
                    with_identity=False,
                    with_geometry=False,
                )
                if not candidates:
                    raise GallerySeedError("no face detected")
                face = max(
                    candidates,
                    key=lambda item: float(
                        (item.get("quality") or {}).get("quality") or 0.0
                    ),
                )
                finalized = face_module.finalize_identity(image, face)
                embedding = finalized.get("embedding")
                quality = finalized.get("quality") or face.get("quality")
                if embedding is None or not finalized.get("match_ready"):
                    raise GallerySeedError("face embedding is not match-ready")
                vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
                result = gallery_module.with_gallery_locked(
                    session_id,
                    face_module.FACE_DIM,
                    lambda gallery: gallery.enroll_labeled(
                        vector,
                        quality,
                        label=subject.label,
                        attributes=_subject_attributes(seed, subject),
                        quality_gate=face_gallery_quality_ok,
                    ),
                )
            except Exception as exc:
                raise GallerySeedError(
                    f"failed to seed face image for {subject.label}: "
                    f"{image_path.name}: {exc}"
                ) from exc
            if not result.get("enrolled"):
                raise GallerySeedError(
                    f"face image rejected for {subject.label}: "
                    f"{image_path.name}: {result.get('quality_reason') or 'unknown'}"
                )
            subject_id = result["subject_id"]
            shots += 1
            total_shots += 1
        subjects.append(
            {
                "label": subject.label,
                "source_id": subject.source_id,
                "subject_id": subject_id,
                "shots": shots,
            }
        )
    return {"subjects": subjects, "total_shots": total_shots}


__all__ = [
    "GallerySeed",
    "GallerySeedError",
    "GallerySeedSubject",
    "load_gallery_seed",
    "seed_body_gallery",
    "seed_face_gallery",
]
