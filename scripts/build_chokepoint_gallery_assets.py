from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont, ImageOps

from app import detector, face
from app.core.config import DATA_DIR


SAMPLE_NAME = "chokepoint_p2e_s5_c2_crowded_24s"
PEOPLE = [
    ("Alice", "0024"),
    ("Bob", "0006"),
    ("Carol", "0019"),
    ("David", "0017"),
    ("Eve", "0025"),
]


def _font(size: int):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _face_crop(source_path: Path, output_path: Path) -> Image.Image:
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    candidates = face.detect(
        image,
        with_quality=True,
        enhance_blurry=False,
        with_identity=False,
        with_geometry=False,
    )
    if not candidates:
        raise RuntimeError(f"no face found in {source_path}")
    item = max(
        candidates,
        key=lambda row: float(row.get("det_score") or 0.0),
    )
    x1, y1, x2, y2 = (float(value) for value in item["bbox"])
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * 1.55
    box = (
        max(0, int(center_x - side / 2)),
        max(0, int(center_y - side * 0.55)),
        min(image.width, int(center_x + side / 2)),
        min(image.height, int(center_y + side * 0.45)),
    )
    crop = image.crop(box).resize(
        (512, 512),
        Image.Resampling.LANCZOS,
    )
    crop.save(output_path, quality=95)
    return crop


def _annotation_index(xml_path: Path) -> dict[str, list[tuple[int, tuple[float, float]]]]:
    by_person: dict[str, list[tuple[int, tuple[float, float]]]] = {}
    for frame in ET.parse(xml_path).getroot().findall("frame"):
        frame_number = int(frame.attrib["number"])
        for person in frame.findall("person"):
            left = person.find("leftEye")
            right = person.find("rightEye")
            eye_center = (
                (int(left.attrib["x"]) + int(right.attrib["x"])) / 2,
                (int(left.attrib["y"]) + int(right.attrib["y"])) / 2,
            )
            by_person.setdefault(person.attrib["id"], []).append(
                (frame_number, eye_center)
            )
    return by_person


def _body_crop(
    frame_path: Path,
    eye_center: tuple[float, float],
    output_path: Path,
) -> Image.Image:
    with Image.open(frame_path) as source:
        image = source.convert("RGB")
    eye_x, eye_y = eye_center
    people = [
        detection
        for detection in detector.detect_objects(
            frame_path.read_bytes(),
            conf=0.2,
        )["detections"]
        if detection["label"] == "person"
        and detection["box"][0] <= eye_x <= detection["box"][2]
        and detection["box"][1] <= eye_y <= detection["box"][3]
    ]
    if not people:
        raise RuntimeError(f"no body found in {frame_path}")
    detection = max(
        people,
        key=lambda row: (
            (row["box"][2] - row["box"][0])
            * (row["box"][3] - row["box"][1])
        ),
    )
    x1, y1, x2, y2 = (int(value) for value in detection["box"])
    crop = image.crop(
        (
            max(0, x1),
            max(0, y1),
            min(image.width, x2),
            min(image.height, y2),
        )
    )
    crop.save(output_path, quality=95)
    return crop


def main() -> int:
    samples = DATA_DIR / "samples"
    gallery = samples / f"{SAMPLE_NAME}_gallery"
    export = DATA_DIR / "exports" / "gallery_ppt"
    still_root = DATA_DIR / "external" / "chokepoint_still"
    dataset_root = DATA_DIR / "external" / "chokepoint"
    body_sequence = "P2E_S1_C2.1"

    shutil.rmtree(gallery, ignore_errors=True)
    gallery.mkdir(parents=True)
    export.mkdir(parents=True, exist_ok=True)
    for child in export.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    annotations = _annotation_index(
        dataset_root / "groundtruth" / f"{body_sequence}.xml"
    )

    subjects = []
    contact_rows = []
    for alias, person_id in PEOPLE:
        gallery_person = gallery / alias.lower()
        export_person = export / alias
        gallery_person.mkdir()
        export_person.mkdir()

        face_paths = []
        face_tiles = []
        for mood in ("Neutral", "Smile"):
            output = gallery_person / f"face_{mood.lower()}.jpg"
            tile = _face_crop(
                still_root / mood / f"ID{person_id}.JPG",
                output,
            )
            face_paths.append(
                str(output.relative_to(samples)).replace("\\", "/")
            )
            face_tiles.append(tile)
            shutil.copy2(
                output,
                export_person / f"{alias}_face_{mood.lower()}.jpg",
            )

        observations = annotations[person_id]
        picks = [
            observations[round((len(observations) - 1) * quantile)]
            for quantile in (0.2, 0.5, 0.8)
        ]
        body_paths = []
        body_tiles = []
        for frame_number, eye_center in picks:
            output = gallery_person / f"body_{frame_number}.jpg"
            tile = _body_crop(
                dataset_root
                / body_sequence
                / f"{frame_number:08d}.jpg",
                eye_center,
                output,
            )
            body_paths.append(
                str(output.relative_to(samples)).replace("\\", "/")
            )
            body_tiles.append(tile)
            shutil.copy2(
                output,
                export_person / f"{alias}_body_{frame_number}.jpg",
            )

        subjects.append(
            {
                "label": alias,
                "source_id": f"ChokePoint ID{person_id}",
                "body_images": body_paths,
                "face_images": face_paths,
                "attributes": [
                    "independent controlled Still face enrollment",
                    "historical body references from P2E_S1.1 C2",
                    "target P2E_S5.1 excluded",
                ],
            }
        )
        contact_rows.append(
            (alias, person_id, face_tiles, body_tiles)
        )

    manifest = {
        "version": 1,
        "dataset": {
            "name": "ChokePoint",
            "sequence": "P2E_S5.1 crowded",
            "analysis_camera": "C2",
            "frame_range": "136-841",
            "source_fps": "30",
            "continuity": "single contiguous original frame run",
            "license": "non-commercial research use",
            "gallery_source": (
                "Controlled Still Neutral/Smile plus historical "
                "P2E_S1.1 C2 body references"
            ),
            "gallery_independence": (
                "No P2E_S5.1 frame or synchronized camera frame "
                "is used for enrollment"
            ),
        },
        "subjects": subjects,
    }
    (samples / f"{SAMPLE_NAME}.gallery.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    width = 1500
    row_height = 330
    sheet = Image.new(
        "RGB",
        (width, 100 + row_height * len(contact_rows) + 90),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (30, 20),
        "Independent ChokePoint Gallery - PPT Contact Sheet",
        font=_font(38),
        fill="black",
    )
    draw.text(
        (30, 65),
        (
            "Faces: controlled Still photos | Bodies: separate "
            "P2E_S1.1 session | Target P2E_S5.1 excluded"
        ),
        font=_font(20),
        fill=(60, 60, 60),
    )
    for row, (alias, person_id, faces, bodies) in enumerate(
        contact_rows
    ):
        y = 100 + row * row_height
        draw.multiline_text(
            (25, y + 15),
            f"{alias}\nID{person_id}",
            font=_font(28),
            fill="black",
        )
        tiles = (faces[0], faces[1], bodies[1])
        labels = (
            "Neutral face",
            "Smile face",
            "Independent body",
        )
        for column, (tile, label) in enumerate(zip(tiles, labels)):
            display = tile.copy()
            display.thumbnail((350, 250), Image.Resampling.LANCZOS)
            x = 210 + column * 420
            sheet.paste(display, (x, y + 10))
            draw.text(
                (x, y + 270),
                label,
                font=_font(20),
                fill=(50, 50, 50),
            )
    draw.text(
        (25, sheet.height - 55),
        (
            "ChokePoint Dataset (NICTA), non-commercial research "
            "use. Derivative contact sheet; target-video frames excluded."
        ),
        font=_font(18),
        fill=(80, 80, 80),
    )
    sheet.save(export / "gallery_people_contact_sheet.png")

    readme = """Gallery presentation assets

Dataset: ChokePoint (NICTA)
Use: non-commercial research/demo only.
Faces: official controlled Still.tar.xz Neutral/Smile photographs.
Bodies: independent P2E_S1.1 camera C2 sequence.
Target video P2E_S5.1 is fully held out and contributes no gallery image.
Aliases Alice/Bob/Carol/David/Eve are demo labels, not real names.
"""
    (export / "README.txt").write_text(readme, encoding="utf-8")
    with zipfile.ZipFile(
        export / "gallery_people_ppt_assets.zip",
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in export.rglob("*"):
            if (
                path.is_file()
                and path.name != "gallery_people_ppt_assets.zip"
            ):
                archive.write(path, path.relative_to(export))

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
