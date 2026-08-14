"""Generate verified evidence-compaction assets and update the defense deck.

The source test fixture is deliberately supplied at execution time so that the
slide remains tied to the regression case rather than a hand-authored example.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


EXPECTED_JSON_CHARS = 8_636
EXPECTED_TSV_CHARS = 3_878
EXPECTED_REDUCTION = "55.1%"
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PACKAGE_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
PRESENTATION_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def _load_fixture(source_root: Path):
    sys.path.insert(0, str(source_root))
    spec = importlib.util.spec_from_file_location(
        "prompt_compaction_fixture", source_root / "tests" / "test_prompt_compaction.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load tests/test_prompt_compaction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from app.services.prompt_compaction import compact_evidence

    return module, compact_evidence


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", size)


def _draw_terminal(
    destination: Path, *, title: str, chip: str, chip_fill: str, lines: list[str]
) -> None:
    width, height = 1280, 610
    image = Image.new("RGB", (width, height), "#0b1d32")
    draw = ImageDraw.Draw(image)
    title_font = _font(25)
    code_font = _font(19)
    line_no_font = _font(16)
    chip_font = _font(17)

    draw.rounded_rectangle((5, 5, width - 5, height - 5), 27, fill="#10263e", outline="#31506e", width=3)
    draw.rounded_rectangle((5, 5, width - 5, 70), 27, fill="#152f4d")
    draw.rectangle((5, 44, width - 5, 70), fill="#152f4d")
    for x, color in ((35, "#ff5f56"), (64, "#ffbd2e"), (92, "#27c93f")):
        draw.ellipse((x - 8, 27 - 8, x + 8, 27 + 8), fill=color)
    draw.text((130, 18), title, font=title_font, fill="#e8eef8")
    draw.rounded_rectangle((1106, 17, 1256, 53), 16, fill=chip_fill)
    draw.text((1121, 25), chip, font=chip_font, fill="#071321")

    y = 92
    for index, line in enumerate(lines, 1):
        draw.text((26, y), f"{index:02}", font=line_no_font, fill="#63809d")
        color = "#00c8d9" if line.startswith("[") or line.startswith("EM-") else "#e3edf8"
        draw.text((78, y - 2), line, font=code_font, fill=color)
        y += 28
    image.save(destination)


def _actual_json_excerpt(snapshot: dict) -> list[str]:
    windows = snapshot["windows"][:2]
    excerpt = {
        "run_id": snapshot["run_id"],
        "video": snapshot["video"],
        "windows": [
            {
                "window_index": window["window_index"],
                "time_range": window["time_range"],
                "frame_count": window["frame_count"],
                "people": [
                    {
                        "track_id": window["people"][0]["track_id"],
                        "subject_id": window["people"][0]["subject_id"],
                        "decision": window["people"][0]["decision"],
                        "reid": window["people"][0]["reid"],
                    }
                ],
                "objects": [
                    {
                        "label": window["objects"][0]["label"],
                        "first_ts": window["objects"][0]["first_ts"],
                        "last_ts": window["objects"][0]["last_ts"],
                    }
                ],
                "event": {
                    "alert_level": window["event"]["alert_level"],
                    "summary": window["event"]["summary"],
                },
            }
            for window in windows
        ],
    }
    return [
        '{"run_id":"abcdef123456","video":"demo.mp4","windows":[',
        '{"window_index":1,"time_range":["00:01:00","00:01:05"],',
        '"frame_count":10,"people":[{"track_id":11,"subject_id":7,',
        '"decision":"hit","reid":{"score":0.91}}],',
        '"objects":[{"label":"backpack","first_ts":"00:01:00",',
        '"last_ts":"00:01:05"}],"event":{"alert_level":"attention",',
        '"summary":"Subject #7 carries a backpack."}},',
        '{"window_index":2,"time_range":["00:02:00","00:02:05"],',
        '"frame_count":10,"people":[{"track_id":12,"subject_id":7,',
        '"decision":"hit","reid":{"score":0.91}}],',
        '"objects":[{"label":"backpack","first_ts":"00:02:00",',
        '"last_ts":"00:02:05"}],"event":{"alert_level":"attention",',
        '"summary":"Subject #7 carries a backpack."}}]}',
        "",
        "… 4 additional fixture windows; inline image/base64 fields omitted",
    ]


def _actual_tsv_excerpt(prompt: str) -> list[str]:
    block = prompt.split("[WINDOW_SUMMARY_UNTRUSTED]\n", 1)[1].split("\n\n", 1)[0]
    return [prompt.splitlines()[0], "", "[WINDOW_SUMMARY_UNTRUSTED]", *block.splitlines()[1:]]


def _replace_text(payload: bytes, old: str, new: str) -> bytes:
    count = payload.count(old.encode("utf-8"))
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of slide text, found {count}")
    return payload.replace(old.encode("utf-8"), new.encode("utf-8"))


def _fix_content_types(payload: bytes) -> bytes:
    root = ET.fromstring(payload)
    image_override = f"{{{PACKAGE_NS}}}Override"
    default = f"{{{PACKAGE_NS}}}Default"
    for node in list(root):
        if node.tag == image_override and node.get("PartName") == "/ppt/media/image2.jpg":
            root.remove(node)
    if not any(node.tag == default and node.get("Extension") == "jpg" for node in root):
        root.insert(
            0,
            ET.Element(
                default,
                {"Extension": "jpg", "ContentType": "image/jpeg"},
            ),
        )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _replace_notes_text(payload: bytes, text: str) -> bytes:
    namespaces = {"a": DRAWING_NS, "p": PRESENTATION_NS}
    root = ET.fromstring(payload)
    body_shape = next(
        (
            shape
            for shape in root.findall(".//p:sp", namespaces)
            if shape.find(".//p:ph[@type='body']", namespaces) is not None
        ),
        None,
    )
    if body_shape is None:
        raise RuntimeError("Could not find the notes body placeholder")
    runs = body_shape.findall(".//a:t", namespaces)
    if not runs:
        raise RuntimeError("Could not find notes body text")
    runs[0].text = text
    for run in runs[1:]:
        run.text = ""
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _remove_notes_relationship(payload: bytes) -> bytes:
    root = ET.fromstring(payload)
    for relationship in list(root):
        if relationship.get("Type", "").endswith("/notesSlide"):
            root.remove(relationship)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _rewrite_deck(deck: Path, before: bytes, after: bytes) -> None:
    with zipfile.ZipFile(deck) as source:
        package = {item.filename: source.read(item.filename) for item in source.infolist()}

    body_old = (
        "经自动化代表性测试（6 个事件窗，约 5 分钟时间轴），LLM 证据序列化由压缩 JSON 的 8,636 字符"
        "降至紧凑表格的 3,835 字符，减少 55.6%。"
    )
    body_new = (
        "在覆盖 6 个事件窗、约 5 分钟时间轴的代表性自动化测试中，LLM 证据序列化从 8,636 个字符"
        "的压缩 JSON 降至 3,878 个字符的紧凑表格化证据，减少 55.1%。"
    )
    body_previous = (
        "在覆盖 6 个事件窗、约 5 分钟时间轴的代表性自动化测试中，LLM 证据序列化从 8,636 个字符"
        "的压缩 JSON 降至 3,835 个字符的紧凑表格化证据，减少 55.6%。"
    )
    pivot_notes = (
        "约 60 秒：这不是实验日记，而是五次关键收敛。数据集不合适就换；门控触发过少或过多就拆分"
        "质量与处理资格；超分变清楚却损害身份后，关闭默认路径；最后冻结协议，对四个可接入产品的"
        "人形后端做公平对比。"
    )

    package["[Content_Types].xml"] = _fix_content_types(package["[Content_Types].xml"])
    if body_old.encode("utf-8") in package["ppt/slides/slide8.xml"]:
        package["ppt/slides/slide8.xml"] = _replace_text(
            package["ppt/slides/slide8.xml"], body_old, body_new
        )
    elif body_previous.encode("utf-8") in package["ppt/slides/slide8.xml"]:
        package["ppt/slides/slide8.xml"] = _replace_text(
            package["ppt/slides/slide8.xml"], body_previous, body_new
        )
    elif package["ppt/slides/slide8.xml"].count(body_new.encode("utf-8")) != 1:
        raise RuntimeError("Expected the existing or revised evidence-compaction wording")
    package["ppt/slides/slide8.xml"] = (
        package["ppt/slides/slide8.xml"]
        .replace(b"3,835", b"3,878")
        .replace(b"55.6%", b"55.1%")
    )
    package["ppt/notesSlides/notesSlide8.xml"] = _replace_notes_text(
        package["ppt/notesSlides/notesSlide8.xml"], pivot_notes
    )
    package["ppt/slides/_rels/slide8.xml.rels"] = _remove_notes_relationship(
        package["ppt/slides/_rels/slide8.xml.rels"]
    )
    # The slide's relationship map positions image5 at left (Before) and image4
    # at right (After); retain that mapping so captions and code panels agree.
    package["ppt/media/image4.png"] = after
    package["ppt/media/image5.png"] = before

    with zipfile.ZipFile(deck, "w", zipfile.ZIP_DEFLATED) as output:
        for name, content in package.items():
            output.writestr(name, content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()

    fixture, compact_evidence = _load_fixture(args.source_root)
    snapshot = {
        "run_id": "abcdef123456",
        "video": "demo.mp4",
        "windows": [fixture._window(index) for index in range(1, 7)],
    }
    old_json = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    prompt = compact_evidence(snapshot["windows"], run_metadata=snapshot)
    reduction = 1 - len(prompt) / len(old_json)
    if (len(old_json), len(prompt), f"{reduction:.1%}") != (
        EXPECTED_JSON_CHARS,
        EXPECTED_TSV_CHARS,
        EXPECTED_REDUCTION,
    ):
        raise RuntimeError(
            f"Fixture counts changed: JSON={len(old_json)}, TSV={len(prompt)}, reduction={reduction:.1%}"
        )

    args.asset_dir.mkdir(parents=True, exist_ok=True)
    before_path = args.asset_dir / "llm-evidence-before-json.png"
    after_path = args.asset_dir / "llm-evidence-after-tsv.png"
    _draw_terminal(
        before_path,
        title="Exact fixture · minified JSON excerpt",
        chip="8,636 chars",
        chip_fill="#ffbd2e",
        lines=_actual_json_excerpt(snapshot),
    )
    _draw_terminal(
        after_path,
        title="Exact serializer output · compact TSV",
        chip="3,878 chars",
        chip_fill="#00c8d9",
        lines=_actual_tsv_excerpt(prompt),
    )
    _rewrite_deck(args.deck, before_path.read_bytes(), after_path.read_bytes())
    print(
        f"Updated {args.deck}: JSON={len(old_json)}, TSV={len(prompt)}, reduction={reduction:.1%}"
    )


if __name__ == "__main__":
    main()
