"""Comparison figures for schema-v3 experiment artifacts."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .common import _resolve


def _font(size: int):
    for path in (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _panel(image: Image.Image | None, title: str, detail: str, size: int = 240) -> Image.Image:
    panel = Image.new("RGB", (size, size + 76), "white")
    if image is None:
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, size - 1, size - 1), outline="#999999", width=2)
        draw.text((20, size // 2 - 10), "NO IMAGE", fill="#aa0000", font=_font(20))
    else:
        panel.paste(
            image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS),
            (0, 0),
        )
    draw = ImageDraw.Draw(panel)
    draw.text((4, size + 8), title[:34], fill="black", font=_font(16))
    draw.text((4, size + 38), detail[:44], fill="#333333", font=_font(13))
    return panel


def _comparison(
    output: Path,
    record: dict,
    scores: dict[str, dict],
    gallery_examples: dict[str, dict],
    artifact_dir: Path,
    manifest_path: Path,
) -> None:
    panels = []
    gt_gallery = gallery_examples.get(record["pid"])
    gt_image = (
        Image.open(_resolve(gt_gallery["aligned_path"], manifest_path.parent))
        if gt_gallery
        else None
    )
    panels.append(_panel(gt_image, f"GT check-in {record['pid']}", "original Gallery"))
    original_path = _resolve(record.get("original_aligned_path"), artifact_dir)
    panels.append(
        _panel(
            Image.open(original_path) if original_path else None,
            "Original Query",
            f"frame={record.get('face_best_frame_index')} {record.get('eligibility')}",
        )
    )
    sr_path = _resolve(record.get("superres_aligned_path"), artifact_dir)
    panels.append(
        _panel(
            Image.open(sr_path) if sr_path else None,
            "GFPGAN Query",
            (
                f"success={record.get('superres_succeeded')} "
                f"fiqa_diag={record.get('post_superres_accepted')}"
            ),
        )
    )
    added_pred = set()
    for arm in ("A_original", "B_all_superres"):
        score = scores[arm]
        pred = score["pred"]
        if pred and pred != record["pid"] and pred not in added_pred:
            predicted = gallery_examples.get(pred)
            predicted_image = (
                Image.open(_resolve(predicted["aligned_path"], manifest_path.parent))
                if predicted
                else None
            )
            panels.append(
                _panel(
                    predicted_image,
                    f"{arm[0]} predicted {pred}",
                    f"score={score['score']}",
                )
            )
            added_pred.add(pred)
    width = 20 + len(panels) * 260
    canvas = Image.new("RGB", (width, 386), "#f3f3f3")
    draw = ImageDraw.Draw(canvas)
    transition = (
        f"A={'correct' if scores['A_original']['rank1_correct'] else 'wrong'} -> "
        f"B={'correct' if scores['B_all_superres']['rank1_correct'] else 'wrong'}"
    )
    draw.text(
        (16, 10),
        (
            f"{record['sample_id']} | eligibility={record.get('eligibility')} | "
            f"{transition}"
        ),
        fill="black",
        font=_font(16),
    )
    draw.text(
        (16, 38),
        (
            f"A pred={scores['A_original']['pred']} score={scores['A_original']['score']} | "
            f"B pred={scores['B_all_superres']['pred']} score={scores['B_all_superres']['score']}"
        ),
        fill="#333333",
        font=_font(14),
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, (16 + index * 260, 68))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, subsampling=0)


def _open_copy(path: Path | None) -> Image.Image | None:
    if path is None or not path.is_file():
        return None
    with Image.open(path) as opened:
        return opened.convert("RGB").copy()


def _chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    *,
    lower: list[float] | None = None,
    upper: list[float] | None = None,
    colors: list[str] | None = None,
) -> None:
    width, height = 1000, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 24), title, fill="black", font=_font(24))
    left, top, right, bottom = 90, 90, width - 40, height - 110
    draw.line((left, top, left, bottom), fill="#333333", width=2)
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    for tick in range(6):
        y = bottom - (bottom - top) * tick / 5
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        draw.text((35, y - 8), f"{tick / 5:.1f}", fill="#555555", font=_font(13))
    count = max(1, len(values))
    slot = (right - left) / count
    for index, (label, value) in enumerate(zip(labels, values)):
        x0 = left + slot * index + slot * 0.18
        x1 = left + slot * (index + 1) - slot * 0.18
        y = bottom - max(0.0, min(1.0, value)) * (bottom - top)
        color = (colors or ["#4c78a8"] * count)[index]
        draw.rectangle((x0, y, x1, bottom), fill=color)
        draw.text((x0, bottom + 12), label[:20], fill="black", font=_font(13))
        draw.text((x0, y - 22), f"{value:.3f}", fill="black", font=_font(13))
        if lower and upper:
            center = (x0 + x1) / 2
            low_y = bottom - lower[index] * (bottom - top)
            high_y = bottom - upper[index] * (bottom - top)
            draw.line((center, high_y, center, low_y), fill="#111111", width=3)
            draw.line((center - 8, high_y, center + 8, high_y), fill="#111111", width=2)
            draw.line((center - 8, low_y, center + 8, low_y), fill="#111111", width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _scatter(
    path: Path,
    title: str,
    series: list[tuple[str, list[tuple[float, float]], str]],
    *,
    x_label: str,
    y_label: str,
) -> None:
    width, height = 1000, 650
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 20), title, fill="black", font=_font(23))
    all_points = [point for _, points, _ in series for point in points]
    xs = [point[0] for point in all_points] or [-1.0, 1.0]
    ys = [point[1] for point in all_points] or [-1.0, 1.0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min, x_max = x_min - 1, x_max + 1
    if y_min == y_max:
        y_min, y_max = y_min - 1, y_max + 1
    left, top, right, bottom = 100, 80, 950, 550
    draw.rectangle((left, top, right, bottom), outline="#333333", width=2)
    draw.text((430, 590), x_label, fill="black", font=_font(15))
    draw.text((10, 280), y_label, fill="black", font=_font(15))
    for series_index, (label, points, color) in enumerate(series):
        for x_value, y_value in points:
            x = left + (x_value - x_min) / (x_max - x_min) * (right - left)
            y = bottom - (y_value - y_min) / (y_max - y_min) * (bottom - top)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.text((720, 88 + 24 * series_index), label, fill=color, font=_font(14))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _matrix_panel(
    output: Path,
    frozen: dict,
    shared: dict,
    gallery: dict | None,
    backend_records: dict[str, dict],
    manifest_path: Path,
    artifact_root: Path,
) -> None:
    entries: list[tuple[str, Image.Image | None]] = [
        (
            "Gallery",
            _open_copy(
                _resolve(gallery.get("aligned_path"), manifest_path.parent)
                if gallery
                else None
            ),
        ),
        (
            "A original",
            _open_copy(_resolve(shared.get("normalized_path"), artifact_root)),
        ),
    ]
    for backend, label in (
        ("gfpgan", "GFPGAN"),
        ("codeformer", "CodeFormer w=1"),
        ("realesrgan_x2plus", "RealESRGAN x2"),
    ):
        record = backend_records.get(backend, {})
        entries.append(
            (
                label,
                _open_copy(_resolve(record.get("native_path"), artifact_root)),
            )
        )
    panels = [
        _panel(
            image,
            title,
            f"{frozen.get('category', (frozen.get('quality') or {}).get('category'))} "
            f"{frozen.get('eligibility')}",
            size=180,
        )
        for title, image in entries
    ]
    canvas = Image.new("RGB", (20 + 200 * len(panels), 300), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), frozen["sample_id"], fill="black", font=_font(15))
    for index, panel in enumerate(panels):
        canvas.paste(panel.resize((190, 250)), (10 + index * 200, 40))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _median_rule_quality(rows: list[dict]) -> dict | None:
    candidates = [
        row for row in rows if row.get("aligned_path") and
        (row.get("quality") or {}).get("rule_quality") is not None
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            float((row.get("quality") or {}).get("rule_quality")),
            row["sample_id"],
        )
    )
    return candidates[(len(candidates) - 1) // 2]


def render_matrix_figures(
    artifact_root: Path,
    payload: dict,
    manifest_path: Path,
    query_records: list[dict],
    gallery_records: list[dict],
    rows_by_arm: dict[str, list[dict]],
    records_by_backend: dict[str, list[dict]],
    recoverable_main: dict,
    paired: dict,
) -> list[Path]:
    """Render the complete publication figure set without optional plotting deps."""
    figure_dir = artifact_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    labels = ["A", "C1 GFPGAN", "C2 CodeFormer", "C3 RealESRGAN"]
    arms = [
        "A_original",
        "C1_gated_gfpgan",
        "C2_gated_codeformer_w1",
        "C3_gated_realesrgan_x2plus",
    ]
    values = [float(recoverable_main[arm]["rank1_rate"] or 0.0) for arm in arms]
    cis = [
        recoverable_main[arm]["rank1_uncertainty"].get(
            "pid_cluster_bootstrap_95ci"
        ) or [values[index], values[index]]
        for index, arm in enumerate(arms)
    ]
    rank_path = figure_dir / "recoverable_rank1_ci.png"
    _chart(
        rank_path,
        "Frozen recoverable cohort: Rank-1 with PID-cluster 95% CI",
        labels,
        values,
        lower=[ci[0] for ci in cis],
        upper=[ci[1] for ci in cis],
    )
    paths.append(rank_path)

    transition_path = figure_dir / "rescue_harm_transitions.png"
    transition_keys = [
        "recoverable_A_vs_C1_gated_gfpgan",
        "recoverable_A_vs_C2_gated_codeformer_w1",
        "recoverable_A_vs_C3_gated_realesrgan_x2plus",
    ]
    transition_values = [
        paired[key]["wrong_to_correct"] / max(1, paired[key]["samples"])
        for key in transition_keys
    ]
    harm_values = [
        paired[key]["correct_to_wrong"] / max(1, paired[key]["samples"])
        for key in transition_keys
    ]
    _chart(
        transition_path,
        "Rescue (blue) and harm (red) rates vs A",
        ["GFPGAN rescue", "CodeFormer rescue", "RealESRGAN rescue",
         "GFPGAN harm", "CodeFormer harm", "RealESRGAN harm"],
        transition_values + harm_values,
        colors=["#2a9d8f"] * 3 + ["#e76f51"] * 3,
    )
    paths.append(transition_path)

    score_lookup = {
        arm: {row["sample_id"]: row for row in rows}
        for arm, rows in rows_by_arm.items()
    }
    scatter_series = []
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for index, (backend, c_arm) in enumerate((
        ("gfpgan", "C1_gated_gfpgan"),
        ("codeformer", "C2_gated_codeformer_w1"),
        ("realesrgan_x2plus", "C3_gated_realesrgan_x2plus"),
    )):
        points = []
        for record in records_by_backend[backend]:
            before = record.get("fiqa_before")
            after = record.get("fiqa_after")
            a_score = score_lookup["A_original"][record["sample_id"]]
            c_score = score_lookup[c_arm][record["sample_id"]]
            if (
                before is not None
                and after is not None
                and a_score.get("margin") is not None
                and c_score.get("margin") is not None
            ):
                points.append(
                    (
                        float(after) - float(before),
                        float(c_score["margin"]) - float(a_score["margin"]),
                    )
                )
        scatter_series.append((backend, points, colors[index]))
    scatter_path = figure_dir / "fiqa_delta_margin_scatter.png"
    _scatter(
        scatter_path,
        "FIQA delta vs identity-margin delta",
        scatter_series,
        x_label="FIQA after - before",
        y_label="C margin - A margin",
    )
    paths.append(scatter_path)

    latency_path = figure_dir / "accuracy_vs_latency.png"
    latency_points = []
    for index, (backend, c_arm) in enumerate((
        ("gfpgan", "C1_gated_gfpgan"),
        ("codeformer", "C2_gated_codeformer_w1"),
        ("realesrgan_x2plus", "C3_gated_realesrgan_x2plus"),
    )):
        latencies = [
            float(row["latency_seconds"])
            for row in records_by_backend[backend]
            if row.get("latency_seconds") is not None
        ]
        latency_points.append(
            (
                backend,
                [(
                    sum(latencies) / len(latencies) if latencies else 0.0,
                    float(recoverable_main[c_arm]["rank1_rate"] or 0.0),
                )],
                colors[index],
            )
        )
    _scatter(
        latency_path,
        "Accuracy vs transform latency",
        latency_points,
        x_label="mean seconds / aligned face",
        y_label="recoverable Rank-1",
    )
    paths.append(latency_path)

    eligibility_path = figure_dir / "b_vs_c_eligibility.png"
    eligibility_labels, eligibility_values, eligibility_colors = [], [], []
    for backend_index, (b_arm, c_arm, short) in enumerate((
        ("B1_all_gfpgan", "C1_gated_gfpgan", "G"),
        ("B2_all_codeformer_w1", "C2_gated_codeformer_w1", "C"),
        ("B3_all_realesrgan_x2plus", "C3_gated_realesrgan_x2plus", "R"),
    )):
        for eligibility in ("direct", "recoverable", "unusable"):
            for arm, suffix, color in (
                (b_arm, "B", "#6c8ebf"),
                (c_arm, "C", "#82b366"),
            ):
                cohort = [
                    row
                    for row in rows_by_arm[arm]
                    if row["eligibility"] == eligibility
                ]
                eligibility_labels.append(f"{short}{suffix}-{eligibility[:3]}")
                eligibility_values.append(
                    sum(row["rank1_correct"] for row in cohort) / len(cohort)
                    if cohort
                    else 0.0
                )
                eligibility_colors.append(color)
    _chart(
        eligibility_path,
        "B vs C by frozen eligibility (B denominator is never gate-filtered)",
        eligibility_labels,
        eligibility_values,
        colors=eligibility_colors,
    )
    paths.append(eligibility_path)

    gallery_by_pid = {}
    for row in gallery_records:
        gallery_by_pid.setdefault(row["pid"], row)
    shared_by_id = {row["sample_id"]: row for row in query_records}
    backend_by_id = {
        backend: {row["sample_id"]: row for row in records}
        for backend, records in records_by_backend.items()
    }
    qualitative_rows = []
    for category in ("clear", "marginal", "poor"):
        qualitative_rows.append(
            _median_rule_quality(
                [
                    row
                    for row in payload["queries"]
                    if (row.get("quality") or {}).get("category") == category
                ]
            )
        )
    qualitative_path = figure_dir / "qualitative_grid.png"
    row_images = []
    for frozen in qualitative_rows:
        temporary = figure_dir / f".qualitative-{len(row_images)}.png"
        if frozen:
            _matrix_panel(
                temporary,
                frozen,
                shared_by_id[frozen["sample_id"]],
                gallery_by_pid.get(frozen["pid"]),
                {
                    backend: rows.get(frozen["sample_id"], {})
                    for backend, rows in backend_by_id.items()
                },
                manifest_path,
                artifact_root,
            )
            row_images.append(_open_copy(temporary))
            temporary.unlink(missing_ok=True)
        else:
            row_images.append(Image.new("RGB", (1020, 300), "#eeeeee"))
    grid = Image.new("RGB", (1020, 300 * len(row_images)), "white")
    for index, image in enumerate(row_images):
        if image:
            grid.paste(image, (0, index * 300))
    grid.save(qualitative_path)
    paths.append(qualitative_path)

    recoverable_dir = figure_dir / "all40_recoverable_panels"
    for frozen in payload["queries"]:
        if frozen.get("eligibility") != "recoverable":
            continue
        output = recoverable_dir / f"{frozen['sample_id']}.png"
        _matrix_panel(
            output,
            frozen,
            shared_by_id[frozen["sample_id"]],
            gallery_by_pid.get(frozen["pid"]),
            {
                backend: rows.get(frozen["sample_id"], {})
                for backend, rows in backend_by_id.items()
            },
            manifest_path,
            artifact_root,
        )
        paths.append(output)
    return paths
