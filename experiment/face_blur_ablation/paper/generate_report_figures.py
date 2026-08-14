from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageOps


ARMS = {
    "A": "A_original",
    "B1": "B1_all_gfpgan",
    "C1": "C1_gated_gfpgan",
    "B2": "B2_all_codeformer_w1",
    "C2": "C2_gated_codeformer_w1",
    "B3": "B3_all_realesrgan_x2plus",
    "C3": "C3_gated_realesrgan_x2plus",
}

QUALITATIVE_SAMPLES = (
    ("query_0280_c508_o1_t1686", "0280", "marginal/direct"),
    ("query_0277_c507_o6_t1111", "0277", "poor/recoverable"),
    ("query_0297_c639_o3_t0496", "0297", "poor/unusable"),
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_true(value: str) -> bool:
    return value.strip().lower() == "true"


def _save_figure(fig: plt.Figure, path: Path, *, pad_inches: float) -> None:
    fig.savefig(
        path,
        format=path.suffix.removeprefix("."),
        bbox_inches="tight",
        pad_inches=pad_inches,
    )
    if path.suffix == ".svg":
        text = path.read_text(encoding="utf-8")
        path.write_text(
            "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
            encoding="utf-8",
        )


def _save_rank_chart(rows: list[dict[str, str]], output_dir: Path) -> None:
    aligned_ids = {
        row["sample_id"]
        for row in rows
        if row["arm"] == "A_original" and row["score"].strip()
    }
    denominator = len(aligned_ids)
    if denominator != 156:
        raise ValueError(f"Expected 156 aligned queries, found {denominator}")

    rows_by_arm = {
        arm: {
            row["sample_id"]: row
            for row in rows
            if row["arm"] == arm and row["sample_id"] in aligned_ids
        }
        for arm in ARMS.values()
    }
    labels = list(ARMS)
    rank1 = [
        100
        * sum(_is_true(row["rank1_correct"]) for row in rows_by_arm[ARMS[label]].values())
        / denominator
        for label in labels
    ]
    rank5 = [
        100
        * sum(_is_true(row["rank5_correct"]) for row in rows_by_arm[ARMS[label]].values())
        / denominator
        for label in labels
    ]

    colors = [
        "#4C78A8",
        "#E45756",
        "#F2A6A4",
        "#E6A32F",
        "#F3D18A",
        "#54A24B",
        "#A6D49F",
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8.0,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.0), sharex=True)
    for axis, values, title in zip(axes, (rank1, rank5), ("Rank-1", "Rank-5")):
        positions = list(range(len(labels)))
        bars = axis.barh(
            positions,
            values,
            color=colors,
            edgecolor="#444444",
            linewidth=0.45,
            height=0.68,
        )
        axis.axvline(
            values[0],
            color="#333333",
            linestyle=(0, (4, 3)),
            linewidth=1.0,
            label=f"A reference ({values[0]:.1f}%)",
        )
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlim(0, 70)
        axis.set_title(title, loc="left", pad=2, weight="bold")
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.legend(loc="lower right", frameon=False, fontsize=7.3)
        for bar, value in zip(bars, values):
            axis.text(
                value + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                ha="left",
                fontsize=7.4,
            )
    axes[-1].set_xlabel("Correct queries among the 156 aligned queries (%)")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.12, hspace=0.24)
    for suffix in ("svg", "pdf"):
        _save_figure(
            fig,
            output_dir / f"all_aligned_rank1_rank5.{suffix}",
            pad_inches=0.02,
        )
    plt.close(fig)


def _open_square(path: Path, size: int = 320) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.fit(
            opened.convert("RGB"),
            (size, size),
            method=Image.Resampling.LANCZOS,
        )


def _gallery_path(artifact_root: Path, pid: str) -> Path:
    candidates = sorted(
        (artifact_root / "images" / "shared" / "gallery").glob(
            f"checkin_{pid}_*.png"
        )
    )
    if not candidates:
        raise FileNotFoundError(f"No Gallery image for PID {pid}")
    return candidates[0]


def _query_image_paths(artifact_root: Path, sample_id: str) -> list[Path]:
    filename = f"{sample_id}.png"
    return [
        artifact_root / "images" / "shared" / "query" / filename,
        artifact_root / "images" / "gfpgan" / "native" / filename,
        artifact_root / "images" / "codeformer" / "native" / filename,
        artifact_root / "images" / "realesrgan_x2plus" / "native" / filename,
    ]


def _status_text(row: dict[str, str]) -> tuple[str, str]:
    score = float(row["score"])
    correct = _is_true(row["rank1_correct"])
    accepted = score >= 0.45
    correctness = "R1 correct" if correct else "R1 wrong"
    decision = "accept" if accepted else "reject"
    if accepted:
        color = "#0072B2"
    elif correct:
        color = "#D55E00"
    else:
        color = "#9C3D72"
    return f"{score:.3f} | {correctness} | {decision}", color


def _save_qualitative_grid(
    rows: list[dict[str, str]], artifact_root: Path, output_dir: Path
) -> None:
    indexed = {(row["sample_id"], row["arm"]): row for row in rows}
    columns = (
        ("Gallery", None),
        ("A original", "A_original"),
        ("B1 GFPGAN", "B1_all_gfpgan"),
        ("B2 CodeFormer", "B2_all_codeformer_w1"),
        ("B3 Real-ESRGAN", "B3_all_realesrgan_x2plus"),
    )
    fig = plt.figure(figsize=(7.2, 5.25))
    grid = GridSpec(
        4,
        6,
        figure=fig,
        width_ratios=(0.95, 1, 1, 1, 1, 1),
        height_ratios=(0.13, 1, 1, 1),
        hspace=0.30,
        wspace=0.07,
    )
    fig.text(
        0.012,
        0.987,
        "Product threshold: Top-1 similarity score >= 0.45",
        ha="left",
        va="top",
        fontsize=9.3,
        weight="bold",
    )
    for column_index, (label, _) in enumerate(columns, start=1):
        axis = fig.add_subplot(grid[0, column_index])
        axis.axis("off")
        axis.text(
            0.5,
            0.12,
            label,
            ha="center",
            va="bottom",
            fontsize=8.3,
            weight="bold",
        )

    row_letters = ("a", "b", "c")
    for row_index, ((sample_id, pid, quality), letter) in enumerate(
        zip(QUALITATIVE_SAMPLES, row_letters), start=1
    ):
        sample_parts = sample_id.split("_")
        wrapped_sample = (
            f"{sample_parts[0]}_{sample_parts[1]}_\n"
            f"{'_'.join(sample_parts[2:])}"
        )
        label_axis = fig.add_subplot(grid[row_index, 0])
        label_axis.axis("off")
        label_axis.text(
            0.02,
            0.93,
            f"({letter}) {quality}\n{wrapped_sample}",
            ha="left",
            va="top",
            fontsize=6.5,
            linespacing=1.18,
        )

        image_paths = [_gallery_path(artifact_root, pid), *_query_image_paths(
            artifact_root, sample_id
        )]
        for column_index, ((_, arm), image_path) in enumerate(
            zip(columns, image_paths), start=1
        ):
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            axis = fig.add_subplot(grid[row_index, column_index])
            axis.imshow(_open_square(image_path))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#A8A8A8")
                spine.set_linewidth(0.55)
            if arm is None:
                status, color = f"GT identity {pid}", "#4D4D4D"
            else:
                status, color = _status_text(indexed[(sample_id, arm)])
            axis.text(
                0.5,
                -0.075,
                status,
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=5.7,
                color=color,
                weight="bold",
            )

    fig.subplots_adjust(left=0.012, right=0.995, top=0.95, bottom=0.025)
    for suffix in ("svg", "pdf"):
        _save_figure(
            fig,
            output_dir / f"qualitative_scores_grid.{suffix}",
            pad_inches=0.025,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(artifact_root / "per_sample_long.csv")
    _save_rank_chart(rows, output_dir)
    _save_qualitative_grid(rows, artifact_root, output_dir)


if __name__ == "__main__":
    main()
