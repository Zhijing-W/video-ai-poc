"""从协议 A 与 B3 JSON 生成论文结果图（SVG + PNG）。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
RUNS = EXPERIMENT_DIR / "results" / "runs"
OUT = EXPERIMENT_DIR / "results" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    }
)

COLORS = {
    "face": "#4C78A8",
    "body": "#F58518",
    "gait": "#54A24B",
    "arc": "#4C78A8",
    "ada": "#E45756",
    "sr": "#72B7B2",
    "gray": "#9D9D9D",
}


def latest(pattern: str) -> Path:
    paths = sorted(RUNS.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"找不到结果：{RUNS / pattern}")
    return paths[-1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, stem: str):
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def pct_axis(ax, maximum=1.0):
    ax.set_ylim(0, maximum)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", alpha=0.25)


def label_bars(ax, bars, values, decimals=1):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{100 * value:.{decimals}f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_a_coverage(a: dict):
    values = [
        a["results"]["F"]["coverage"]["face"] / 27,
        a["results"]["B"]["coverage"]["body"] / 27,
        a["results"]["G"]["coverage"]["gait"] / 27,
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(
        ["人脸", "人形", "步态"],
        values,
        color=[COLORS["face"], COLORS["body"], COLORS["gait"]],
        width=0.58,
    )
    label_bars(ax, bars, values)
    ax.set_title("协议A：严格质量门控后的建档覆盖率")
    ax.set_ylabel("成功建立参考特征的身份比例")
    ax.text(0, values[0] / 2, "1/27", ha="center", color="white", weight="bold")
    ax.text(1, values[1] / 2, "27/27", ha="center", color="white", weight="bold")
    ax.text(2, values[2] / 2, "23/27", ha="center", color="white", weight="bold")
    pct_axis(ax, 1.08)
    save(fig, "figure_a_coverage")


def plot_a_modalities(a: dict):
    arms = ["F", "B", "G", "FB", "BG", "FBG"]
    rank1 = [a["results"][arm]["summary"]["rank1_rate"] for arm in arms]
    tpir5 = [
        a["results"][arm]["summary"]["operating_points"]["0.050"]["tpir"]
        for arm in arms
    ]
    x = np.arange(len(arms))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    b1 = ax.bar(x - width / 2, rank1, width, label="强制选第一名时认对（Rank-1）", color="#4C78A8")
    b2 = ax.bar(x + width / 2, tpir5, width, label="误识率≤5%时正确接受（TPIR）", color="#F58518")
    label_bars(ax, b1, rank1)
    label_bars(ax, b2, tpir5)
    ax.set_xticks(x, ["仅人脸", "仅人形", "仅步态", "人脸+人形", "人形+步态", "三路"])
    ax.set_title("协议A：不同身份线索组合的识别表现")
    ax.set_ylabel("识别比例")
    ax.legend(loc="upper right")
    pct_axis(ax, 0.66)
    save(fig, "figure_a_modalities")


def plot_a_quality(a: dict):
    bins = ["clear", "marginal", "poor", "none"]
    labels = ["清晰脸", "一般质量", "低质量脸", "未检测到脸"]
    data = a["results"]["B"]["summary"]["by_quality_bin"]
    rank1 = [data[name]["rank1_rate"] or 0 for name in bins]
    tpir5 = [data[name].get("tpir_at_fmr_0.050") or 0 for name in bins]
    counts = [data[name]["total"] for name in bins]
    x = np.arange(len(bins))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    b1 = ax.bar(
        x - width / 2,
        rank1,
        width,
        color="#4C78A8",
        label="强制选择第一名时的身份正确率",
    )
    b2 = ax.bar(
        x + width / 2,
        tpir5,
        width,
        color="#F58518",
        label="陌生人误识率≤5%时，库内人员正确识别并接受率",
    )
    label_bars(ax, b1, rank1)
    label_bars(ax, b2, tpir5)
    ax.set_xticks(x, [f"{label}\n(n={count})" for label, count in zip(labels, counts)])
    ax.set_title("协议A：人形路线在不同人脸质量场景中的表现")
    ax.set_ylabel("识别比例")
    ax.legend()
    pct_axis(ax, 0.68)
    save(fig, "figure_a_quality_buckets")


def plot_a_scores(a: dict):
    specs = [("F", "face", "人脸"), ("B", "body", "人形"), ("G", "gait", "步态")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, (arm, route, title) in zip(axes, specs):
        rows = a["results"][arm]["rows"]
        genuine = [
            row["routes"][route]["score"]
            for row in rows
            if row["genuine"] and route in row.get("routes", {})
        ]
        imposters = [
            row["routes"][route]["score"]
            for row in rows
            if not row["genuine"] and route in row.get("routes", {})
        ]
        boxes = ax.boxplot(
            [genuine, imposters],
            tick_labels=["库内同人", "库外陌生人"],
            patch_artist=True,
            widths=0.55,
            showfliers=False,
        )
        boxes["boxes"][0].set_facecolor("#4C78A8")
        boxes["boxes"][1].set_facecolor("#E45756")
        for box in boxes["boxes"]:
            box.set_alpha(0.75)
        ax.set_title(f"{title}最高相似度")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 1.03)
    axes[0].set_ylabel("相似度")
    fig.suptitle("协议A：库内同人与库外陌生人的分数是否能分开", y=1.03, fontsize=13)
    save(fig, "figure_a_score_distributions")


def plot_b3_rank1(b3: dict):
    aggregate = b3["test"]["aggregate"]
    variants = ["FR0", "FR1", "FR2", "FR3"]
    labels = ["ArcFace", "AdaFace", "ArcFace+\nGFPGAN", "AdaFace+\nGFPGAN"]
    means = [aggregate[name]["rank1"]["mean"] for name in variants]
    lows = [aggregate[name]["rank1"]["ci95_low"] for name in variants]
    highs = [aggregate[name]["rank1"]["ci95_high"] for name in variants]
    errors = np.array(
        [[mean - low for mean, low in zip(means, lows)], [high - mean for mean, high in zip(means, highs)]]
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(
        labels,
        means,
        yerr=errors,
        capsize=5,
        color=[COLORS["arc"], COLORS["ada"], COLORS["sr"], "#B279A2"],
        width=0.62,
    )
    label_bars(ax, bars, means)
    ax.set_title("协议B3：强制选择第一名时的身份正确率（5次划分，95%置信区间）")
    ax.set_ylabel("成功提取人脸后，第一名身份正确的比例")
    pct_axis(ax, 0.48)
    save(fig, "figure_b3_rank1_ci")


def plot_b3_tpir_fmr(b3: dict):
    aggregate = b3["test"]["aggregate"]
    variants = ["FR0", "FR1", "FR2", "FR3"]
    labels = ["ArcFace", "AdaFace", "ArcFace+GFPGAN", "AdaFace+GFPGAN"]
    targets = ["0.010", "0.050", "0.100"]
    target_labels = [
        "严格档\n校准陌生人误识≤1%",
        "平衡档\n校准陌生人误识≤5%",
        "宽松档\n校准陌生人误识≤10%",
    ]
    x = np.arange(len(variants))
    width = 0.24
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for index, (target, label) in enumerate(zip(targets, target_labels)):
        tpir = [aggregate[name]["operating_points"][target]["tpir"]["mean"] for name in variants]
        fmr = [aggregate[name]["operating_points"][target]["actual_fmr"]["mean"] for name in variants]
        bars1 = ax1.bar(x + (index - 1) * width, tpir, width, label=label)
        bars2 = ax2.bar(x + (index - 1) * width, fmr, width, label=label)
        label_bars(ax1, bars1, tpir)
        label_bars(ax2, bars2, fmr)
    ax1.set_title("使用训练人员校准并固定阈值后的测试结果")
    ax1.set_ylabel("库内人员正确识别并接受率")
    ax2.set_title("同一固定阈值下的测试陌生人误识率")
    ax2.set_ylabel("库外人员被错误接受的比例")
    for ax in (ax1, ax2):
        ax.set_xticks(x, labels, rotation=12)
        ax.legend()
        pct_axis(ax, 0.16)
    save(fig, "figure_b3_tpir_and_actual_fmr")


def plot_b3_coverage(b3: dict):
    repetitions = b3["test"]["repetitions"]
    genuine_total = np.mean(
        [rep["results"]["FR0"]["face_query_coverage"]["genuine_total"] for rep in repetitions]
    )
    genuine_scorable = np.mean(
        [rep["results"]["FR0"]["face_query_coverage"]["genuine_scorable"] for rep in repetitions]
    )
    imposter_total = np.mean(
        [rep["results"]["FR0"]["face_query_coverage"]["imposter_total"] for rep in repetitions]
    )
    imposter_scorable = np.mean(
        [rep["results"]["FR0"]["face_query_coverage"]["imposter_scorable"] for rep in repetitions]
    )
    values = [genuine_scorable / genuine_total, imposter_scorable / imposter_total]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(
        ["库内人员轨迹", "库外陌生人轨迹"],
        values,
        color=["#4C78A8", "#E45756"],
        width=0.55,
    )
    label_bars(ax, bars, values)
    ax.set_title("协议B3：测试轨迹中能够提取人脸特征的比例")
    ax.set_ylabel("可评分轨迹 / 全部轨迹")
    ax.text(
        0,
        values[0] / 2,
        f"{genuine_scorable:.1f}/{genuine_total:.1f}",
        ha="center",
        color="white",
        weight="bold",
    )
    ax.text(
        1,
        values[1] / 2,
        f"{imposter_scorable:.1f}/{imposter_total:.1f}",
        ha="center",
        color="white",
        weight="bold",
    )
    pct_axis(ax, 1.0)
    save(fig, "figure_b3_face_query_coverage")


def plot_b3_degradation(b3: dict):
    tags = b3["test"]["aggregate"]["FR0"]["by_degradation_tag"]
    order = ["clear", "small-face", "pose", "blur"]
    labels = ["清晰脸", "小脸", "姿态偏转", "模糊"]
    counts = [tags[tag]["mean_tracks_per_repeat"] for tag in order]
    rank1 = [tags[tag]["rank1"]["mean"] or 0 for tag in order]
    fig, ax1 = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(order))
    bars = ax1.bar(x, counts, color=["#54A24B", "#F58518", "#B279A2", "#72B7B2"], alpha=0.82)
    ax1.set_ylabel("每次实验的平均轨迹数")
    ax1.set_xticks(x, labels)
    ax1.set_title("协议B3：主要人脸退化类型与第一名身份正确率")
    ax2 = ax1.twinx()
    ax2.plot(x, rank1, color="#4C78A8", marker="o", linewidth=2.2, label="第一名身份正确率")
    ax2.set_ylabel("第一名身份正确率")
    ax2.set_ylim(0, 0.65)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7, f"{count:.1f}", ha="center")
    for xi, value in zip(x, rank1):
        ax2.text(xi, value + 0.035, f"{100 * value:.1f}%", ha="center", color="#4C78A8")
    ax1.grid(axis="y", alpha=0.2)
    save(fig, "figure_b3_degradation_tags")


def plot_b3_gfpgan(b3: dict):
    first_repeat = b3["test"]["repetitions"][0]["results"]["FR2"]
    attempted = first_repeat["superres_attempted_probe_frames"]
    enhanced = first_repeat["enhanced_probe_frames"]
    paired = [
        rep["superres_paired"]["FR2_vs_FR0"]
        for rep in b3["test"]["repetitions"]
    ]
    changed = sum(item["prediction_changed"] for item in paired)
    improved = sum(item["rank1_improved"] for item in paired)
    degraded = sum(item["rank1_degraded"] for item in paired)
    values = [attempted, enhanced, changed, improved, degraded]
    labels = ["独立尝试帧", "独立增强帧", "5次划分中预测变化", "由错变对", "由对变错"]
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    bars = ax.bar(labels, values, color=["#9D9D9D", "#72B7B2", "#F2CF5B", "#54A24B", "#E45756"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.35, str(value), ha="center", fontsize=10)
    ax.set_title("协议B3：GFPGAN实际触发及对识别结果的影响")
    ax.set_ylabel("数量")
    ax.set_ylim(0, max(values + [1]) * 1.25)
    ax.grid(axis="y", alpha=0.22)
    save(fig, "figure_b3_gfpgan_effect")


def main():
    a_path = latest("mevid_e2e_e27_i25_*.json")
    b3_path = latest("mevid_face_b3_train50_r5_test27_*.json")
    a = load(a_path)
    b3 = load(b3_path)
    plot_a_coverage(a)
    plot_a_modalities(a)
    plot_a_quality(a)
    plot_a_scores(a)
    plot_b3_rank1(b3)
    plot_b3_tpir_fmr(b3)
    plot_b3_coverage(b3)
    plot_b3_degradation(b3)
    plot_b3_gfpgan(b3)
    print(f"协议A：{a_path}")
    print(f"协议B3：{b3_path}")
    print(f"输出目录：{OUT}")


if __name__ == "__main__":
    main()
