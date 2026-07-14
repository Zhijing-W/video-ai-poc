# -*- coding: utf-8 -*-
"""从已存的结果 JSON 生成汇报用图（无需重跑评测，秒出）。

产出 results/ 下：
  fig_ablation.svg/.png   —— 各 arm × 各质量桶 Rank-1 总览（完整消融）
  fig_headline.svg/.png   —— 头条：S0 仅人脸 vs S5 +人形（无脸/极糊/总体）
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

RES = Path(__file__).resolve().parent.parent / "results" / "legacy_market"
data = json.loads((RES / "face_blur_eval_results.json").read_text(encoding="utf-8"))
arms = data["arms"]
BIN_LABEL = {"clear": "清晰", "marginal": "勉强", "poor": "不合格", "none": "无脸", "overall": "总体"}


def _r1(arm, b):
    d = arms[arm]["overall"] if b == "overall" else arms[arm]["by_bin"].get(b)
    return d.get("rank1") if d else None


def _present_bins():
    """取实际有样本的质量桶（避免画空桶），顺序 clear→marginal→poor→none。"""
    present = [b for b in ["clear", "marginal", "poor", "none"]
               if any((arms[a]["by_bin"].get(b) or {}).get("total", 0) > 0 for a in arms)]
    return present + ["overall"]


# ---------- 图1：完整消融总览 ----------
def fig_ablation():
    bins = _present_bins()
    order = list(arms.keys())
    x = np.arange(len(bins))
    w = 0.8 / len(order)
    palette = {"S0": "#9AA0A6", "S1": "#F7630C", "S2": "#FFB900", "S5": "#5C2D91", "full": "#107C10"}
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, a in enumerate(order):
        vals = [(_r1(a, b) or 0) for b in bins]
        bars = ax.bar(x + i * w, vals, w, label=f"{a} · {arms[a]['note']}", color=palette.get(a, "#0078D4"))
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1.5, f"{v:.0f}",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x + w * (len(order) - 1) / 2)
    ax.set_xticklabels([BIN_LABEL[b] for b in bins], fontsize=12)
    ax.set_ylabel("Rank-1 准确率 (%)", fontsize=12)
    ax.set_ylim(0, 108)
    ax.set_title(f"糊脸消融 · 各方案按人脸质量分桶 Rank-1（{data['dataset']}，{data['max_subjects']}身份）",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.99, -0.12, "注：桶按客户人脸过滤逻辑（模糊+角度）分级 clear/marginal/poor + 无脸；空桶已省略",
            transform=ax.transAxes, ha="right", fontsize=8.5, color="#888888")
    plt.tight_layout()
    fig.savefig(RES / "fig_ablation.svg", facecolor="white")
    fig.savefig(RES / "fig_ablation.png", dpi=140, facecolor="white")


# ---------- 图2：头条 S0 vs S5 ----------
def fig_headline():
    bins = [b for b in _present_bins() if b != "clear"]  # 差脸桶 + 总体（clear 作对照另表）
    s0 = [(_r1("S0", b) or 0) for b in bins]
    s5 = [(_r1("S5", b) or 0) for b in bins]
    x = np.arange(len(bins))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    b1 = ax.bar(x - w / 2, s0, w, label="仅人脸（baseline）", color="#D83B01")
    b2 = ax.bar(x + w / 2, s5, w, label="人脸 + 人形兜底", color="#107C10")
    for bars in (b1, b2):
        for rect in bars:
            v = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1.5, f"{v:.0f}%",
                    ha="center", fontsize=12, fontweight="bold")
    # 增益标注：放在两柱中间上方，避免压住图例/柱子
    for xi, a, b in zip(x, s0, s5):
        if b - a > 3:
            ax.text(xi, max(a, b) + 6, f"↑ +{b - a:.0f}", ha="center",
                    fontsize=12, fontweight="bold", color="#107C10")
    ax.set_xticks(x)
    ax.set_xticklabels([BIN_LABEL[b] for b in bins], fontsize=13)
    ax.set_ylabel("身份识别 Rank-1 (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title("脸糊到认不出时，人形把身份救回来", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper center", ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(RES / "fig_headline.svg", facecolor="white")
    fig.savefig(RES / "fig_headline.png", dpi=140, facecolor="white")


fig_ablation()
fig_headline()
print("[OK] fig_ablation.svg/.png")
print("[OK] fig_headline.svg/.png")
