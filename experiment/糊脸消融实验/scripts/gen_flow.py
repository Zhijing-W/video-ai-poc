# -*- coding: utf-8 -*-
"""生成「糊脸对比实验」流程图。

Output:
    docs/实验流程.svg (canonical)
    docs/实验流程.png (preview)

约定（与 phase4 logic flow 一致）：SVG 为准、节点标题加粗 + 小字说明、同类节点用容器分组、
留白无重叠、未就绪能力（步态）用虚线灰框预留。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

BLUE = ("#CFE4FA", "#0078D4")
ORANGE = ("#FFF4CE", "#F7630C")
GREEN = ("#DFF6DD", "#107C10")
PURPLE = ("#E8DAEF", "#5C2D91")
TEAL = ("#C5F0F5", "#0C8599")
GRAY = ("#E3E3E3", "#495057")
RESERVED = ("#F2F2F2", "#9AA0A6")

XMAX, YMAX = 1520, 1380
fig, ax = plt.subplots(figsize=(15.2, 13.8))
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.invert_yaxis()
ax.axis("off")


def box(cx, cy, w, h, text, color, dashed=False, reserved=False):
    fc, ec = color
    ls = (0, (5, 4)) if dashed else "solid"
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=16",
        fc=fc, ec=ec, lw=2.0, linestyle=ls, zorder=3))
    lines = text.split("\n")
    title, details = lines[0], lines[1:]
    tcolor = "#4A4A4A" if reserved else "#111111"
    dcolor = "#666666" if reserved else "#333333"
    if details:
        ax.text(cx, cy - h * 0.30, title, ha="center", va="center",
                fontsize=15.5, fontweight="bold", color=tcolor, zorder=4)
        ax.text(cx, cy + h * 0.13, "\n".join(details), ha="center", va="center",
                fontsize=11.6, color=dcolor, linespacing=1.4, zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=15.5, fontweight="bold", color=tcolor, zorder=4)


def container(x, y, w, h, label, color):
    fc, ec = color
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=20",
        fc=fc, ec=ec, lw=1.6, linestyle=(0, (2, 3)), alpha=0.30, zorder=1))
    ax.text(x + 20, y + 28, label, ha="left", va="center",
            fontsize=14.5, fontweight="bold", color=ec, zorder=2)


def arrow(x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#444444",
                                shrinkA=0, shrinkB=0), zorder=2)
    if text:
        ax.text((x1 + x2) / 2 + 12, (y1 + y2) / 2, text,
                fontsize=11, color="#444444", ha="left", va="center", zorder=5)


CX = 760
ax.text(CX, 44, "糊脸消融实验 · 流程图（Market-1501 · 先按脸质量分桶，再比各 arm 认人）",
        ha="center", va="center", fontsize=21, fontweight="bold", color="#111111")

# ① 数据源
box(CX, 142, 1080, 92,
    "① 数据源 · Market-1501（真实监控低清人形 crop + 身份标注）\n"
    "按身份取图、每人 cap N 张；图片→跳过抽帧；身份跟「人」走 → 糊脸样本也有真值标签", BLUE)
arrow(CX, 188, CX, 232)

# ========== 阶段 A：只用人脸做质量分桶 ==========
container(70, 232, 1380, 290, "阶段 A · 人脸质量分桶（只用人脸，不碰人形/步态）", ORANGE)
box(CX, 320, 980, 96,
    "② 调产品 face.detect(with_quality)  ← 只跑人脸\n"
    "检测脸 + assess_quality（det_score / 尺寸 / 正脸度 / 清晰度）；人形步态完全不参与判质量", ORANGE)
arrow(CX, 368, CX, 410)
box(CX, 462, 980, 88,
    "③ 按质量分桶 clear · blur · tiny · none（无脸）\n"
    "差脸桶 = blur + tiny + none  ← 实验主战场（清晰桶留作对照）", ORANGE)
arrow(CX, 522, CX, 566)

# ④ gallery/probe
box(CX, 612, 1080, 88,
    "④ 划分 gallery / probe\n"
    "Gallery = 每人最清晰几张 → 身份模板；Probe = 其余图（在差脸桶上评）", TEAL)
arrow(CX, 656, CX, 700)

# ========== 阶段 B：arm 矩阵（各 arm 自带特征提取，人形在 S5/全栈内部） ==========
container(40, 700, 1440, 360,
          "阶段 B · arm 矩阵：在差脸桶上比「认人」准确率（每个 arm 各自提特征 + 配置产品开关）", GRAY)
ax.text(60, 760,
        "A 做强脸（橙）= 救 blur 桶；  B 跨模态兜底（紫）= 连 none 桶也救；  全栈（绿）= 天花板",
        ha="left", va="center", fontsize=11.5, color="#555555", zorder=2)
ABY, ABH, ABW = 900, 150, 255
box(200, ABY, ABW, ABH, "S0 baseline\n纯 ArcFace 仅人脸\n（差脸桶易崩）", GRAY)
box(480, ABY, ABW, ABH, "S1 +AdaFace\n换质量自适应识别\n重提脸（做强脸）", ORANGE)
box(760, ABY, ABW, ABH, "S2 +超分\nGFPGAN 糊脸预处理\n重提脸（做强脸）", ORANGE)
box(1040, ABY, ABW, ABH, "S5 +人形\narm 内部才提 ReID\n脸糊退人形兜底", PURPLE)
box(1320, ABY, ABW, ABH, "全栈\n超分+AdaFace+人形\n全开（天花板）", GREEN)
ax.text(CX, 1020, "（步态 gait 预留：需视频 + GREW 权重，到位后作为第二个兜底 arm 加入）",
        ha="center", va="center", fontsize=10.8, color="#888888", style="italic", zorder=2)
arrow(CX, 1060, CX, 1102)

# ⑤ 打分（复用 eval_phase3 评分，产品里没有）
box(CX, 1156, 1180, 92,
    "⑤ 按桶打分（评测胶水，非产品代码）\n"
    "各 arm 的 subject_id / 比对结果 vs 真值 → 差脸桶 Rank-1；复用 eval_phase3 评分逻辑", GREEN)
arrow(CX, 1202, CX, 1244)

# ⑥ 产出
box(CX, 1294, 1180, 92,
    "⑥ 出表 + 出图\n"
    "差脸桶 baseline vs S1/S2/S5 vs 全栈 的 Rank-1 → results/（每手段救回多少 + 全栈天花板）", GREEN)

OUT = Path(__file__).resolve().parent.parent / "results" / "legacy_market"
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "实验流程.svg", facecolor="white")
fig.savefig(OUT / "实验流程.png", dpi=130, facecolor="white")
print("[OK] ->", OUT / "实验流程.svg")
print("[OK] ->", OUT / "实验流程.png")
