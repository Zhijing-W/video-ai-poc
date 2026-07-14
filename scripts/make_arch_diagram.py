"""生成视频理解 PoC 架构图（矢量 SVG + 预览 PNG）。

设计目标：
- 每个组件标注真实技术名 + Azure 资源名 + 当前状态徽标
- 反映实际进度：Blob 已接入（非可选）、ACR 镜像已构建、App Service 因配额暂缓
- 输出矢量 SVG（放大不失真）+ 高分辨率 PNG（Markdown 预览）
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

# 中文字体（Windows 自带微软雅黑）；SVG 内文字转路径，跨机不缺字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

FIG_W, FIG_H = 16.0, 9.6
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ---------- 配色 ----------
C_USER = "#475569"
C_APP = "#2563eb"
C_FF = "#ea580c"
C_AI = "#16a34a"
C_JSON = "#0d9488"
C_BLOB = "#7c3aed"
C_ACR = "#4f46e5"
C_ZONE = "#3b82f6"
INK = "#0f172a"
SUB = "#475569"


def box(x, y, w, h, lines, color, fc="#ffffff", badge=None, badge_color=None):
    """画圆角框；lines=[(文字,字号,加粗,颜色),...] 从上到下排布。"""
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=2, edgecolor=color, facecolor=fc, zorder=3,
    )
    ax.add_patch(p)

    line_h = 0.30
    block = line_h * len(lines)
    cursor = y + (h + block) / 2 - line_h / 2
    for text, size, bold, col in lines:
        ax.text(x + w / 2, cursor, text, ha="center", va="center",
                fontsize=size, fontweight=("bold" if bold else "normal"),
                color=col, zorder=4)
        cursor -= line_h

    if badge:
        bc = badge_color or color
        ax.text(x + w / 2, y + 0.16, badge, ha="center", va="center",
                fontsize=8, fontweight="bold", color="#ffffff", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc=bc, ec="none"))
    return (x, y, w, h)


def arrow(p1, p2, label="", color=INK, rad=0.0, off=(0, 0), ls="-"):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=20,
        linewidth=2.2, color=color, zorder=5,
        connectionstyle=f"arc3,rad={rad}", linestyle=ls,
    )
    ax.add_patch(a)
    if label:
        mx = (p1[0] + p2[0]) / 2 + off[0]
        my = (p1[1] + p2[1]) / 2 + off[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=9.3,
                color=color, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=color, lw=0.8))


# ---------- 标题 ----------
ax.text(FIG_W / 2, 9.2, "视频理解 PoC · Phase 1 架构（LLM-first MVP）",
        ha="center", va="center", fontsize=19, fontweight="bold", color=INK)
ax.text(FIG_W / 2, 8.7,
        "Video → ffmpeg 抽帧 → Azure OpenAI (gpt-4o Vision) → 结构化 JSON",
        ha="center", va="center", fontsize=11.5, color=SUB)

# ---------- Azure 云区域（虚线分组）----------
zone = FancyBboxPatch(
    (3.2, 1.95), 12.45, 6.15,
    boxstyle="round,pad=0.02,rounding_size=0.12",
    linewidth=2, edgecolor=C_ZONE, facecolor="#f0f7ff",
    linestyle=(0, (6, 4)), zorder=1,
)
ax.add_patch(zone)
ax.text(3.45, 7.85, "Azure 云  ·  资源组 rg-video-understanding-poc  ·  区域 japaneast（东京）",
        ha="left", va="center", fontsize=10.5, fontweight="bold", color=C_ZONE)

# ---------- 节点 ----------
# 用户（云外）
box(0.3, 5.3, 2.55, 1.5, [
    ("用户 / 浏览器", 12.5, True, INK),
    ("Client（HTML + JS）", 9.0, False, SUB),
    ("上传视频 · 轮询结果", 8.8, False, SUB),
], C_USER)

# App Service / FastAPI（上行左）
box(3.6, 4.7, 3.4, 2.6, [
    ("FastAPI 后端", 13.0, True, INK),
    ("Uvicorn / Gunicorn", 9.2, False, C_APP),
    ("App Service for Containers", 8.6, False, SUB),
    ("app-zhijing-video", 8.6, False, SUB),
    ("POST /upload-video", 8.4, False, "#64748b"),
    ("GET /status · /result", 8.4, False, "#64748b"),
], C_APP, fc="#eff6ff", badge="配额暂缓 · 现本地 uvicorn", badge_color="#d97706")

# ffmpeg（上行中）
box(7.6, 5.5, 2.85, 1.7, [
    ("ffmpeg 抽帧", 12.5, True, INK),
    ("video_processor.py", 8.8, False, C_FF),
    ("fps=1/5 · scale=768", 8.6, False, SUB),
    ("最多 8 帧 → JPEG", 8.6, False, SUB),
], C_FF, fc="#fff7ed", badge="已实现", badge_color=C_FF)

# Azure OpenAI（上行右）
box(11.05, 5.5, 3.3, 1.7, [
    ("Azure OpenAI", 12.5, True, INK),
    ("gpt-4o Vision · 2024-11-20", 8.6, False, C_AI),
    ("GlobalStandard · 东京", 8.6, False, SUB),
    ("aoai-zhijing-video-jpe", 8.4, False, SUB),
], C_AI, fc="#f0fdf4", badge="已部署", badge_color=C_AI)

# ACR 镜像仓库（下行左）
box(3.6, 2.5, 3.4, 1.6, [
    ("ACR 容器镜像仓库", 11.5, True, INK),
    ("Azure Container Registry", 8.6, False, C_ACR),
    ("acrzhijingvideo · video-understanding:v1", 8.0, False, SUB),
], C_ACR, fc="#eef2ff", badge="镜像已构建并推送", badge_color=C_ACR)

# Blob 存储（下行中，已接入）
box(7.6, 2.5, 2.85, 1.6, [
    ("Azure Blob 存储", 12.0, True, INK),
    ("Blob Storage（StorageV2）", 8.4, False, C_BLOB),
    ("stzhijingvideo", 8.6, False, SUB),
    ("原视频 + result.json", 8.4, False, SUB),
], C_BLOB, fc="#faf5ff", badge="已接入", badge_color=C_BLOB)

# 结构化 JSON（下行右）
box(11.05, 2.5, 3.3, 1.6, [
    ("结构化 JSON 输出", 12.0, True, INK),
    ("response_format = json_object", 8.2, False, C_JSON),
    ("summary · objects · events", 8.4, False, SUB),
    ("notification · confidence · evidence", 7.8, False, SUB),
], C_JSON, fc="#f0fdfa", badge="已实现", badge_color=C_JSON)

# ---------- 箭头（主流程 ①~⑤）----------
arrow((2.85, 6.05), (3.6, 6.05), "① 上传视频\n(multipart)", INK, off=(0, 0.45))
arrow((7.0, 6.3), (7.6, 6.3), "② 抽帧", C_FF, off=(0, 0.3))
arrow((10.45, 6.35), (11.05, 6.35), "③ 关键帧\n+ 提示词", C_AI, off=(0, 0.45))
arrow((12.7, 5.5), (12.7, 4.1), "④ 强制 JSON", C_JSON, off=(1.05, 0))
# ⑤ JSON → 用户：沿底部走线，绕开所有框（L 形折线）
ret_verts = [(11.5, 2.5), (11.5, 1.7), (1.45, 1.7), (1.45, 5.28)]
ret_path = MplPath(ret_verts, [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO])
ax.add_patch(FancyArrowPatch(
    path=ret_path, arrowstyle="-|>", mutation_scale=20,
    linewidth=2.2, color=C_USER, zorder=5,
))
ax.text(6.4, 1.7, "⑤ 返回结果 / 轮询 job_id 取 result", ha="center", va="center",
        fontsize=9.3, color=C_USER, fontweight="bold", zorder=7,
        bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=C_USER, lw=0.8))
# ACR → App（镜像部署，向上）
arrow((4.6, 4.1), (4.6, 4.7), "镜像部署\n(docker pull)", C_ACR, off=(-1.0, 0))
# App → Blob（存档，已接入实线）
arrow((6.2, 4.7), (8.5, 4.1), "存档原视频 + 结果", C_BLOB, rad=-0.1, off=(0.15, 0.3))

# ---------- 脚注 ----------
ax.text(0.3, 1.35,
        "Phase 1：LLM-first MVP —— 定时抽帧（笨办法但快速验证「gpt-4o 能看懂视频」的可行性）。",
        ha="left", va="center", fontsize=9, color="#94a3b8", style="italic")
ax.text(0.3, 1.02,
        "本图只画 Phase 1 本阶段架构；省钱/可控的演进（YOLO + 事件门控 + 级联）见 architecture-phase2。",
        ha="left", va="center", fontsize=9, color="#94a3b8", style="italic")
ax.text(15.65, 1.02, "徽标：绿=已实现 · 紫=已接入 · 橙=暂缓",
        ha="right", va="center", fontsize=8.5, color="#94a3b8")

out_dir = Path(__file__).resolve().parents[2] / "assets"
out_dir.mkdir(parents=True, exist_ok=True)
svg_path = out_dir / "architecture-phase1.svg"
png_path = out_dir / "architecture-phase1.png"
fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
fig.savefig(png_path, dpi=170, bbox_inches="tight", facecolor="white")
print(f"saved: {svg_path}")
print(f"saved: {png_path}")
