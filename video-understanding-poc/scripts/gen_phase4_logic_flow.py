# -*- coding: utf-8 -*-
"""Generate the Phase 4 branching decision-tree logic-flow diagram.

Output:
    docs/phase4-logic-flow.svg (canonical)
    docs/phase4-logic-flow.png (preview)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

BLUE = ("#CFE4FA", "#0078D4")
ORANGE = ("#FFF4CE", "#F7630C")
GREEN = ("#DFF6DD", "#107C10")
PURPLE = ("#E8DAEF", "#5C2D91")
TEAL = ("#C5F0F5", "#0C8599")
GRAY = ("#E3E3E3", "#495057")
RED = ("#FDE7E9", "#D13438")
RESERVED = ("#F2F2F2", "#9AA0A6")
P1 = ("#FFF4CE", "#F7630C")
P2 = ("#E8DAEF", "#5C2D91")
OPTIONAL = ("#E3E3E3", "#495057")

fig, ax = plt.subplots(figsize=(24, 31))
ax.set_xlim(0, 380)
ax.set_ylim(0, 505)
ax.invert_yaxis()
ax.axis("off")

BOXES: list[tuple[str, float, float, float, float]] = []


def badge(cx: float, cy: float, text: str, color=ORANGE, fs: float = 6.8) -> None:
    """Small priority/status badge."""
    fc, ec = color
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold",
        color=ec,
        linespacing=1.05,
        bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec=ec, lw=1.15),
        zorder=7,
    )


def rbox(
    cx: float,
    cy: float,
    w: float,
    h: float,
    text: str,
    col,
    fs: float = 7.4,
    *,
    lw: float = 1.8,
    ls: str = "-",
    reserved: bool = False,
    badge_text: str | None = None,
    badge_color=ORANGE,
    name: str | None = None,
) -> None:
    """Rounded box with centered multiline text."""
    fc, ec = RESERVED if reserved else col
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.2,rounding_size=0.9",
            fc=fc,
            ec=ec,
            lw=lw,
            ls="--" if reserved else ls,
            zorder=3,
        )
    )
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color="#4A4A4A" if reserved else "#111111",
        linespacing=1.18,
        zorder=4,
    )
    if badge_text:
        badge(cx + w / 2 - 5.6, cy - h / 2 + 2.6, badge_text, badge_color, fs=6.5)
    if name:
        BOXES.append((name, cx, cy, w, h))


def reserved_box(
    cx: float,
    cy: float,
    w: float,
    h: float,
    text: str,
    priority: str,
    fs: float = 6.8,
    *,
    name: str | None = None,
) -> None:
    color = P1 if priority == "P1" else P2 if priority == "P2" else OPTIONAL
    rbox(cx, cy, w, h, text, GRAY, fs, reserved=True, badge_text=priority, badge_color=color, name=name)


def diamond(
    cx: float,
    cy: float,
    w: float,
    h: float,
    text: str,
    col,
    fs: float = 7.0,
    *,
    reserved: bool = False,
    badge_text: str | None = None,
    name: str | None = None,
) -> None:
    """Diamond decision node."""
    fc, ec = RESERVED if reserved else col
    pts = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.8, ls="--" if reserved else "-", zorder=3))
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color="#4A4A4A" if reserved else "#111111",
        linespacing=1.12,
        zorder=4,
    )
    if badge_text:
        badge(cx + w / 2 - 2.5, cy - h / 2 + 1.8, badge_text, P1 if badge_text == "P1" else P2, fs=6.1)
    if name:
        BOXES.append((name, cx, cy, w, h))


def arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str | None = None,
    *,
    fs: float = 6.3,
    ls: str = "-",
    color: str = "#666666",
    lw: float = 1.45,
    rad: float = 0.0,
) -> None:
    """Arrow with an optional white-background label."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=ls,
            shrinkA=1.5,
            shrinkB=2.0,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=2,
    )
    if label:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            label,
            fontsize=fs,
            color="#333333" if ls == "-" else "#777777",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.88),
            zorder=6,
        )


def side_panel(x: float, y: float, w: float, h: float, title: str, lines: list[str]) -> None:
    """Explanatory panel with left-aligned bullets."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.35,rounding_size=0.9",
            fc="#FAFAFA",
            ec=GRAY[1],
            lw=1.5,
            zorder=1,
        )
    )
    ax.text(x + w / 2, y + 5.0, title, ha="center", va="center", fontsize=8.4, fontweight="bold", zorder=2)
    yy = y + 11.0
    for line in lines:
        ax.text(x + 3.0, yy, line, ha="left", va="top", fontsize=6.8, color="#222222", linespacing=1.2, zorder=2)
        yy += 9.0


# Title / legend
ax.text(
    190,
    6.0,
    "Phase 4 · 身份感知 · 多帧事件理解 logic flow（沿计划的决策树 · 实线=已实现 / 虚线灰=预留未实现）",
    ha="center",
    va="center",
    fontsize=13.0,
    fontweight="bold",
)
ax.text(
    190,
    12.0,
    "蓝=输入/选帧  橙=判定/分窗  绿=核心CV  紫=认人/身份  青=输出  红=gpt-4o(贵)  | 实线=已实现  虚线+灰=预留(标 P1/P2)",
    ha="center",
    va="center",
    fontsize=8.4,
    color="#444444",
)

CX = 190
MAIN_W = 126

# ---------------- Spine: input -> local CV -> provider registry ----------------
rbox(CX, 25, MAIN_W, 8.0, "① 输入：视频流 / 视频文件\n本地把视频当“流”逐帧处理", BLUE, 8.1, name="S1")
rbox(
    CX,
    40,
    MAIN_W,
    10.8,
    "② 选帧①定时密采样 extract_frames(fps=2)\n全部帧跑本地CV(廉价)；只精选喂LLM ← 省钱杠杆",
    ORANGE,
    7.2,
    badge_text="选帧①",
    name="S2",
)
rbox(
    CX,
    58,
    MAIN_W,
    12.5,
    "③ 逐帧本地CV：YOLO检测 + ByteTrack跟踪 → 稳定track_id\n灰度指纹(去重)；person crop清晰度(选最佳帧)",
    GREEN,
    7.2,
    name="S3",
)
rbox(
    CX,
    78,
    MAIN_W,
    11.0,
    "④ 事件提供器注册表（可插拔；对每帧提特征）\nprovider: 检测 → 特征 → 结构化信号",
    GREEN,
    7.4,
    name="S4",
)
arrow(CX, 29.2, CX, 34.2)
arrow(CX, 45.8, CX, 51.5)
arrow(CX, 64.5, CX, 72.5)

# ---------------- Provider fan-out lanes ----------------
rbox(CX, 99, 98, 7.8, "LANE A — 人物 provider（person）★主线", PURPLE, 7.6, name="A0")
arrow(CX, 83.8, CX, 94.5, "主线", fs=6.1)

# Peripheral planned providers, kept as reserved slots.
reserved_box(30, 104, 50, 12, "LANE B\n宠物 provider\n宠物脸→宠物档案", "P2", 6.0, name="B")
reserved_box(30, 136, 50, 13, "LANE D\n包裹/物品 + OCR\n投递/被取/移动\n车牌/单号", "P2", 5.7, name="D")
reserved_box(30, 234, 50, 12, "LANE F\n异常事件\n跌倒/火焰/玻璃破碎", "可选", 5.9, name="F")
reserved_box(350, 104, 52, 11, "LANE C\n车辆 provider\n驶入 / 离开", "可选", 6.0, name="C")
reserved_box(350, 136, 52, 11, "LANE E\n区域/越界/徘徊", "可选", 6.0, name="E")
reserved_box(350, 168, 52, 13, "LANE G\n目标级轨迹 +\n手物交互坐标", "P2", 5.8, name="G")
reserved_box(350, 201, 52, 13, "LANE H\n＋预留槽\n新事件类型即插即用", "可选", 5.7, name="H")
# Reserved provider fan-out uses side "buses" to keep the main person tree readable.
left_bus_x, right_bus_x = 62, 318
ax.plot([left_bus_x, left_bus_x], [92, 248], color="#9AA0A6", lw=1.0, ls="--", zorder=1)
ax.plot([right_bus_x, right_bus_x], [92, 248], color="#9AA0A6", lw=1.0, ls="--", zorder=1)
arrow(CX - 32, 84.0, left_bus_x, 92, "预留 providers", fs=5.5, ls="--", color="#9AA0A6", lw=1.0, rad=-0.08)
arrow(CX + 32, 84.0, right_bus_x, 92, "预留 providers", fs=5.5, ls="--", color="#9AA0A6", lw=1.0, rad=0.08)
for y in [104, 136, 234]:
    arrow(left_bus_x, y, 55, y, ls="--", color="#9AA0A6", lw=1.0)
for y in [104, 136, 168, 201]:
    arrow(right_bus_x, y, 324, y, ls="--", color="#9AA0A6", lw=1.0)

# Person provider branches: face / body / gait.
rbox(92, 120, 68, 10, "A1 人脸 face.py\nInsightFace检测+对齐\nArcFace 512d + 质量评估", PURPLE, 6.5, name="A1")
rbox(190, 120, 68, 10, "A2 人形 ReID body\nOSNet 512d →\n主体记忆库 gallery", PURPLE, 6.5, name="A2")
reserved_box(288, 120, 68, 12, "A3 步态 gait\napp/gait.py: YOLO-Pose骨架序列\n→ OpenGait；LiDAR→LidarGait++", "P1", 5.8, name="A3")
arrow(CX - 30, 102.8, 92, 114.5)
arrow(CX, 102.8, 190, 114.5)
arrow(CX + 30, 102.8, 288, 113.5, ls="--", color="#9AA0A6")

# A1: face quality fork + blurry-face arsenal.
diamond(92, 139, 48, 16, "人脸质量？", ORANGE, 6.8, name="A1Q")
arrow(92, 125.3, 92, 130.2)
rbox(55, 157, 48, 8.6, "清晰正脸\n→ 强身份信号", PURPLE, 6.5, name="A1clear")
rbox(111, 157, 48, 8.6, "糊脸/侧脸/小脸\n→ 降权", PURPLE, 6.5, name="A1blur")
arrow(82, 145.5, 55, 152.5, "清晰", fs=5.8)
arrow(102, 145.5, 111, 152.5, "糊/侧/小", fs=5.8)
rbox(83, 176, 70, 8.8, "多帧脸聚合 fuse_embeddings\n质量加权融合（已实现）", PURPLE, 6.1, name="A1fuse")
arrow(111, 161.5, 83, 171.0)
reserved_box(29, 196, 42, 9.2, "AdaFace / MagFace\n低质量脸升级", "P1", 5.6, name="AdaFace")
reserved_box(83, 198, 62, 12, "3D-68 几何 cue\nbuffalo_l 1k3d68 /\nMICA·Deep3DFaceRecon", "P1", 5.2, name="3D")
reserved_box(137, 196, 42, 9.2, "人脸超分\nGFP-GAN / CodeFormer", "P1", 5.4, name="SR")
reserved_box(83, 217, 72, 8.8, "人脸库比对\nface → 人员库（接口对齐/未接）", "可选", 5.7, name="FaceGallery")
arrow(111, 161.5, 29, 191.0, ls="--", color="#9AA0A6", lw=1.05)
arrow(111, 161.5, 83, 191.5, ls="--", color="#9AA0A6", lw=1.05)
arrow(111, 161.5, 137, 191.0, ls="--", color="#9AA0A6", lw=1.05)
arrow(83, 180.8, 83, 212.0, ls="--", color="#9AA0A6", lw=1.05)

# A2: gallery decision + stitching/fusion.
diamond(190, 140, 54, 17, "gallery\n裁决", ORANGE, 6.6, name="A2Q")
arrow(190, 125.3, 190, 131.0)
rbox(168, 160, 43, 8.8, "hit 命中复用\nnew 新建档", PURPLE, 6.2, name="HitNew")
rbox(221, 160, 48, 8.8, "grey 灰区\n待会话内裁决", PURPLE, 6.3, name="Grey")
arrow(178, 147.8, 168, 155.0, "hit/new", fs=5.8)
arrow(202, 147.8, 221, 155.0, "grey", fs=5.8)
rbox(
    166,
    181,
    66,
    11,
    "灰区轨迹缝合 _stitch_orphans\nReID余弦≥0.45 → 并入最相近主体",
    PURPLE,
    5.8,
    badge_text="本会话新增",
    badge_color=GREEN,
    name="Stitch",
)
rbox(224, 202, 56, 10, "多帧融合 track_fusion\n最佳帧 + 投票 + 多线索", PURPLE, 5.9, name="Fusion")
arrow(221, 164.8, 166, 175.5)
arrow(221, 164.8, 224, 196.5)

# Person convergence.
rbox(
    CX,
    235,
    92,
    10.5,
    "A 汇聚：三路多线索融合 → 稳定 subject_id\nface + body 已实线；gait 预留虚线接入",
    PURPLE,
    6.5,
    name="AConv",
)
arrow(55, 161.8, 150, 230.0, "face强信号", fs=5.8)
arrow(83, 221.7, 150, 230.0, ls="--", color="#9AA0A6", lw=1.05)
arrow(166, 186.6, 185, 229.5)
arrow(224, 207.3, 205, 229.5)
arrow(288, 126.5, 230, 230.0, ls="--", color="#9AA0A6", lw=1.15, label="gait cue")

# Provider lanes converge back to spine.
rbox(CX, 257, MAIN_W, 10.5, "⑤ 语义事件标注\n事件=语义信号，非像素/ffmpeg场景变化：new_track / track_left / count_change / identity_hit", ORANGE, 6.8, name="S5")
arrow(CX, 240.5, CX, 251.5)
arrow(left_bus_x, 248, CX - MAIN_W / 2, 257, "预留信号汇入", fs=5.5, ls="--", color="#9AA0A6", lw=1.0)
arrow(right_bus_x, 248, CX + MAIN_W / 2, 257, "预留信号汇入", fs=5.5, ls="--", color="#9AA0A6", lw=1.0)

# ---------------- Spine after provider convergence ----------------
rbox(CX, 276, MAIN_W, 9.0, "⑥ 结构化事件信号\n全部帧·文本紧凑 → 可全量带给LLM", TEAL, 7.0, name="S6")
arrow(CX, 262.5, CX, 271.0)

diamond(
    CX,
    300,
    92,
    18,
    "⑦ 流式分窗\n_split_windows",
    ORANGE,
    7.0,
    badge_text="流式分窗\n窗=1次LLM",
    name="S7",
)
arrow(CX, 280.8, CX, 290.5)
rbox(114, 324, 74, 10, "关窗A：活动结束\n连续 ≥ quiet 2s 无人", ORANGE, 6.5, name="S7a")
rbox(266, 324, 78, 11, "关窗B：时长封顶\n窗帧数 ≥ 30s×fps → 冲刷开新窗\n防长事件欠采样", ORANGE, 5.8, name="S7b")
arrow(CX - 22, 308.5, 114, 318.5, "活动结束", fs=5.8)
arrow(CX + 22, 308.5, 266, 318.5, "时长封顶", fs=5.8)

rbox(
    CX,
    351,
    MAIN_W,
    12.0,
    "⑧ 选帧②事件驱动关键帧 select_keyframes\n事件帧必留 + 每track最佳帧 + 相邻去重(灰度指纹余弦)\n保时序 + ≤KEYFRAME_MAX(默认24，demo常用8)",
    ORANGE,
    6.4,
    badge_text="选帧②",
    name="S8",
)
arrow(114, 329.5, CX - 35, 345.0)
arrow(266, 330.0, CX + 35, 345.0)

rbox(
    CX,
    375,
    MAIN_W,
    12.0,
    "⑨ 身份打包 _group_people + format_identity_context\n按subject合并：同一人多track只列一个人，防误导LLM计数\n约定身份外部给定；糊脸退人形ReID·步态",
    PURPLE,
    6.3,
    name="S9",
)
arrow(CX, 357.3, CX, 369.0)

diamond(290, 398, 56, 17, "dry-run？", GRAY, 7.0, name="DryRun")
rbox(290, 424, 64, 10, "是 → 跳过LLM\n仅出身份 + 关键帧\n验链路不花钱", TEAL, 6.1, name="DryYes")
rbox(
    CX,
    407,
    MAIN_W,
    18.0,
    "⑩ understand_event 多模态 gpt-4o 跨帧事件理解\n少量关键帧(图)+身份(文本)；EVENT_SYSTEM 分离 WHO=身份 / WHAT=动作\n撞429退避重试 → JSON{events, summary, alert_level, notification}",
    RED,
    6.2,
    name="S10",
)
arrow(CX, 381.3, 267, 394.5)
arrow(278, 406, 252, 407, "否", fs=5.8)
arrow(290, 406.8, 290, 418.5, "是", fs=5.8)

# Output layer fan-out.
rbox(CX, 448, MAIN_W, 8.5, "11 输出层：事件窗时间线 / 导出 / 评估闭环", TEAL, 7.2, name="S11")
arrow(CX, 416.3, CX, 443.5)
arrow(290, 429.3, 230, 443.5)
rbox(
    102,
    472,
    116,
    13.0,
    "11 事件窗时间线（已实现）\n告警徽章 + 概述 + 逐条事件 + 关键帧缩略图 + 身份卡\n入口：CLI event_understand_demo.py / Web /eventmonitor / JSON下载",
    TEAL,
    5.9,
    name="Timeline",
)
reserved_box(238, 472, 78, 12.0, "highlight 视频片段拼接\napp/clip_export.py\n按事件裁剪/拼接(ffmpeg)", "P2", 5.8, name="Clip")
reserved_box(
    335,
    472,
    78,
    13.0,
    "bad-case 评估闭环\n模糊人脸 baseline vs\n+增强/+多帧脸/+人形/+gait/+3D",
    "P1",
    5.4,
    name="BadCase",
)
arrow(CX - 38, 452.5, 102, 465.5)
arrow(CX, 452.5, 238, 466.0, ls="--", color="#9AA0A6", lw=1.1)
arrow(CX + 38, 452.5, 335, 465.5, ls="--", color="#9AA0A6", lw=1.1)

# Side legend / priority panel.
side_panel(
    8,
    386,
    82,
    66,
    "状态图例 / 优先级",
    [
        "• 实线=已实现；虚线+灰=预留未实现",
        "• P1：gait / AdaFace·3D·超分 / bad-case评估",
        "• P2：OCR / 宠物身份 / 手物交互 / 视频片段拼接",
        "• 本会话新增：灰区轨迹缝合、流式分窗时长上限、eventmonitor页、JSON导出",
    ],
)

# Small visual anchors to reinforce implemented vs reserved path semantics.
ax.plot([105, 145], [492, 492], color="#666666", lw=1.6)
ax.text(148, 492, "实线路径已实现", va="center", fontsize=6.5, color="#333333")
ax.plot([225, 265], [492, 492], color="#9AA0A6", lw=1.4, ls="--")
ax.text(268, 492, "虚线灰为预留槽（均带 P1/P2/可选 badge）", va="center", fontsize=6.5, color="#555555")


def _warn_overlaps() -> None:
    """Print conservative box-overlap warnings for manual layout tuning."""
    warnings: list[str] = []
    for i, (n1, x1, y1, w1, h1) in enumerate(BOXES):
        for n2, x2, y2, w2, h2 in BOXES[i + 1 :]:
            if abs(x1 - x2) < (w1 + w2) / 2 + 1.0 and abs(y1 - y2) < (h1 + h2) / 2 + 1.0:
                warnings.append(f"{n1} overlaps {n2}")
    if warnings:
        print("[layout-warning] " + "; ".join(warnings[:12]))


def main() -> None:
    _warn_overlaps()
    root = Path(__file__).resolve().parents[1]
    svg_out = root / "docs" / "phase4-logic-flow.svg"
    png_out = root / "docs" / "phase4-logic-flow.png"
    svg_out.parent.mkdir(parents=True, exist_ok=True)
    # SVG is the canonical vector diagram; PNG is a convenience preview.
    plt.savefig(svg_out, format="svg", bbox_inches="tight", facecolor="white")
    plt.savefig(png_out, dpi=110, bbox_inches="tight", facecolor="white")
    print(svg_out)
    print(png_out)


if __name__ == "__main__":
    main()
