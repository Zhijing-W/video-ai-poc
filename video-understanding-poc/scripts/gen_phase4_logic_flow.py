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
from matplotlib.font_manager import FontProperties
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
CIRCLED_CJK_FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
CIRCLED_CJK_FONT = (
    FontProperties(fname=str(CIRCLED_CJK_FONT_PATH))
    if CIRCLED_CJK_FONT_PATH.exists()
    else FontProperties(family="Noto Sans SC")
)

# Compact readable canvas: bigger text on a tighter diagram for fit-to-width viewing.
FIG_W, FIG_H = 24, 47.7
XMAX, YMAX = 1600, 3180
FONT_SCALE = 2.25
BADGE_FS = 12.5
NODE_MIN_FS = 13.5
LABEL_MIN_FS = 12.5
TITLE_FS = 26.0
LEGEND_FS = 15.5
PANEL_TITLE_FS = 16.0
PANEL_BODY_FS = 13.0
ANCHOR_FS = 13.0


def scaled_font_size(fs: float, minimum: float) -> float:
    return max(fs * FONT_SCALE, minimum)


def badge_font_size(fs: float) -> float:
    return max(fs, BADGE_FS)


fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.invert_yaxis()
ax.axis("off")

BOXES: list[tuple[str, float, float, float, float]] = []
CONNECTORS: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
TEXT_AREAS: list[tuple[str, object, float, float, float, float]] = []


def badge(cx: float, cy: float, text: str, color=ORANGE, fs: float = BADGE_FS) -> None:
    """Small priority/status badge."""
    fc, ec = color
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=badge_font_size(fs),
        fontweight="bold",
        color=ec,
        linespacing=1.05,
        bbox=dict(boxstyle="round,pad=0.28", fc=fc, ec=ec, lw=1.35),
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
    fontproperties: FontProperties | None = None,
) -> None:
    """Rounded box with centered multiline text."""
    fc, ec = RESERVED if reserved else col
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.2,rounding_size=5.0",
            fc=fc,
            ec=ec,
            lw=lw,
            ls="--" if reserved else ls,
            zorder=3,
        )
    )
    text_obj = ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=scaled_font_size(fs, NODE_MIN_FS),
        color="#4A4A4A" if reserved else "#111111",
        linespacing=1.18,
        fontproperties=fontproperties,
        zorder=4,
    )
    if name:
        BOXES.append((name, cx, cy, w, h))
        TEXT_AREAS.append((name, text_obj, cx, cy, w, h))
    if badge_text:
        badge(cx + w / 2 - 36, cy - h / 2 - 8, badge_text, badge_color, fs=BADGE_FS)


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
    text_obj = ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=scaled_font_size(fs, NODE_MIN_FS),
        color="#4A4A4A" if reserved else "#111111",
        linespacing=1.12,
        zorder=4,
    )
    if name:
        BOXES.append((name, cx, cy, w, h))
        TEXT_AREAS.append((name, text_obj, cx, cy, w, h))
    if badge_text:
        badge(cx + w / 2 - 30, cy - h / 2 - 8, badge_text, P1 if badge_text == "P1" else P2, fs=BADGE_FS)


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
    CONNECTORS.append((label or "", (x1, y1), (x2, y2)))
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
            fontsize=scaled_font_size(fs, LABEL_MIN_FS),
            color="#333333" if ls == "-" else "#777777",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.88),
            zorder=6,
        )


def route_arrow(
    points: list[tuple[float, float]],
    label: str | None = None,
    *,
    label_xy: tuple[float, float] | None = None,
    fs: float = 6.3,
    ls: str = "-",
    color: str = "#666666",
    lw: float = 1.45,
) -> None:
    """Polyline connector; only the final segment has an arrowhead."""
    if len(points) < 2:
        raise ValueError("route_arrow needs at least two points")
    for p1, p2 in zip(points[:-1], points[1:]):
        CONNECTORS.append((label or "", p1, p2))
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=2)
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls, shrinkA=1.5, shrinkB=2.0),
        zorder=2,
    )
    if label:
        lx, ly = label_xy if label_xy is not None else ((points[0][0] + points[-1][0]) / 2, (points[0][1] + points[-1][1]) / 2)
        ax.text(
            lx,
            ly,
            label,
            fontsize=scaled_font_size(fs, LABEL_MIN_FS),
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
            boxstyle="round,pad=0.35,rounding_size=4.0",
            fc="#FAFAFA",
            ec=GRAY[1],
            lw=1.5,
            zorder=1,
        )
    )
    ax.text(x + w / 2, y + 30, title, ha="center", va="center", fontsize=PANEL_TITLE_FS, fontweight="bold", zorder=2)
    yy = y + 68
    for line in lines:
        ax.text(x + 18, yy, line, ha="left", va="top", fontsize=PANEL_BODY_FS, color="#222222", linespacing=1.2, zorder=2)
        yy += 48
    BOXES.append((f"panel:{title}", x + w / 2, y + h / 2, w, h))


# Title / legend
ax.text(
    800,
    35,
    "Phase 4 · 身份感知 · 多帧事件理解 logic flow（沿计划的决策树 · 实线=已实现 / 虚线灰=预留未实现）",
    ha="center",
    va="center",
    fontsize=TITLE_FS,
    fontweight="bold",
)
ax.text(
    800,
    68,
    "蓝=输入/选帧  橙=判定/分窗  绿=核心CV  紫=认人/身份  青=输出  红=gpt-4o(贵)  | 实线=已实现  虚线+灰=预留(标 P1/P2)",
    ha="center",
    va="center",
    fontsize=LEGEND_FS,
    color="#444444",
)

CX = 800
MAIN_W = 680
LEFT_BUS_X = 220
RIGHT_BUS_X = 1380

# ---------------- Spine: input -> local CV -> provider registry ----------------
rbox(CX, 130, MAIN_W, 62, "① 输入：视频流 / 视频文件\n本地把视频当“流”逐帧处理", BLUE, 8.1, name="S1")
rbox(
    CX,
    215,
    MAIN_W,
    64,
    "② 选帧①定时密采样 extract_frames(fps=2)\n全部帧跑本地CV(廉价)；只精选喂LLM ← 省钱杠杆",
    ORANGE,
    7.2,
    badge_text="选帧①",
    name="S2",
)
rbox(
    CX,
    320,
    MAIN_W,
    74,
    "③ 逐帧本地CV：YOLO检测 + ByteTrack跟踪 → 稳定track_id\n灰度指纹(去重)；person crop清晰度(选最佳帧)",
    GREEN,
    7.2,
    name="S3",
)
rbox(
    CX,
    435,
    MAIN_W,
    64,
    "④ 事件提供器注册表（可插拔；对每帧提特征）\nprovider: 检测 → 特征 → 结构化信号",
    GREEN,
    7.4,
    name="S4",
)
arrow(CX, 153, CX, 181)
arrow(CX, 247, CX, 283)
arrow(CX, 357, CX, 403)

# ---------------- Provider fan-out lanes ----------------
rbox(CX, 545, 400, 50, "LANE A — 人物 provider（person）★主线", PURPLE, 7.6, name="A0")
arrow(CX, 467, CX, 517, "主线", fs=6.1)

# Peripheral planned providers, kept as reserved slots.
reserved_box(90, 565, 170, 70, "LANE B\n宠物 provider\n宠物脸→宠物档案", "P2", 6.0, name="B")
reserved_box(90, 680, 170, 96, "LANE D\n包裹/物品 + OCR\n投递/被取/移动\n车牌/单号", "P2", 5.7, name="D")
reserved_box(90, 795, 170, 70, "LANE F\n异常事件\n跌倒/火焰/玻璃破碎", "可选", 5.9, name="F")
reserved_box(1510, 565, 170, 70, "LANE C\n车辆 provider\n驶入 / 离开", "可选", 6.0, name="C")
reserved_box(1510, 680, 170, 70, "LANE E\n区域/越界/徘徊", "可选", 6.0, name="E")
reserved_box(1510, 795, 170, 78, "LANE G\n目标级轨迹 +\n手物交互坐标", "P2", 5.8, name="G")
reserved_box(1510, 910, 170, 78, "LANE H\n＋预留槽\n新事件类型即插即用", "可选", 5.7, name="H")
# Reserved provider fan-out uses wide side buses to keep the main person tree readable.
ax.plot([LEFT_BUS_X, LEFT_BUS_X], [490, 830], color="#9AA0A6", lw=1.0, ls="--", zorder=1)
ax.plot([RIGHT_BUS_X, RIGHT_BUS_X], [490, 950], color="#9AA0A6", lw=1.0, ls="--", zorder=1)
CONNECTORS.extend(
    [
        ("left provider bus", (LEFT_BUS_X, 490), (LEFT_BUS_X, 830)),
        ("right provider bus", (RIGHT_BUS_X, 490), (RIGHT_BUS_X, 950)),
    ]
)
arrow(CX - MAIN_W / 2, 435, LEFT_BUS_X, 490, "预留 providers", fs=5.5, ls="--", color="#9AA0A6", lw=1.0)
arrow(CX + MAIN_W / 2, 435, RIGHT_BUS_X, 490, "预留 providers", fs=5.5, ls="--", color="#9AA0A6", lw=1.0)
for y in [565, 680, 795]:
    arrow(LEFT_BUS_X, y, 176, y, ls="--", color="#9AA0A6", lw=1.0)
for y in [565, 680, 795, 910]:
    arrow(RIGHT_BUS_X, y, 1424, y, ls="--", color="#9AA0A6", lw=1.0)

# Person provider branches: face / body / gait.
rbox(450, 665, 270, 78, "A1 人脸 face.py\nInsightFace检测+对齐\nArcFace 512d + 质量评估", PURPLE, 6.5, name="A1")
rbox(800, 665, 270, 78, "A2 人形 ReID body\nOSNet 512d →\n主体记忆库 gallery", PURPLE, 6.5, name="A2")
rbox(
    1165,
    665,
    350,
    104,
    "A3 步态 gait.py：SkeletonGait++\n(OpenGait, GREW 权重) 已实现\n第三路身份信号(无脸/背身兜底)",
    PURPLE,
    6.1,
    name="A3",
)
reserved_box(1230, 805, 260, 56, "可选预留\nLiDAR → LidarGait++", "可选", 5.8, name="LidarGait")
arrow(CX - 115, 570, 450, 624)
arrow(CX, 570, 800, 624)
arrow(CX + 115, 570, 1165, 613, color=PURPLE[1], lw=1.55)
arrow(1165, 717, 1230, 777, ls="--", color="#9AA0A6", lw=1.05)

# A1: face quality fork + blurry-face arsenal.
diamond(450, 805, 190, 84, "人脸质量？", ORANGE, 6.8, name="A1Q")
arrow(450, 704, 450, 761)
rbox(330, 955, 210, 58, "清晰正脸\n→ 强身份信号", PURPLE, 6.5, name="A1clear")
rbox(590, 955, 230, 58, "糊脸/侧脸/小脸\n→ 降权", PURPLE, 6.5, name="A1blur")
arrow(405, 842, 330, 925, "清晰", fs=5.8)
arrow(495, 842, 590, 925, "糊/侧/小", fs=5.8)
rbox(450, 1100, 330, 58, "多帧脸聚合 fuse_embeddings\n质量加权融合（已实现）", PURPLE, 6.1, name="A1fuse")
arrow(590, 984, 450, 1071)
reserved_box(270, 1260, 200, 60, "AdaFace / MagFace\n低质量脸升级", "P1", 5.6, name="AdaFace")
reserved_box(510, 1260, 240, 74, "3D-68 几何 cue\nbuffalo_l 1k3d68 /\nMICA·Deep3DFaceRecon", "P1", 5.2, name="3D")
reserved_box(740, 1260, 200, 60, "人脸超分\nGFP-GAN / CodeFormer", "P1", 5.4, name="SR")
reserved_box(500, 1400, 360, 58, "人脸库比对\nface → 人员库（接口对齐/未接）", "可选", 5.7, name="FaceGallery")
route_arrow([(590, 984), (1270, 1030), (1270, 1190), (270, 1190), (270, 1230)], ls="--", color="#9AA0A6", lw=1.05)
route_arrow([(590, 984), (1270, 1030), (1270, 1190), (510, 1190), (510, 1223)], ls="--", color="#9AA0A6", lw=1.05)
route_arrow([(590, 984), (1270, 1030), (1270, 1190), (740, 1190), (740, 1230)], ls="--", color="#9AA0A6", lw=1.05)
route_arrow([(450, 1129), (450, 1175), (900, 1175), (900, 1371), (500, 1371)], ls="--", color="#9AA0A6", lw=1.05)

# A2: gallery decision + stitching/fusion.
diamond(850, 805, 190, 84, "gallery\n裁决", ORANGE, 6.6, name="A2Q")
arrow(800, 704, 850, 761)
rbox(830, 955, 210, 58, "hit 命中复用\nnew 新建档", PURPLE, 6.2, name="HitNew")
rbox(1110, 955, 220, 58, "grey 灰区\n待会话内裁决", PURPLE, 6.3, name="Grey")
arrow(810, 842, 830, 925, "hit/new", fs=5.8)
arrow(895, 842, 1110, 925, "grey", fs=5.8)
rbox(
    850,
    1120,
    300,
    72,
    "灰区轨迹缝合 _stitch_orphans\nReID余弦≥0.45 → 并入最相近主体",
    PURPLE,
    5.8,
    badge_text="本会话新增",
    badge_color=GREEN,
    name="Stitch",
)
rbox(1100, 1260, 260, 66, "多帧融合 track_fusion\n最佳帧 + 投票 + 多线索", PURPLE, 5.9, name="Fusion")
arrow(1110, 984, 850, 1084)
arrow(1110, 984, 1100, 1227)

# Person convergence.
rbox(
    CX,
    1535,
    MAIN_W,
    72,
    "A 汇聚：三路多线索融合 → 稳定 subject_id\nface + body + gait 已实线；LiDAR gait 可选预留",
    PURPLE,
    6.5,
    name="AConv",
)
route_arrow([(330, 984), (330, 1005), (1315, 1005), (1315, 1485), (1060, 1505)], "face强信号", label_xy=(1120, 1005), fs=5.8)
arrow(500, 1429, 610, 1499, ls="--", color="#9AA0A6", lw=1.05)
route_arrow([(850, 1156), (930, 1165), (930, 1485), (760, 1499)])
arrow(1100, 1293, 930, 1499)
route_arrow([(1165, 717), (1365, 760), (1365, 1488), (1035, 1502)], "gait cue", label_xy=(1275, 755), fs=5.8, color=PURPLE[1], lw=1.45)

# Provider lanes converge back to spine through outer gutters, below the person tree.
rbox(CX, 1660, MAIN_W, 60, "⑤ 语义事件标注\n事件=语义信号，非像素/ffmpeg场景变化：new_track / track_left / count_change / identity_hit", ORANGE, 6.8, name="S5")
arrow(CX, 1571, CX, 1630)
route_arrow([(LEFT_BUS_X, 830), (20, 870), (20, 1630), (540, 1630)], "预留信号汇入", label_xy=(135, 1620), fs=5.5, ls="--", color="#9AA0A6", lw=1.0)
route_arrow([(RIGHT_BUS_X, 950), (1580, 990), (1580, 1630), (1060, 1630)], "预留信号汇入", label_xy=(1465, 1620), fs=5.5, ls="--", color="#9AA0A6", lw=1.0)

# ---------------- Spine after provider convergence ----------------
rbox(CX, 1775, MAIN_W, 64, "⑥ 结构化事件信号\n全部帧·文本紧凑 → 可全量带给LLM", TEAL, 7.0, name="S6")
arrow(CX, 1690, CX, 1750)

diamond(
    CX,
    1900,
    360,
    90,
    "⑦ 流式分窗\n_split_windows",
    ORANGE,
    7.0,
    badge_text="流式分窗\n窗=1次LLM",
    name="S7",
)
arrow(CX, 1800, CX, 1855)
rbox(500, 2045, 310, 58, "关窗A：活动结束\n连续 ≥ quiet 2s 无人", ORANGE, 6.5, name="S7a")
rbox(1100, 2045, 330, 68, "关窗B：时长封顶\n窗帧数 ≥ 30s×fps → 冲刷开新窗\n防长事件欠采样", ORANGE, 5.8, name="S7b")
arrow(CX - 95, 1938, 500, 2016, "活动结束", fs=5.8)
arrow(CX + 95, 1938, 1100, 2011, "时长封顶", fs=5.8)

rbox(
    CX,
    2175,
    MAIN_W,
    75,
    "⑧ 选帧②事件驱动关键帧 select_keyframes\n事件帧必留 + 每track最佳帧 + 相邻去重(灰度指纹余弦)\n保时序 + ≤KEYFRAME_MAX(默认24，demo常用8)",
    ORANGE,
    6.4,
    badge_text="选帧②",
    name="S8",
)
arrow(500, 2074, CX - 125, 2138)
arrow(1100, 2079, CX + 125, 2138)

rbox(
    CX,
    2325,
    MAIN_W,
    75,
    "⑨ 身份打包 _group_people + format_identity_context\n按subject合并：同一人多track只列一个人，防误导LLM计数\n约定身份外部给定；糊脸退人形ReID·步态",
    PURPLE,
    6.3,
    name="S9",
)
arrow(CX, 2213, CX, 2288)

diamond(1320, 2465, 230, 90, "dry-run？", GRAY, 7.0, name="DryRun")
rbox(1320, 2600, 260, 70, "是 → 跳过LLM\n仅出身份 + 关键帧\n验链路不花钱", TEAL, 6.1, name="DryYes")
rbox(
    CX,
    2470,
    MAIN_W,
    105,
    "⑩ understand_event 多模态 gpt-4o 跨帧事件理解\n少量关键帧(图)+身份(文本)；EVENT_SYSTEM 分离 WHO=身份 / WHAT=动作\n撞429退避重试 → JSON{events, summary, alert_level, notification}",
    RED,
    6.2,
    name="S10",
)
arrow(CX, 2363, 1320, 2420)
arrow(1206, 2465, 1060, 2470, "否", fs=5.8)
arrow(1320, 2510, 1320, 2565, "是", fs=5.8)

# Output layer fan-out.
rbox(CX, 2705, MAIN_W, 50, "11 输出层：事件窗时间线 / 导出 / 评估闭环", TEAL, 7.2, name="S11")
arrow(CX, 2523, CX, 2680)
arrow(1320, 2635, 930, 2680)
rbox(
    350,
    2845,
    500,
    80,
    "11 事件窗时间线（已实现）\n告警徽章 + 概述 + 逐条事件 + 关键帧缩略图 + 身份卡\n入口：CLI event_understand_demo.py / Web /eventmonitor / JSON下载",
    TEAL,
    5.9,
    name="Timeline",
)
rbox(
    350,
    3000,
    580,
    118,
    "⑫ 跨窗整段事件总结 summarize_event_windows（已实现）\n"
    "所有事件窗理解完后，纯文本整合多窗叙述 + 身份名册\n"
    "→ 整段视频【连贯事件故事】；ReID 身份跨窗关联同一人\n"
    "便宜；dry-run 跳过；输出 overall_summary / story[时间·身份·动作]\n"
    "/ overall_alert_level / notification",
    TEAL,
    5.35,
    name="OverallSummary",
    fontproperties=CIRCLED_CJK_FONT,
)
reserved_box(900, 2845, 310, 70, "highlight 视频片段拼接\napp/clip_export.py\n按事件裁剪/拼接(ffmpeg)", "P2", 5.8, name="Clip")
reserved_box(
    1320,
    2845,
    310,
    80,
    "bad-case 评估闭环\n模糊人脸 baseline vs\n+增强/+多帧脸/+人形/+gait/+3D",
    "P1",
    5.4,
    name="BadCase",
)
arrow(CX - 160, 2730, 350, 2805)
arrow(350, 2885, 350, 2938, "多窗事件结果", fs=5.8)
arrow(CX, 2730, 900, 2810, ls="--", color="#9AA0A6", lw=1.1)
arrow(CX + 160, 2730, 1320, 2805, ls="--", color="#9AA0A6", lw=1.1)

# Side legend / priority panel.
side_panel(
    20,
    2370,
    430,
    260,
    "状态图例 / 优先级",
    [
        "• 实线=已实现；虚线+灰=预留未实现",
        "• A3 步态：SkeletonGait++ GREW 权重已实现",
        "• P1：AdaFace·3D·超分 / bad-case评估",
        "• P2：OCR / 宠物身份 / 手物交互 / 视频片段拼接",
        "• 可选：LiDAR gait / LidarGait++ 预留",
    ],
)

# Small visual anchors to reinforce implemented vs reserved path semantics.
ax.plot([450, 610], [3145, 3145], color="#666666", lw=1.6)
ax.text(625, 3145, "实线路径已实现", va="center", fontsize=ANCHOR_FS, color="#333333")
ax.plot([900, 1060], [3145, 3145], color="#9AA0A6", lw=1.4, ls="--")
ax.text(1075, 3145, "虚线灰为预留槽（均带 P1/P2/可选 badge）", va="center", fontsize=ANCHOR_FS, color="#555555")


def _warn_overlaps() -> None:
    """Print conservative box-overlap warnings for manual layout tuning."""
    warnings: list[str] = []
    for i, (n1, x1, y1, w1, h1) in enumerate(BOXES):
        for n2, x2, y2, w2, h2 in BOXES[i + 1 :]:
            if abs(x1 - x2) < (w1 + w2) / 2 + 4.0 and abs(y1 - y2) < (h1 + h2) / 2 + 4.0:
                warnings.append(f"{n1} overlaps {n2}")
    if warnings:
        print("[layout-warning] " + "; ".join(warnings[:16]))


def _warn_text_fit() -> None:
    """Warn if enlarged node text exceeds its box bounds."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    warnings: list[str] = []
    pad = 8.0
    for name, text_obj, cx, cy, w, h in TEXT_AREAS:
        text_bbox = text_obj.get_window_extent(renderer=renderer)
        p0 = ax.transData.transform((cx - w / 2 + pad, cy - h / 2 + pad))
        p1 = ax.transData.transform((cx + w / 2 - pad, cy + h / 2 - pad))
        xmin, xmax = sorted((p0[0], p1[0]))
        ymin, ymax = sorted((p0[1], p1[1]))
        if text_bbox.x0 < xmin or text_bbox.x1 > xmax or text_bbox.y0 < ymin or text_bbox.y1 > ymax:
            warnings.append(name)
    if warnings:
        print("[text-warning] text may exceed: " + ", ".join(warnings[:16]))


def _point_in_rect(px: float, py: float, cx: float, cy: float, w: float, h: float, margin: float = 4.0) -> bool:
    return (cx - w / 2 - margin) <= px <= (cx + w / 2 + margin) and (cy - h / 2 - margin) <= py <= (cy + h / 2 + margin)


def _segment_intersects_rect(
    p1: tuple[float, float], p2: tuple[float, float], cx: float, cy: float, w: float, h: float, margin: float = 2.0
) -> bool:
    """Liang-Barsky segment/axis-aligned-rectangle intersection test."""
    x0, y0 = p1
    x1, y1 = p2
    xmin, xmax = cx - w / 2 - margin, cx + w / 2 + margin
    ymin, ymax = cy - h / 2 - margin, cy + h / 2 + margin
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
            continue
        ratio = qi / pi
        if pi < 0:
            if ratio > u2:
                return False
            if ratio > u1:
                u1 = ratio
        else:
            if ratio < u1:
                return False
            if ratio < u2:
                u2 = ratio
    return u1 <= u2


def _warn_connector_crossings() -> None:
    """Warn if a connector segment passes through a box other than its endpoint box."""
    warnings: list[str] = []
    for label, p1, p2 in CONNECTORS:
        for name, cx, cy, w, h in BOXES:
            if _point_in_rect(*p1, cx, cy, w, h) or _point_in_rect(*p2, cx, cy, w, h):
                continue
            if _segment_intersects_rect(p1, p2, cx, cy, w, h):
                warnings.append(f"connector {label or p1} crosses {name}")
                break
    if warnings:
        print("[route-warning] " + "; ".join(warnings[:16]))


def main() -> None:
    _warn_text_fit()
    _warn_overlaps()
    _warn_connector_crossings()
    root = Path(__file__).resolve().parents[1]
    svg_out = root / "docs" / "phase4-logic-flow.svg"
    png_out = root / "docs" / "phase4-logic-flow.png"
    svg_out.parent.mkdir(parents=True, exist_ok=True)
    # SVG is the canonical vector diagram; PNG is a convenience preview.
    # Keep the explicit compact canvas so preview dimensions stay deterministic.
    plt.savefig(svg_out, format="svg", facecolor="white")
    plt.savefig(png_out, dpi=110, facecolor="white")
    print(svg_out)
    print(png_out)


if __name__ == "__main__":
    main()
