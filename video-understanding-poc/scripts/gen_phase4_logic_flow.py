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

# Wide readable canvas: preserve large text while giving the identity tree room.
FIG_W, FIG_H = 30, 47.7
XMAX, YMAX = 2000, 3180
FONT_SCALE = 2.25
BADGE_FS = 12.5
NODE_MIN_FS = 13.5
NODE_TITLE_MIN_FS = 14.8
NODE_DETAIL_MIN_FS = 11.8
LABEL_MIN_FS = 12.5
TITLE_FS = 26.0
LEGEND_FS = 15.5
PANEL_TITLE_FS = 16.0
PANEL_BODY_FS = 13.0
ANCHOR_FS = 13.0
POINT_TO_DATA_Y = YMAX / (FIG_H * 0.97 * 72.0)


def scaled_font_size(fs: float, minimum: float) -> float:
    return max(fs * FONT_SCALE, minimum)


def badge_font_size(fs: float) -> float:
    return max(fs, BADGE_FS)


def node_title_font_size(fs: float) -> float:
    return max(fs * FONT_SCALE * 1.08, NODE_TITLE_MIN_FS)


def node_detail_font_size(fs: float) -> float:
    return max(fs * FONT_SCALE * 0.82, NODE_DETAIL_MIN_FS)


fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.invert_yaxis()
ax.axis("off")

BOXES: list[tuple[str, float, float, float, float]] = []
CONNECTORS: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
TEXT_AREAS: list[tuple[str, list[object], float, float, float, float]] = []


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


def node_text(
    cx: float,
    cy: float,
    w: float,
    h: float,
    text: str,
    fs: float,
    *,
    reserved: bool = False,
    fontproperties: FontProperties | None = None,
) -> list[object]:
    """Draw a node title line plus smaller detail lines."""
    lines = text.splitlines()
    title = lines[0] if lines else ""
    details = "\n".join(lines[1:])
    title_fs = node_title_font_size(fs)
    detail_fs = node_detail_font_size(fs)
    title_color = "#4A4A4A" if reserved else "#111111"
    detail_color = "#666666" if reserved else "#333333"
    if not details:
        return [
            ax.text(
                cx,
                cy,
                title,
                ha="center",
                va="center",
                fontsize=title_fs,
                fontweight="bold",
                color=title_color,
                fontproperties=fontproperties,
                zorder=4,
            )
        ]

    detail_lines = len(details.splitlines())
    title_line_pt = title_fs * 1.08
    gap_pt = max(2.0, detail_fs * 0.18)
    detail_block_pt = detail_lines * detail_fs * 1.15
    total_data_h = (title_line_pt + gap_pt + detail_block_pt) * POINT_TO_DATA_Y
    top_y = cy - total_data_h / 2
    detail_y = top_y + (title_line_pt + gap_pt) * POINT_TO_DATA_Y
    return [
        ax.text(
            cx,
            top_y,
            title,
            ha="center",
            va="top",
            fontsize=title_fs,
            fontweight="bold",
            color=title_color,
            fontproperties=fontproperties,
            zorder=4,
        ),
        ax.text(
            cx,
            detail_y,
            details,
            ha="center",
            va="top",
            fontsize=detail_fs,
            color=detail_color,
            linespacing=1.15,
            fontproperties=fontproperties,
            zorder=4,
        ),
    ]


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
    """Rounded box with title/detail text hierarchy."""
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
    text_objs = node_text(cx, cy, w, h, text, fs, reserved=reserved, fontproperties=fontproperties)
    if name:
        BOXES.append((name, cx, cy, w, h))
        TEXT_AREAS.append((name, text_objs, cx, cy, w, h))
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
    """Diamond decision node with title/detail text hierarchy."""
    fc, ec = RESERVED if reserved else col
    pts = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.8, ls="--" if reserved else "-", zorder=3))
    text_objs = node_text(cx, cy, w, h, text, fs, reserved=reserved)
    if name:
        BOXES.append((name, cx, cy, w, h))
        TEXT_AREAS.append((name, text_objs, cx, cy, w, h))
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


def identity_container(x: float, y: float, w: float, h: float, title: str) -> None:
    """Large non-semantic grouping container for the person-identity provider cluster."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.6,rounding_size=10.0",
            fc=(0.92, 0.84, 0.96, 0.16),
            ec=PURPLE[1],
            lw=2.4,
            ls="-",
            zorder=0.4,
        )
    )
    ax.text(
        x + w / 2,
        y + 28,
        title,
        ha="center",
        va="center",
        fontsize=17.0,
        fontweight="bold",
        color=PURPLE[1],
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=PURPLE[1], lw=1.0, alpha=0.92),
        zorder=2.6,
    )


# Title / legend
ax.text(
    XMAX / 2,
    35,
    "Phase 4 · 身份感知 · 多帧事件理解 logic flow（沿计划的决策树 · 实线=已实现 / 虚线灰=预留未实现）",
    ha="center",
    va="center",
    fontsize=TITLE_FS,
    fontweight="bold",
)
ax.text(
    XMAX / 2,
    68,
    "蓝=输入/选帧  橙=判定/分窗  绿=核心CV  紫=认人/身份  青=输出  红=gpt-4o(贵)  | 实线=已实现  虚线+灰=预留(标 P1/P2)",
    ha="center",
    va="center",
    fontsize=LEGEND_FS,
    color="#444444",
)

CX = 800
MAIN_W = 680
LEFT_BUS_X = 205
RIGHT_BUS_X = 1810

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
reserved_box(1910, 565, 170, 70, "LANE C\n车辆 provider\n驶入 / 离开", "可选", 6.0, name="C")
reserved_box(1910, 680, 170, 70, "LANE E\n区域/越界/徘徊", "可选", 6.0, name="E")
reserved_box(1910, 795, 170, 78, "LANE G\n目标级轨迹 +\n手物交互坐标", "P2", 5.8, name="G")
reserved_box(1910, 910, 170, 78, "LANE H\n＋预留槽\n新事件类型即插即用", "可选", 5.7, name="H")
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
    arrow(RIGHT_BUS_X, y, 1824, y, ls="--", color="#9AA0A6", lw=1.0)

# Person provider branches: face / body / gait.
identity_container(220, 600, 1510, 1020, "人物身份 provider（人脸 + 人形 ReID + 步态 → subject_id）")
rbox(450, 700, 270, 78, "A1 人脸 face.py\nInsightFace检测+对齐\nArcFace 512d + 质量评估", PURPLE, 6.5, name="A1")
rbox(1000, 700, 270, 78, "A2 人形 ReID body\nOSNet 512d →\n主体记忆库 gallery", PURPLE, 6.5, name="A2")
rbox(
    1450,
    700,
    350,
    104,
    "A3 步态 gait.py：SkeletonGait++\n(OpenGait, GREW 权重) 已实现\n第三路身份信号(无脸/背身兜底)",
    PURPLE,
    6.1,
    name="A3",
)
reserved_box(1580, 830, 260, 56, "可选预留\nLiDAR → LidarGait++", "可选", 5.8, name="LidarGait")
arrow(CX - 115, 570, 450, 659)
arrow(CX, 570, 1000, 659)
arrow(CX + 115, 570, 1450, 648, color=PURPLE[1], lw=1.55)
arrow(1450, 752, 1580, 802, ls="--", color="#9AA0A6", lw=1.05)

# A1: face quality fork + blurry-face arsenal.
diamond(450, 850, 190, 84, "人脸质量？", ORANGE, 6.8, name="A1Q")
arrow(450, 739, 450, 806)
rbox(350, 1015, 230, 62, "清晰正脸\n强身份 / 入库", PURPLE, 6.2, name="A1clear")
rbox(680, 1015, 240, 62, "糊脸/侧脸/小脸\n降权 / 只查不建", PURPLE, 6.1, name="A1blur")
arrow(405, 887, 350, 984, "清晰正脸", fs=5.8)
arrow(495, 887, 680, 984, "糊/侧/小", fs=5.8)
rbox(365, 1190, 260, 68, "多帧脸聚合 fuse_embeddings\n质量加权融合（已实现）", PURPLE, 5.4, name="A1fuse")
arrow(350, 1046, 365, 1156, "多帧聚合", fs=5.4, color=PURPLE[1], lw=1.1)
rbox(
    760,
    1120,
    270,
    72,
    "AdaFace 识别后端\nIR-101 WebFace12M，质量自适应\n低清脸更强（替代 ArcFace，可切换）",
    PURPLE,
    5.5,
    name="AdaFace",
)
rbox(
    760,
    1240,
    270,
    82,
    "3D-68 几何 cue\nbuffalo_l 1k3d68 → 15维姿态/尺度不变\n面部几何，糊脸兜底",
    PURPLE,
    5.0,
    name="3D",
)
rbox(
    760,
    1345,
    270,
    62,
    "人脸超分\nGFP-GAN，识别前把糊小脸拉清再提 embedding",
    PURPLE,
    5.4,
    name="SR",
)
rbox(
    620,
    1440,
    660,
    80,
    "人脸库比对\n"
    "face → 人脸 gallery：清晰脸入库+高置信命中(matched)，糊脸只查不建(防污染)\n"
    "ArcFace 阈值独立(FACE_HIT/NEW_THRESH)",
    PURPLE,
    5.6,
    name="FaceGallery",
)
route_arrow([(350, 1046), (225, 1085), (225, 1388), (430, 1400)], "强身份/入库", label_xy=(228, 1290), fs=5.5, color=PURPLE[1], lw=1.1)
route_arrow([(365, 1224), (365, 1385), (430, 1400)], color=PURPLE[1], lw=1.1)
arrow(680, 1046, 760, 1084, "降权", fs=5.8, color=PURPLE[1], lw=1.2)
arrow(760, 1156, 760, 1199, "P1增强", fs=5.2, ls="--", color="#9AA0A6", lw=1.05)
arrow(760, 1281, 760, 1314, ls="--", color="#9AA0A6", lw=1.05)
arrow(760, 1376, 760, 1400, "查库(不入库)", fs=5.4, ls="--", color="#9AA0A6", lw=1.05)

# A2: gallery decision + stitching/cross-route merge/fusion.
diamond(1025, 850, 190, 84, "gallery\n裁决", ORANGE, 6.6, name="A2Q")
arrow(1000, 739, 1025, 806)
rbox(1000, 1015, 210, 58, "hit 命中复用\nnew 新建档", PURPLE, 6.2, name="HitNew")
rbox(1300, 1015, 220, 58, "grey 灰区\n待会话内裁决", PURPLE, 6.3, name="Grey")
arrow(985, 887, 1000, 986, "hit/new", fs=5.8)
arrow(1070, 887, 1300, 986, "grey", fs=5.8)
rbox(
    1110,
    1180,
    300,
    72,
    "灰区轨迹缝合 _stitch_orphans\nReID余弦≥0.45 → 并入最相近主体",
    PURPLE,
    5.8,
    badge_text="本会话新增",
    badge_color=GREEN,
    name="Stitch",
)
rbox(1320, 1320, 260, 66, "多帧融合 track_fusion\n最佳帧 + 投票 + 多线索", PURPLE, 5.9, name="Fusion")
arrow(1300, 1044, 1110, 1144)
arrow(1300, 1044, 1320, 1287)
rbox(
    1320,
    1430,
    560,
    108,
    "跨track三路合并 _merge_tracks_cross_route\n"
    "人脸库/人形库/步态库 任一路认出同一人即合并\n"
    "清晰命中脸/命中步态才当锚点(糊脸不锚)；多路印证→高置信\n"
    "统一 subject_id → 下游 _group_people 自然归并",
    PURPLE,
    5.55,
    badge_text="本会话新增",
    badge_color=GREEN,
    name="CrossRouteMerge",
)

# Person convergence.
rbox(
    CX,
    1555,
    1100,
    72,
    "A 汇聚：三路身份融合 → subject_id\nidentity_fusion.fuse_identity：人脸+人形+步态 质量加权（清晰脸主导/糊脸退人形步态/多路一致加成）→ 统一身份置信度（已实现）",
    PURPLE,
    6.5,
    badge_text="已实现",
    badge_color=GREEN,
    name="AConv",
)
arrow(950, 1440, 1040, 1440, color=PURPLE[1], lw=1.35)
route_arrow(
    [(1000, 1044), (930, 1085), (930, 1370), (1040, 1415)],
    "人形库 subject_id",
    label_xy=(925, 1260),
    fs=5.4,
    color=PURPLE[1],
    lw=1.2,
)
route_arrow([(1110, 1216), (1110, 1338), (1105, 1376)], "灰区缝合", label_xy=(1080, 1298), fs=5.4, color=PURPLE[1], lw=1.2)
arrow(1320, 1353, 1320, 1376, color=PURPLE[1], lw=1.25)
route_arrow(
    [(1450, 752), (1745, 760), (1745, 1430), (1600, 1430)],
    "gait hit",
    label_xy=(1670, 760),
    fs=5.8,
    color=PURPLE[1],
    lw=1.45,
)
arrow(1320, 1484, 1000, 1519, "统一 subject_id", fs=5.5, color=PURPLE[1], lw=1.45)

# Provider lanes converge back to spine through outer gutters, below the person tree.
rbox(CX, 1660, MAIN_W, 60, "⑤ 语义事件标注\n事件=语义信号，非像素/ffmpeg场景变化：new_track / track_left / count_change / identity_hit", ORANGE, 6.8, name="S5")
arrow(CX, 1591, CX, 1630)
route_arrow([(LEFT_BUS_X, 830), (20, 870), (20, 1630), (540, 1630)], "预留信号汇入", label_xy=(135, 1620), fs=5.5, ls="--", color="#9AA0A6", lw=1.0)
route_arrow([(RIGHT_BUS_X, 950), (1980, 990), (1980, 1630), (1060, 1630)], "预留信号汇入", label_xy=(1845, 1620), fs=5.5, ls="--", color="#9AA0A6", lw=1.0)

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
    "⑨ 身份打包 _group_people + format_identity_context\n按统一subject合并：跨route合并后同一人多track只列一个人\n约定身份外部给定；糊脸退人形ReID·步态，防误导LLM计数",
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
    for name, text_objs, cx, cy, w, h in TEXT_AREAS:
        bboxes = [obj.get_window_extent(renderer=renderer) for obj in text_objs]
        text_bbox = bboxes[0]
        for bbox in bboxes[1:]:
            text_bbox.x0 = min(text_bbox.x0, bbox.x0)
            text_bbox.x1 = max(text_bbox.x1, bbox.x1)
            text_bbox.y0 = min(text_bbox.y0, bbox.y0)
            text_bbox.y1 = max(text_bbox.y1, bbox.y1)
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
