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

# Wide readable canvas: preserve large text while keeping Phase 4 focused.
FIG_W, FIG_H = 37.6, 38.0
XMAX, YMAX = 2420, 2540
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


def scene_lane_container(x: float, y: float, w: float, h: float, title: str) -> None:
    """Mixed-status scene provider lane container."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.6,rounding_size=10.0",
            fc=(0.90, 0.96, 0.90, 0.18),
            ec=GREEN[1],
            lw=2.2,
            ls="-",
            zorder=0.35,
        )
    )
    ax.text(
        x + w / 2,
        y + 28,
        title,
        ha="center",
        va="center",
        fontsize=16.2,
        fontweight="bold",
        color=GREEN[1],
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=GREEN[1], lw=1.0, alpha=0.94),
        zorder=2.6,
    )


# Title / legend
ax.text(
    XMAX / 2,
    35,
    "Phase 4 · 身份感知 · 多帧事件理解 logic flow（运行时主链路 → 事件报告）",
    ha="center",
    va="center",
    fontsize=TITLE_FS,
    fontweight="bold",
)
ax.text(
    XMAX / 2,
    68,
    "蓝=输入/选帧  绿=本地CV  紫=人物身份  橙=事件窗/关键帧  红=多模态LLM  青=事件报告  | 实线=已实现  虚线灰=预留未实现(P1/P2)",
    ha="center",
    va="center",
    fontsize=LEGEND_FS,
    color="#444444",
)

GCX = XMAX / 2
CX = 900
MAIN_W = 760

# ---------------- 1. Input and local CV spine ----------------
rbox(GCX, 130, MAIN_W, 62, "① 输入：视频流 / 视频文件\n本地把视频当“流”逐帧处理", BLUE, 8.1, name="S1")
rbox(
    GCX,
    215,
    MAIN_W,
    64,
    "② 选帧①定时密采样 extract_frames(fps=2)\n抽成时序帧；后面只把关键帧喂给LLM",
    ORANGE,
    7.2,
    badge_text="选帧①",
    name="S2",
)
rbox(
    GCX,
    320,
    MAIN_W,
    74,
    "③ 便宜第一遍·逐帧本地CV：YOLO检测 + BoT-SORT+ReID / ByteTrack → 稳定 track_id\n只判占用/轨迹/清晰度指纹（喂给⑥分窗）；重活(人脸/步态)推迟到分窗后",
    GREEN,
    7.2,
    name="S3",
)
rbox(
    GCX,
    435,
    MAIN_W,
    64,
    "④ 语义信号抽取层\n从检测/跟踪/身份结果生成结构化事件信号，不靠 ffmpeg 像素场景切换",
    GREEN,
    7.4,
    name="S4",
)
arrow(GCX, 153, GCX, 181)
arrow(GCX, 247, GCX, 283)
arrow(GCX, 357, GCX, 403)
# ④ → ⑤ 流式分窗（前置：先用便宜的 YOLO 占用切窗，重活 LANE A 只在窗内跑）→ 再分到 LANE A / LANE D
arrow(GCX, 467, GCX, 486)
diamond(GCX, 524, 320, 74, "⑤ 流式分窗\n_split_windows", ORANGE, 6.6, badge_text="窗=1次LLM", name="S5win")
rbox(GCX - 455, 524, 300, 52, "关窗A：活动结束\n连续 quiet 秒无人", ORANGE, 5.5, name="S5a")
rbox(GCX + 430, 524, 260, 56, "关窗B：时长封顶\n防长事件欠采样", ORANGE, 5.5, name="S5b")
arrow(GCX - 160, 524, GCX - 305, 524)
arrow(GCX + 160, 524, GCX + 300, 524)
ax.plot([GCX, GCX], [561, 578], color="#666666", lw=1.45, zorder=2)
CONNECTORS.append(("provider split trunk", (GCX, 561), (GCX, 578)))
route_arrow([(GCX, 578), (CX, 578), (CX, 650)], "person lane · track门控", label_xy=(1000, 560), fs=6.1)
route_arrow([(GCX, 578), (2055, 578), (2055, 662)], "scene lane", label_xy=(1625, 560), fs=6.1, color=GREEN[1], lw=1.25)

# ---------------- 2. Person identity cluster ----------------
identity_container(100, 600, 1600, 980, "LANE A — 人物身份 provider（分窗后·按 track·门控通过才跑，不逐帧；太短/太低质 track 整条跳过）")
rbox(
    430,
    740,
    380,
    96,
    "A1 人脸 face.py（每 track 最佳帧）\nassess_quality 分级：模糊(拉普拉斯+关键点置信度+可插拔深度FIQA)\n+角度(yaw/pitch，低头更严)；InsightFace/AdaFace",
    PURPLE,
    5.4,
    name="A1",
)
rbox(
    900,
    740,
    310,
    78,
    "A2 人形 ReID（每 track 最佳帧）\nOSNet 512d → 主体记忆库 gallery",
    PURPLE,
    6.3,
    name="A2",
)
rbox(
    1370,
    750,
    360,
    88,
    "A3 步态 gait.py\nSkeletonGait++ + GREW 权重；分窗后只在活动窗帧采序列；无脸/背身兜底",
    PURPLE,
    6.2,
    name="A3",
)

arrow(CX - 120, 650, 430, 698)
arrow(CX, 650, 900, 698)
arrow(CX + 120, 650, 1370, 708, color=PURPLE[1], lw=1.55)

# A1: face quality fork + deployed blurry-face arsenal.
diamond(430, 870, 190, 82, "人脸质量?\n角度yaw/pitch·模糊", ORANGE, 6.4, name="A1Q")
arrow(430, 779, 430, 826)
rbox(300, 955, 220, 58, "清晰(clear)\n入人脸库 / 满权重", PURPLE, 6.0, name="A1clear")
rbox(560, 955, 230, 58, "糊/侧/低质(marginal/poor)\n降权；只查不建", PURPLE, 6.0, name="A1blur")
arrow(385, 856, 300, 926, "clear", fs=5.5)
arrow(475, 856, 560, 926, "marginal/poor", fs=5.5)
rbox(
    560,
    1075,
    270,
    74,
    "低质脸增强/降级策略\n糊脸触发GFP-GAN(非极端侧脸)；软性连续降权\nAdaFace/3D-68 可开关",
    PURPLE,
    5.0,
    badge_text="已接入",
    badge_color=GREEN,
    name="FaceEnhance",
)
rbox(
    430,
    1160,
    430,
    72,
    "人脸 gallery\n清晰脸建档/高置信命中；糊脸防污染",
    PURPLE,
    5.9,
    name="FaceGallery",
)
arrow(300, 984, 365, 1127, "入库", fs=5.3, color=PURPLE[1], lw=1.1)
arrow(560, 984, 560, 1038, "增强/降权", fs=5.3, color=PURPLE[1], lw=1.1)
arrow(560, 1112, 495, 1127, "查库", fs=5.3, color=PURPLE[1], lw=1.1)

# A2: body ReID decision and session stitching.
diamond(900, 870, 190, 82, "gallery\n裁决", ORANGE, 6.6, name="A2Q")
arrow(900, 779, 900, 826)
rbox(800, 955, 205, 58, "hit/new\n命中复用 / 新建档", PURPLE, 6.0, name="HitNew")
rbox(1040, 955, 205, 58, "grey 灰区\n待会话内裁决", PURPLE, 6.0, name="Grey")
arrow(865, 856, 800, 926, "hit/new", fs=5.5)
arrow(935, 856, 1040, 926, "grey", fs=5.5)
rbox(
    1040,
    1085,
    300,
    68,
    "灰区/低质轨迹缝合 _stitch_orphans\n时间不重叠才可并；本地subject用高阈值防误并",
    PURPLE,
    5.7,
    name="Stitch",
)
arrow(1040, 984, 1040, 1048)

# A3: gait gallery.
rbox(
    1370,
    955,
    300,
    68,
    "步态 gallery\n姿态+剪影序列 → gait subject",
    PURPLE,
    5.8,
    name="GaitGallery",
)
arrow(1370, 794, 1370, 918, "步态序列", fs=5.3, color=PURPLE[1], lw=1.1)

# Cross-route merge and identity confidence.
rbox(
    CX,
    1305,
    1120,
    118,
    "跨 track 三路合并 _merge_tracks_cross_route\n"
    "输入：每条 track 的 body_sid / clear_face_sid / gait_sid\n"
    "并查集：union(ti,tj) if 任一路 route_id 相同；canonical=已有body subject最小值，否则新建\n"
    "输出：统一 subject_id + merge_routes / merge_agree",
    PURPLE,
    5.2,
    badge_text="已实现",
    badge_color=GREEN,
    name="CrossRouteMerge",
)
rbox(
    CX,
    1435,
    1120,
    88,
    "身份置信汇聚 score_identity_confidence\n"
    "confidence=Σ(score×weight)/Σweight+agree_bonus；人脸权重=软性连续 0.5×(0.3+0.7×质量分)\n"
    "输出 fused{confidence,resolved,primary,sources}，给 LLM 解释身份可靠性",
    PURPLE,
    5.6,
    badge_text="已实现",
    badge_color=GREEN,
    name="AConv",
)
route_arrow([(430, 1196), (430, 1225), (560, 1225), (560, 1243)], "clear face id", label_xy=(505, 1216), fs=5.1, color=PURPLE[1], lw=1.1)
route_arrow([(800, 984), (800, 1135), (760, 1135), (760, 1243)], "body id", label_xy=(745, 1115), fs=5.1, color=PURPLE[1], lw=1.1)
route_arrow([(1040, 1119), (1040, 1178), (980, 1178), (980, 1243)], "stitched/body", label_xy=(1060, 1170), fs=5.1, color=PURPLE[1], lw=1.1)
route_arrow([(1370, 988), (1370, 1225), (1240, 1225), (1240, 1243)], "gait id", label_xy=(1395, 1095), fs=5.1, color=PURPLE[1], lw=1.1)
arrow(CX, 1364, CX, 1389, "统一 subject_id", fs=5.4, color=PURPLE[1], lw=1.25)

# ---------------- Scene-level provider lane ----------------
scene_lane_container(1775, 600, 560, 640, "LANE D — 场景级 provider（OCR + 物体/包裹，已接入）")
rbox(
    2055,
    705,
    500,
    106,
    "物体/包裹检测\n复用 tracker.track_objects 全类别 → 收 OBJECT_CLASSES 非人目标(bag/suitcase/车)\nCOCO 无快递箱类→近似；品牌靠 OCR+LLM 看图",
    GREEN,
    4.95,
    badge_text="已实现",
    badge_color=GREEN,
    name="D1Object",
)
rbox(
    1885,
    910,
    280,
    112,
    "OCR 场景文字\napp/ocr.py\nRapidOCR/PaddleOCR\ntext + bbox",
    GREEN,
    5.0,
    badge_text="已实现",
    badge_color=GREEN,
    name="D2OCR",
)
rbox(
    2225,
    910,
    280,
    112,
    "物体轨迹\nobject_tracks 跨帧\nframe#@ts + 方向\nOBJECT_MIN_FRAMES",
    GREEN,
    4.8,
    badge_text="已实现",
    badge_color=GREEN,
    name="D3Track",
)
rbox(
    2055,
    1168,
    500,
    96,
    "场景级 scene_context + object_context\nOCR scene_context + 物体 object_context（frame#@ts 对齐人物 grounding）\n不进 subject_id / gallery / fusion；并列喂 LLM",
    GREEN,
    4.75,
    badge_text="已实现",
    badge_color=GREEN,
    name="DOut",
)
arrow(2225, 758, 2225, 854, "object boxes", fs=4.8, color=GREEN[1], lw=1.25)
arrow(1885, 966, 1950, 1120, "scene_context", fs=4.8, color=GREEN[1], lw=1.25)
arrow(2225, 966, 2160, 1120, "object_context", fs=4.8, color=GREEN[1], lw=1.25)
route_arrow(
    [(2055, 1216), (2055, 1660), (GCX, 1660)],
    "scene/object signals",
    label_xy=(1905, 1640),
    fs=5.0,
    color=GREEN[1],
    lw=1.25,
)

# ---------------- 3. Event windows, LLM, report ----------------
rbox(
    GCX,
    1740,
    MAIN_W,
    66,
    "⑥ 统一结构化事件信号总线\nperson: new_track/track_left/count_change/identity_hit\nOCR + 物体信号：选帧后加入 LLM",
    ORANGE,
    5.75,
    name="S5",
)
route_arrow([(CX, 1479), (CX, 1660), (GCX, 1660)], "person signals", label_xy=(1045, 1640), fs=5.2, color=PURPLE[1], lw=1.15)
arrow(GCX, 1660, GCX, 1707)   # 各 provider 信号汇聚 → ⑥ 信号总线
# 分窗已前置到 LANE A 之前；信号总线汇聚各 provider 事件后，逐窗做选帧②
arrow(GCX, 1773, GCX, 1863, "逐窗处理")

rbox(
    GCX,
    1900,
    MAIN_W,
    74,
    "⑦ 选帧②事件驱动关键帧 select_keyframes（供 LLM 叙述；身份已在 LANE A 用每track最佳帧认完）\n事件帧必留 + 每 track 最佳帧 + 相邻去重；保时序并限制关键帧数量",
    ORANGE,
    6.4,
    badge_text="选帧②",
    name="S7",
)

rbox(
    GCX,
    2030,
    MAIN_W,
    75,
    "⑧ 多 provider grounding 打包\n人物：identity_context + bbox/center/trajectory\nOCR/场景：scene_context + object_context(物体+轨迹)；并列喂 LLM",
    PURPLE,
    5.55,
    name="S8",
)
arrow(GCX, 1937, GCX, 1993)

rbox(
    GCX,
    2160,
    MAIN_W,
    92,
    "⑨ understand_event 多模态 LLM 跨帧事件理解\n关键帧(图) + 身份上下文(文本) + scene_context(OCR) + object_context(物体/轨迹)\n+ bbox/trajectory grounding → JSON{events, summary, alert_level, notification}",
    RED,
    6.2,
    name="S9",
)
arrow(GCX, 2067, GCX, 2114)
rbox(
    GCX,
    2320,
    900,
    110,
    "输出：事件报告（Web /event-monitor + JSON）\n事件窗时间线：告警等级、概述、逐条事件、关键帧缩略图、身份卡\n跨窗整段总结 summarize_event_windows：把多窗串成连贯 story",
    TEAL,
    6.1,
    badge_text="已实现",
    badge_color=GREEN,
    name="Report",
)
arrow(GCX, 2206, GCX, 2265)

# Small visual anchors to reinforce implemented vs reserved path semantics.
ax.plot([640, 800], [2450, 2450], color="#666666", lw=1.6)
ax.text(815, 2450, "实线=已接入端到端路径", va="center", fontsize=ANCHOR_FS, color="#333333")
ax.plot([1120, 1280], [2450, 2450], color=RESERVED[1], lw=1.4, ls="--")
ax.text(1295, 2450, "虚线灰=预留未实现（P1/P2）", va="center", fontsize=ANCHOR_FS, color="#555555")


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
