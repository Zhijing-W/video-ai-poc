# -*- coding: utf-8 -*-
"""生成 PoC 架构图（中文可读、对齐真实流水线顺序）。

主路径（与 event_analysis_pipeline 一致）：
  样片/上传 → 同步分析接口 → 固定帧率抽帧 → 检测跟踪
  → 切事件时间段 → 人形/人脸/步态认人并融合
  → 选关键帧 → 场景文字/物体（关键帧上）→ 打包 JSON
  → 大模型理解 → 监控台展示

输出：
  docs/poc-architecture.svg / .png
  docs/phase4-logic-flow.svg / .png（兼容旧文件名）
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

BLUE = ("#CFE4FA", "#0078D4")
ORANGE = ("#FFF4CE", "#F7630C")
GREEN = ("#DFF6DD", "#107C10")
PURPLE = ("#E8DAEF", "#5C2D91")
TEAL = ("#C5F0F5", "#0C8599")
GRAY = ("#EDEBE9", "#605E5C")
RED = ("#FDE7E9", "#D13438")
AMBER = ("#FFF4CE", "#C19C00")
MUTED = ("#F3F2F1", "#8A8886")

FIG_W, FIG_H = 28.0, 40.0
XMAX, YMAX = 1800, 2580
FONT_SCALE = 1.5
TITLE_FS = 22.0
SUB_FS = 12.0
LEGEND_FS = 11.2
PANEL_TITLE_FS = 12.5
PANEL_BODY_FS = 10.3
NODE_TITLE_MIN = 12.2
NODE_DETAIL_MIN = 10.0
LABEL_MIN = 10.2
POINT_TO_DATA_Y = YMAX / (FIG_H * 0.97 * 72.0)

BOXES: list[tuple[str, float, float, float, float]] = []
TEXT_AREAS: list[tuple[str, list[object], float, float, float, float]] = []


def scaled(fs: float, minimum: float) -> float:
    return max(fs * FONT_SCALE, minimum)


def node_title_fs(fs: float) -> float:
    return max(fs * FONT_SCALE * 1.05, NODE_TITLE_MIN)


def node_detail_fs(fs: float) -> float:
    return max(fs * FONT_SCALE * 0.82, NODE_DETAIL_MIN)


fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=0.02, right=0.98, top=0.985, bottom=0.015)
ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.invert_yaxis()
ax.axis("off")


def badge(cx: float, cy: float, text: str, color=ORANGE, fs: float = 9.8) -> None:
    fc, ec = color
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=max(fs, 9.8),
        fontweight="bold",
        color=ec,
        linespacing=1.05,
        bbox=dict(boxstyle="round,pad=0.24", fc=fc, ec=ec, lw=1.15),
        zorder=7,
    )


def node_text(cx: float, cy: float, text: str, fs: float, *, muted: bool = False) -> list[object]:
    lines = text.splitlines()
    title = lines[0] if lines else ""
    details = "\n".join(lines[1:])
    title_color = "#605E5C" if muted else "#111111"
    detail_color = "#8A8886" if muted else "#333333"
    tfs = node_title_fs(fs)
    dfs = node_detail_fs(fs)
    if not details:
        return [
            ax.text(cx, cy, title, ha="center", va="center", fontsize=tfs,
                    fontweight="bold", color=title_color, zorder=4)
        ]
    detail_lines = len(details.splitlines())
    title_pt = tfs * 1.08
    gap_pt = max(2.0, dfs * 0.18)
    detail_pt = detail_lines * dfs * 1.15
    total_h = (title_pt + gap_pt + detail_pt) * POINT_TO_DATA_Y
    top_y = cy - total_h / 2
    detail_y = top_y + (title_pt + gap_pt) * POINT_TO_DATA_Y
    return [
        ax.text(cx, top_y, title, ha="center", va="top", fontsize=tfs,
                fontweight="bold", color=title_color, zorder=4),
        ax.text(cx, detail_y, details, ha="center", va="top", fontsize=dfs,
                color=detail_color, linespacing=1.15, zorder=4),
    ]


def rbox(
    cx, cy, w, h, text, col, fs=7.0, *, lw=1.8, ls="-", muted=False,
    badge_text=None, badge_color=GREEN, name=None,
):
    fc, ec = MUTED if muted else col
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.2,rounding_size=5.0",
            fc=fc, ec=ec, lw=lw, ls=ls, zorder=3,
        )
    )
    text_objs = node_text(cx, cy, text, fs, muted=muted)
    if name:
        BOXES.append((name, cx, cy, w, h))
        TEXT_AREAS.append((name, text_objs, cx, cy, w, h))
    if badge_text:
        badge(cx + w / 2 - 48, cy - h / 2 + 14, badge_text, badge_color, fs=9.2)


def container(x, y, w, h, title, color, *, dashed=False, fill_alpha=0.12):
    fc = matplotlib.colors.to_rgba(color[0], fill_alpha)
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.5,rounding_size=8.0",
            fc=fc, ec=color[1], lw=2.0,
            ls="--" if dashed else "-", zorder=0.4,
        )
    )
    ax.text(
        x + w / 2, y + 22, title,
        ha="center", va="center", fontsize=13.5, fontweight="bold", color=color[1],
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color[1], lw=1.0, alpha=0.95),
        zorder=2.5,
    )


def side_panel(x, y, w, h, title, lines, color=GRAY):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.35,rounding_size=4.0",
            fc="#FAFAFA", ec=color[1], lw=1.5, zorder=1,
        )
    )
    ax.text(x + w / 2, y + 24, title, ha="center", va="center",
            fontsize=PANEL_TITLE_FS, fontweight="bold", color=color[1], zorder=2)
    yy = y + 54
    for line in lines:
        ax.text(x + 14, yy, line, ha="left", va="top",
                fontsize=PANEL_BODY_FS, color="#222222", linespacing=1.25, zorder=2)
        yy += 32


def arrow(x1, y1, x2, y2, label=None, *, fs=6.0, ls="-", color="#666666", lw=1.45):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                        shrinkA=1.5, shrinkB=2.0),
        zorder=2,
    )
    if label:
        ax.text(
            (x1 + x2) / 2, (y1 + y2) / 2, label,
            fontsize=scaled(fs, LABEL_MIN),
            color="#333333" if ls == "-" else "#777777",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9),
            zorder=6,
        )


def legend_swatch(x, y, color, label, ls="-"):
    ax.add_patch(Rectangle((x, y - 8), 26, 16, fc=color[0], ec=color[1], lw=1.3, ls=ls, zorder=5))
    ax.text(x + 34, y, label, ha="left", va="center", fontsize=LEGEND_FS, color="#333", zorder=5)


# =============================================================================
# 标题
# =============================================================================
CX = XMAX / 2
MAIN_W = 780

ax.text(CX, 28, "事件监控台 · PoC 架构（已实现能力）",
        ha="center", va="center", fontsize=TITLE_FS, fontweight="bold")
ax.text(
    CX, 60,
    "样片/上传 → 同步分析 → 抽帧检测跟踪 → 切事件段 → 认人融合 → 选关键帧 → 场景文字/物体 → 打包给大模型 → 页面展示",
    ha="center", va="center", fontsize=SUB_FS, color="#444",
)
ax.text(
    CX, 88,
    "绿标=默认开启　琥珀标=已实现但默认关闭（设置里可开）　灰区=规划未做　说明用语面向答辩，不用代码变量名",
    ha="center", va="center", fontsize=SUB_FS - 0.5, color="#666",
)

legend_swatch(160, 120, GREEN, "默认开启")
legend_swatch(360, 120, AMBER, "默认关闭（可开）")
legend_swatch(620, 120, TEAL, "接口 / 页面输出")
legend_swatch(900, 120, RED, "大模型（Azure）")
legend_swatch(1180, 120, MUTED, "规划未做", ls="--")

# =============================================================================
# 1. 入口与设置
# =============================================================================
rbox(280, 195, 340, 88,
     "选择样片 或 上传视频\n支持常见视频格式\n事件监控台首页入口",
     BLUE, 6.0, badge_text="默认开启", badge_color=GREEN, name="IN")
rbox(700, 195, 300, 88,
     "分析关注点（可选）\n用户用一句话说明重点\n会写进大模型提示词",
     BLUE, 5.9, badge_text="可选", badge_color=AMBER, name="FOCUS")
rbox(1100, 195, 300, 88,
     "仅本地分析\n先不调用大模型\n需要时再补做理解",
     BLUE, 5.9, badge_text="默认关闭", badge_color=AMBER, name="DRY")
rbox(1480, 195, 240, 88,
     "同时只跑一路分析\n避免显存/算力打架",
     TEAL, 5.9, badge_text="已实现", name="LOCK")

rbox(CX, 330, 1480, 100,
     "本次分析参数（只对这一次生效，不写进配置文件）\n"
     "开关：人形识别 / 人脸 / 步态 / 场景文字 / 物体\n"
     "可选模型：跟踪器、人脸模型、人脸清晰化、人形特征模型\n"
     "其它：抽帧频率、每段最多关键帧数、单段最长秒数、同人合并阈值",
     BLUE, 5.5, badge_text="已实现", name="SET")

rbox(CX, 460, 1480, 88,
     "同步分析接口（一次请求跑完整段视频）\n"
     "主接口：提交视频并返回完整结果　｜　可先本地分析，再补做大模型理解\n"
     "另有：样片列表、可用模型列表、健康检查",
     TEAL, 5.6, badge_text="默认开启", badge_color=GREEN, name="API")

arrow(280, 239, 420, 280)
arrow(700, 239, 700, 280)
arrow(1100, 239, 980, 280)
arrow(1480, 239, 1320, 280)
arrow(CX, 380, CX, 416)
arrow(CX, 504, CX, 540)

# =============================================================================
# 2. 密采样主链：抽帧 → 检测跟踪 → 事件信号 → 切段
# =============================================================================
rbox(CX, 590, MAIN_W, 82,
     "① 按固定帧率抽帧（默认约每秒 2 帧）\n把整段视频变成有序图片序列，供后续跟踪与认人\n（另有智能抽帧函数，但未接入本主流程）",
     ORANGE, 5.7, badge_text="默认开启", badge_color=GREEN, name="FPS")
rbox(CX, 710, MAIN_W, 86,
     "② 行人检测 + 多目标跟踪\n检测人/物，并跨帧保持同一个跟踪编号\n可选跟踪算法；默认带外观辅助，遮挡交叉更稳",
     GREEN, 5.6, badge_text="默认开启", name="MOT")
rbox(CX, 830, MAIN_W, 82,
     "③ 生成语义事件信号\n例如：新人出现、人离开、人数变化、新物体出现、认出熟人\n这些信号用来切时间段、挑关键帧，不是直接当最终报告",
     GREEN, 5.5, badge_text="默认开启", name="EVT")
rbox(CX, 950, MAIN_W, 92,
     "④ 切成若干「事件时间段」\n规则一：画面连续一段时间没人/没活动 → 结束本段\n规则二：单段太长（默认约 30 秒）→ 强制截断，避免一段塞太多内容\n全程都安静时：整段视频仍作为一段，方便描述空场景",
     ORANGE, 5.4, badge_text="默认开启", badge_color=GREEN, name="WIN")

arrow(CX, 631, CX, 667)
arrow(CX, 753, CX, 789)
arrow(CX, 871, CX, 904)
arrow(CX, 996, CX, 1035)

# =============================================================================
# 3. 认人（密采样/轨迹级，不是关键帧级）
# =============================================================================
container(50, 1050, 1700, 430,
          "⑤ 人物身份识别（在跟踪轨迹上做；人脸/人形各选自己的最佳帧；步态用时间段内序列）",
          PURPLE)

rbox(300, 1160, 400, 118,
     "人形特征识别\n每条轨迹选「身体最清晰」的一帧\n提特征并查询/登记身份库\n可选多种特征模型（默认 OSNet 路线）",
     PURPLE, 5.3, badge_text="默认开启", badge_color=GREEN, name="BODY")
rbox(780, 1160, 400, 118,
     "人脸识别\n另选「脸最合适」的候选帧（不复用身体帧）\n默认 ArcFace，可切 AdaFace\n可选人脸清晰化；侧脸几何线索可开",
     PURPLE, 5.3, badge_text="默认关闭", badge_color=AMBER, name="FACE")
rbox(1260, 1160, 400, 118,
     "步态识别\n不是单帧，而是时间段内的走路序列\n用姿态 + 剪影提步态特征\n无脸/背身时作补充身份信号",
     PURPLE, 5.3, badge_text="默认关闭", badge_color=AMBER, name="GAIT")

rbox(400, 1360, 520, 100,
     "本次分析身份库（仅本次有效）\n同一次分析里记住见过的人\n多人脸/多角度可记多条特征\n分析结束即清空，不跨视频长期存",
     PURPLE, 5.4, badge_text="默认开启", badge_color=GREEN, name="FAISS")
rbox(1100, 1360, 560, 100,
     "身份汇总到「同一个人」\n同视频轨迹断裂可尝试缝合\n人脸/人形/步态任一路对上可合并\n输出统一身份编号 + 可信度说明",
     PURPLE, 5.3, badge_text="默认开启", badge_color=GREEN, name="FUSE")

arrow(300, 1219, 400, 1310, color=PURPLE[1], lw=1.2)
arrow(780, 1219, 700, 1310, color=PURPLE[1], lw=1.2)
arrow(1260, 1219, 1260, 1310, color=PURPLE[1], lw=1.2)
arrow(400, 1410, 700, 1495, color=PURPLE[1], lw=1.15)
arrow(1100, 1410, 980, 1495, color=PURPLE[1], lw=1.15)
arrow(CX, 1035, CX, 1080)

# =============================================================================
# 4. 选关键帧 → 场景旁路 → 打包 → 大模型
# =============================================================================
rbox(CX, 1560, MAIN_W + 40, 88,
     "⑥ 为每个事件时间段挑选关键帧（准备给大模型看的图）\n优先保留有事件的帧、每人轨迹代表帧，并去掉过于相似的相邻帧\n目的：少给大模型几张图，而不是丢掉前面的跟踪/认人结果",
     ORANGE, 5.4, badge_text="默认开启", badge_color=GREEN, name="KF")

container(50, 1675, 1700, 220,
          "⑦ 场景信息（在关键帧上补齐；不参与「是不是同一个人」的融合）", GREEN, fill_alpha=0.10)

rbox(300, 1785, 400, 100,
     "场景文字识别\n只对上面选出的关键帧读字\n如时间戳、车牌、运单号等\n结果作为场景说明，不代表某个人",
     GREEN, 5.3, badge_text="默认关闭", badge_color=AMBER, name="OCR")
rbox(780, 1785, 400, 100,
     "物体/包裹说明\n汇总本段出现的非人物体轨迹\n如行李、车辆等位置变化\n与人物时间线对齐，供大模型引用",
     GREEN, 5.3, badge_text="默认关闭", badge_color=AMBER, name="OBJ")
rbox(1260, 1785, 400, 100,
     "人物位置与走动方向\n关键帧上的框、中心点、走向\n页面可叠加显示\n并写成文字位置说明",
     GREEN, 5.3, badge_text="默认开启", badge_color=GREEN, name="SPATIAL")

rbox(CX, 1985, MAIN_W + 80, 92,
     "⑧ 打成结构化结果包（每一事件时间段一份）\n内容包括：时间范围、关键帧、已融合的人物身份、位置说明、\n场景文字、物体说明 → 再交给大模型做事件叙述",
     TEAL, 5.4, badge_text="默认开启", badge_color=GREEN, name="PACK")

rbox(CX - 300, 2135, 560, 100,
     "⑨ 大模型逐段理解\n看关键帧图片 + 读身份/场景文字\n输出本段发生了什么、告警等级等",
     RED, 5.5, badge_text="默认开启", badge_color=GREEN, name="LLM")
rbox(CX + 320, 2135, 520, 100,
     "⑩ 整段故事串联\n把多段结果收成完整叙述\n总览、时间线要点、通知文案",
     RED, 5.5, badge_text="默认开启", badge_color=GREEN, name="OVERALL")

arrow(CX, 1495, CX, 1516)
arrow(CX, 1604, CX, 1675)
arrow(300, 1835, 520, 1939, color=GREEN[1], lw=1.1)
arrow(780, 1835, 780, 1939, color=GREEN[1], lw=1.1)
arrow(1260, 1835, 1040, 1939, color=GREEN[1], lw=1.1)
arrow(CX, 2031, CX - 300, 2085)
arrow(CX, 2031, CX + 320, 2085)

# =============================================================================
# 5. 页面输出
# =============================================================================
container(50, 2260, 1700, 150, "11. 事件监控台展示", TEAL, fill_alpha=0.10)
rbox(280, 2350, 320, 88,
     "原始结果下载\n完整 JSON\n含配置与各阶段耗时",
     TEAL, 5.5, badge_text="默认开启", badge_color=GREEN, name="JSON")
rbox(680, 2350, 320, 88,
     "人物身份卡片\n头像、各路命中情况\n融合可信度",
     TEAL, 5.5, badge_text="默认开启", badge_color=GREEN, name="CARD")
rbox(1080, 2350, 320, 88,
     "事件时间线\n关键帧预览\n可看位置框叠加",
     TEAL, 5.5, badge_text="默认开启", badge_color=GREEN, name="TL")
rbox(1460, 2350, 240, 88,
     "进度与耗时\n配置摘要\n补做大模型按钮",
     TEAL, 5.5, badge_text="默认开启", badge_color=GREEN, name="TOOLS")

arrow(CX - 300, 2185, 400, 2275)
arrow(CX + 320, 2185, 1200, 2275)

# =============================================================================
# 底部说明
# =============================================================================
side_panel(60, 2445, 540, 100,
           "部署相关（脚本已写）",
           [
               "• 单机 GPU 虚拟机一键脚本与镜像流水线",
               "• 面向演示/小并发，不是大规模集群方案",
           ], AMBER)
side_panel(640, 2445, 540, 100,
           "规划未做（图上不画进主路径）",
           [
               "• 实时摄像头流、消息推送、跨视频长期人库",
               "• 集群自动扩缩与生产级高可用",
           ], MUTED)
side_panel(1220, 2445, 520, 100,
           "读图时记住",
           [
               "• 跟踪/认人靠较密的帧；大模型只看少量关键帧",
               "• 场景文字在选关键帧之后，不进「是不是同一人」",
           ], GRAY)

ax.text(CX, 2565,
        "依据代码：事件分析主流程 · 监控台接口/页面 · 人形/人脸/步态/场景文字 · 关键帧与时间段切分 · 大模型报告",
        ha="center", va="center", fontsize=10.2, color="#777")


def _warn_overlaps() -> None:
    warnings = []
    for i, (n1, x1, y1, w1, h1) in enumerate(BOXES):
        for n2, x2, y2, w2, h2 in BOXES[i + 1:]:
            if abs(x1 - x2) < (w1 + w2) / 2 + 4 and abs(y1 - y2) < (h1 + h2) / 2 + 4:
                warnings.append(f"{n1} overlaps {n2}")
    if warnings:
        print("[layout-warning] " + "; ".join(warnings[:16]))


def _warn_text_fit() -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    warnings = []
    pad = 8.0
    for name, text_objs, cx, cy, w, h in TEXT_AREAS:
        bboxes = [obj.get_window_extent(renderer=renderer) for obj in text_objs]
        tb = bboxes[0]
        for b in bboxes[1:]:
            tb.x0, tb.x1 = min(tb.x0, b.x0), max(tb.x1, b.x1)
            tb.y0, tb.y1 = min(tb.y0, b.y0), max(tb.y1, b.y1)
        p0 = ax.transData.transform((cx - w / 2 + pad, cy - h / 2 + pad))
        p1 = ax.transData.transform((cx + w / 2 - pad, cy + h / 2 - pad))
        xmin, xmax = sorted((p0[0], p1[0]))
        ymin, ymax = sorted((p0[1], p1[1]))
        if tb.x0 < xmin or tb.x1 > xmax or tb.y0 < ymin or tb.y1 > ymax:
            warnings.append(name)
    if warnings:
        print("[text-warning] text may exceed: " + ", ".join(warnings[:16]))


def main() -> None:
    _warn_text_fit()
    _warn_overlaps()
    root = Path(__file__).resolve().parents[1]
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    poc_svg = docs / "poc-architecture.svg"
    poc_png = docs / "poc-architecture.png"
    legacy_svg = docs / "phase4-logic-flow.svg"
    legacy_png = docs / "phase4-logic-flow.png"
    plt.savefig(poc_svg, format="svg", facecolor="white")
    plt.savefig(poc_png, dpi=120, facecolor="white")
    plt.savefig(legacy_svg, format="svg", facecolor="white")
    plt.savefig(legacy_png, dpi=120, facecolor="white")
    print(poc_svg)
    print(poc_png)
    print(legacy_svg)
    print(legacy_png)


if __name__ == "__main__":
    main()
