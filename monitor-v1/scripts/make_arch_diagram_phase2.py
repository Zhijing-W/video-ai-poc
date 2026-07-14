"""生成视频理解 PoC 【Phase 2】架构图（矢量 SVG + 预览 PNG）。

范围约定：本图**只画 Phase 2 本阶段实际新增/改动的架构**，不重复画 Phase 1 的上传管线
（Phase 1 见 assets/architecture-phase1.*）。本阶段实际落地：

- 实时监控主链（cost-aware hybrid · 每帧）：实时帧 → YOLO 廉价初筛 → 事件门控 → 命中才调 gpt-4o
  → 结构化 JSON + 基于 YOLO 框的目标档案 subjects[]。
- LLM监工 + YOLO自动巡航 级联：目标编译（/compile-target）→ YOLO 自动巡航（/cruise-frame，不调 LLM）
  → LLM 定期审计（/analyze-frame 带 plan，每 N 帧）→ 不一致则回填重判。
- 上传批量管线（本阶段新增 Step 7/8）：智能抽帧（scene 突变 OR 定时兜底）→ 两段式 LLM
  （即时事件时间线 + 最终总结）→ 结构化报告 + events[]。

徽标：绿=本阶段已实现 · 橙=本阶段新增 · 灰=Phase 1 沿用。
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

FIG_W, FIG_H = 17.0, 12.5
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(-1.9, 10.6)
ax.axis("off")

# ---------- 配色 ----------
C_SRC = "#475569"     # 视频源（Phase 1 沿用）
C_YOLO = "#ca8a04"    # YOLO 检测
C_GATE = "#dc2626"    # 事件门控（核心）
C_AI = "#16a34a"      # gpt-4o
C_JSON = "#0d9488"    # 结构化 JSON
C_CASC = "#ea580c"    # 级联（本阶段新增）
INK = "#0f172a"
SUB = "#475569"

# 状态徽标色
B_DONE = "#16a34a"    # 绿=本阶段已实现
B_NEW = "#ea580c"     # 橙=本阶段新增（级联）
B_KEEP = "#64748b"    # 灰=Phase 1 沿用


def box(x, y, w, h, lines, color, fc="#ffffff", badge=None, badge_color=None, lw=2.0):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw, edgecolor=color, facecolor=fc, zorder=3,
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
        ax.text(x + w / 2, y + 0.17, badge, ha="center", va="center",
                fontsize=8, fontweight="bold", color="#ffffff", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc=bc, ec="none"))
    return (x, y, w, h)


def arrow(p1, p2, label="", color=INK, rad=0.0, off=(0, 0), ls="-", fs=9.0):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=18,
        linewidth=2.2, color=color, zorder=5,
        connectionstyle=f"arc3,rad={rad}", linestyle=ls,
    )
    ax.add_patch(a)
    if label:
        mx = (p1[0] + p2[0]) / 2 + off[0]
        my = (p1[1] + p2[1]) / 2 + off[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs,
                color=color, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=color, lw=0.8))


# ---------- 标题 ----------
ax.text(FIG_W / 2, 10.25, "视频理解 PoC · Phase 2 架构（cost-aware hybrid · 统一入口 monitor · 实时流 + 级联 + 整段分析）",
        ha="center", va="center", fontsize=16.0, fontweight="bold", color=INK)
ax.text(FIG_W / 2, 9.80,
        "实时流：前端智能抽帧 → YOLO → 门控 → 命中才调 gpt-4o｜目标可被 YOLO 独立判断时切自动巡航 + 定期审计回填",
        ha="center", va="center", fontsize=11.5, color=SUB)
ax.text(FIG_W / 2, 9.46,
        "本图只画 Phase 2 本阶段新增/改动（上传页已并入 monitor）；Phase 1 LLM-first 见 architecture-phase1",
        ha="center", va="center", fontsize=9.2, color="#94a3b8", style="italic")

# ============================================================
#  Band A — 实时监控链（mode① · 每帧）
# ============================================================
bandA = FancyBboxPatch(
    (0.25, 6.05), 16.5, 3.10,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    linewidth=1.6, edgecolor="#cbd5e1", facecolor="#f8fafc",
    linestyle=(0, (6, 4)), zorder=1,
)
ax.add_patch(bandA)
ax.text(0.45, 8.92, "A · 实时监控链（mode① · 每帧 · cost-aware hybrid）",
        ha="left", va="center", fontsize=10.5, fontweight="bold", color="#334155")

ay, ah = 6.55, 1.85
box(0.45, ay, 2.4, ah, [
    ("直播流", 12.0, True, INK),
    ("摄像头 / 视频当直播", 8.2, False, SUB),
    ("浏览器抽帧 640px", 8.0, False, SUB),
], C_SRC, badge="Phase 1 沿用", badge_color=B_KEEP)

box(3.15, ay, 2.5, ah, [
    ("① 前端智能抽帧", 10.8, True, INK),
    ("ticker.js · JS 帧间差异", 7.8, False, C_CASC),
    ("画面没变就跳过", 8.4, True, C_CASC),
    ("不抓不发后端·更省", 7.8, False, SUB),
], C_CASC, fc="#fff7ed", badge="本阶段新增", badge_color=B_NEW)

box(5.95, ay, 2.5, ah, [
    ("② YOLO 物体检测", 10.6, True, INK),
    ("detector.py · yolov8m", 8.0, False, C_YOLO),
    ("box/label/confidence", 7.8, False, SUB),
    ("CPU ~350ms·廉价守门", 7.8, False, SUB),
], C_YOLO, fc="#fefce8", badge="本阶段实现", badge_color=B_DONE)

box(8.75, ay, 2.55, ah, [
    ("③ 事件门控", 11.8, True, INK),
    ("gate.py · 纯规则", 8.2, False, C_GATE),
    ("关键类别/冷却/心跳", 7.8, False, SUB),
    ("命中才放行→调 LLM", 8.0, True, C_GATE),
], C_GATE, fc="#fef2f2", badge="省钱关键", badge_color=B_DONE, lw=2.6)

box(11.6, ay, 2.55, ah, [
    ("④ Azure OpenAI gpt-4o", 9.6, True, INK),
    ("仅命中帧调用", 8.2, True, C_AI),
    ("YOLO 框 grounding", 7.8, False, SUB),
    ("Vision · 无状态", 7.8, False, SUB),
], C_AI, fc="#f0fdf4", badge="已有·改按需", badge_color=B_DONE)

box(14.45, ay, 2.2, ah, [
    ("⑤ 结构化 JSON", 10.5, True, INK),
    ("+ 档案 subjects[]", 8.0, False, C_JSON),
    ("label/box/appearance", 7.4, False, SUB),
    ("接 DB/人脸/人形", 7.8, False, SUB),
], C_JSON, fc="#f0fdfa", badge="本阶段实现", badge_color=B_DONE)

# Band A 箭头
arrow((2.85, 7.48), (3.15, 7.48), "实时帧", C_SRC, off=(0, 0.30), fs=8.0)
arrow((5.65, 7.48), (5.95, 7.48), "变化帧", C_CASC, off=(0, 0.30), fs=8.0)
arrow((8.45, 7.48), (8.75, 7.48), "框/标签", C_YOLO, off=(0, 0.30), fs=8.0)
arrow((11.30, 7.48), (11.6, 7.48), "命中事件", C_GATE, off=(0, 0.30), fs=8.0)
arrow((14.15, 7.48), (14.45, 7.48), "理解+比对", C_AI, off=(0, 0.30), fs=8.0)
# 前端智能抽帧"画面没变跳过"分支（向下丢弃·更省）
arrow((4.40, 6.55), (4.40, 6.10), "画面没变 → 跳过（连后端都不调·更省）", "#94a3b8",
      ls=(0, (4, 3)), off=(0, 0.22), fs=7.8)
# 门控未命中分支（虚线，向下丢弃，省钱）
arrow((10.0, 6.55), (10.0, 6.10), "未命中 → 跳过（不调 LLM）", "#94a3b8",
      ls=(0, (4, 3)), off=(0, 0.22), fs=7.8)

# ============================================================
#  Band B — LLM监工 + YOLO自动巡航 级联
# ============================================================
bandB = FancyBboxPatch(
    (0.25, 1.95), 16.5, 3.55,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    linewidth=1.8, edgecolor=C_CASC, facecolor="#fff7ed",
    linestyle=(0, (6, 4)), zorder=1,
)
ax.add_patch(bandB)
ax.text(0.45, 5.27, "B · LLM监工 + YOLO自动巡航 级联（目标可被 YOLO 独立判断时 → 长时间省钱）",
        ha="left", va="center", fontsize=10.5, fontweight="bold", color="#c2410c")

by, bh = 3.30, 1.70
box(0.55, by, 3.55, bh, [
    ("① 目标编译（LLM 监工）", 10.4, True, INK),
    ("/compile-target · 一次性", 8.4, False, C_CASC),
    ("自然语言报警条件 →", 8.2, False, SUB),
    ("{yolo_class, 颜色, 可否YOLO}", 8.0, False, SUB),
], C_CASC, fc="#ffffff", badge="本阶段新增", badge_color=B_NEW)

box(4.70, by, 3.5, bh, [
    ("② YOLO 自动巡航", 11.4, True, INK),
    ("/cruise-frame · 廉价", 8.4, False, C_CASC),
    ("YOLO+颜色校验 attributes.py", 8.0, False, SUB),
    ("不调 LLM · 缓冲巡航帧", 8.2, False, SUB),
], C_CASC, fc="#ffffff", badge="本阶段新增", badge_color=B_NEW)

box(8.80, by, 3.6, bh, [
    ("③ LLM 定期审计", 11.4, True, INK),
    ("/analyze-frame（带 plan）", 8.2, False, C_CASC),
    ("每 N 帧（默认 8）调 gpt-4o", 8.0, False, SUB),
    ("对比 YOLO巡航 vs gpt-4o", 8.2, True, C_CASC),
], C_CASC, fc="#ffffff", badge="本阶段新增", badge_color=B_NEW)

box(13.0, by, 3.3, bh, [
    ("④ 回填纠正", 11.8, True, INK),
    ("不一致 → 补调缓冲帧", 8.4, False, C_CASC),
    ("改结论 + 搬运统计", 8.2, False, SUB),
    ("一致 → 继续巡航", 8.4, True, C_AI),
], C_CASC, fc="#ffffff", badge="本阶段新增", badge_color=B_NEW)

# Band B 箭头
arrow((4.10, by + bh / 2), (4.70, by + bh / 2), "可独立巡航\n(can_yolo_handle)", C_CASC, off=(0, 0.46), fs=8.2)
arrow((8.20, by + bh / 2), (8.80, by + bh / 2), "每 N 帧审计", C_CASC, off=(0, 0.30), fs=8.4)
arrow((12.40, by + bh / 2), (13.0, by + bh / 2), "发现偏差", C_CASC, off=(0, 0.30), fs=8.4)
# 一致 → 回巡航（循环，虚线绕下方）
arrow((13.6, by), (6.0, by), "审计一致 → 继续巡航（loop）", C_AI,
      rad=-0.22, ls=(0, (5, 3)), off=(0, -0.62), fs=8.6)

# 级联复用上方 YOLO / gpt-4o（跨 band 连接，虚线）
arrow((7.2, ay), (7.2, by + bh), "巡航复用 YOLO", C_YOLO,
      ls=(0, (3, 3)), off=(1.35, 0.0), fs=8.0)
arrow((10.60, by + bh), (12.85, ay), "审计复用 gpt-4o", C_AI,
      rad=0.18, ls=(0, (3, 3)), off=(0.55, 0.05), fs=8.0)

# ============================================================
#  Band C — 整段视频分析（mode② · 流式：视频当流跑实时链 → 末尾总结 → 归档）
# ============================================================
bandC = FancyBboxPatch(
    (0.25, -0.85), 16.5, 2.55,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    linewidth=1.6, edgecolor="#cbd5e1", facecolor="#f8fafc",
    linestyle=(0, (6, 4)), zorder=1,
)
ax.add_patch(bandC)
ax.text(0.45, 1.48, "C · 整段视频分析（mode② · 流式：视频文件当直播流喂入 A 链 → 播完末尾总结 → 自动归档）",
        ha="left", va="center", fontsize=10.5, fontweight="bold", color="#334155")

cy, ch = -0.35, 1.55
box(0.45, cy, 2.95, ch, [
    ("视频文件", 11.5, True, INK),
    ("monitor「整段视频分析」", 8.0, False, C_CASC),
    ("当直播流喂入", 8.2, False, SUB),
], C_SRC, badge="本阶段新增", badge_color=B_NEW)

box(3.90, cy, 3.6, ch, [
    ("① 逐帧分析（复用 A 链）", 9.6, True, INK),
    ("智能抽帧→YOLO→门控→gpt-4o", 7.6, False, C_YOLO),
    ("结果一帧一条实时进日志", 8.0, False, SUB),
    ("流程图照常点亮", 8.0, False, SUB),
], C_GATE, fc="#fef2f2", badge="复用实时链", badge_color=B_DONE)

box(8.00, cy, 3.75, ch, [
    ("② 末尾总结", 11.5, True, INK),
    ("/summarize · summarize_events", 7.8, False, C_AI),
    ("把累积逐帧事件归纳成一段总结", 7.8, False, SUB),
    ("纯文本 · 廉价", 8.0, True, C_AI),
], C_AI, fc="#f0fdf4", badge="本阶段新增", badge_color=B_NEW)

box(12.25, cy, 4.0, ch, [
    ("③ 自动归档", 11.5, True, INK),
    ("/monitor-sessions（含 summary）", 7.8, False, C_JSON),
    ("逐帧日志 + 总结一起存", 8.0, False, SUB),
    ("历史记录可回看", 8.0, False, SUB),
], C_JSON, fc="#f0fdfa", badge="本阶段新增", badge_color=B_NEW)

arrow((3.40, cy + ch / 2), (3.90, cy + ch / 2), "逐帧", C_SRC, off=(0, 0.28))
arrow((7.50, cy + ch / 2), (8.00, cy + ch / 2), "播完", C_GATE, off=(0, 0.28))
arrow((11.75, cy + ch / 2), (12.25, cy + ch / 2), "总结", C_AI, off=(0, 0.28))

# ---------- 脚注 ----------
ax.text(0.30, -1.20,
        "省钱逻辑：实时链（A）门控把「每帧」压到「每事件」；级联（B）把可被 YOLO 判断的目标交给廉价巡航，只每 N 帧用 gpt-4o 审计兜底。",
        ha="left", va="center", fontsize=8.8, color="#94a3b8", style="italic")
ax.text(0.30, -1.47,
        "整段分析（C）：视频文件复用同一条实时链逐帧分析、结果实时进日志，播完只做一次廉价的纯文本末尾总结并归档；不再有上传/弹窗/后端两段式。",
        ha="left", va="center", fontsize=8.8, color="#94a3b8", style="italic")
ax.text(16.70, -1.47, "徽标：绿=已实现/复用 · 橙=本阶段新增 · 灰=Phase 1 沿用",
        ha="right", va="center", fontsize=8.5, color="#94a3b8")

out_dir = Path(__file__).resolve().parents[2] / "assets"
out_dir.mkdir(parents=True, exist_ok=True)
svg_path = out_dir / "architecture-phase2.svg"
png_path = out_dir / "architecture-phase2.png"
fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
fig.savefig(png_path, dpi=170, bbox_inches="tight", facecolor="white")
print(f"saved: {svg_path}")
print(f"saved: {png_path}")
