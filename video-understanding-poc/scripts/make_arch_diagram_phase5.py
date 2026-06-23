"""生成视频理解 PoC 【Phase 5】架构图（矢量 SVG + 预览 PNG）。

范围约定：本图画的是 **「基于 Azure 的全链路上云」** 蓝图——把前几阶段已经做好的
识别算法（YOLO 初筛 + 事件门控 + gpt-4o 精筛 + ReID 主体记忆）搬上 Azure，补齐
「端上推流 → 云端接入 → AML 实时推理 → 事件回传 / 拉流分发 → 难样本闭环训练」整条链路。

三大分带（boustrophedon 全链路自上而下）：
- A 端侧采集与推流（push / ingest）：摄像头 / Azure IoT Edge → RTMP/SRT/WebRTC 推流 → 云端接入网关解码抽帧。
- B Azure 云端实时推理（AML Online Endpoint）：容器化识别服务 = YOLO 初筛 → 门控 → gpt-4o 精筛 + ReID 向量库，模型走 AML Registry 可切换。
- C 出口分发 · 拉流 · 训练闭环：事件 WebSocket 回传前端架构图 / 告警；标注流转封装 HLS/WebRTC 拉流；难样本回流 AML Pipeline 重训回注。

徽标：蓝=Azure 托管服务 · 绿=已有算法沿用(Phase1-3) · 橙=本阶段新建(工程) · 灰=可选/端上/规划。
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

FIG_W, FIG_H = 17.4, 11.8
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ---------- 配色 ----------
C_EDGE = "#7c3aed"    # 端侧 / IoT Edge
C_PUSH = "#ea580c"    # 推流
C_GATE_IN = "#0891b2" # 接入网关
C_AML = "#2563eb"     # AML 在线端点（核心）
C_ALGO = "#16a34a"    # 识别算法（沿用）
C_REID = "#0d9488"    # ReID 主体记忆
C_REG = "#2563eb"     # 模型注册
C_OUT = "#ea580c"     # 出口工程
C_LOOP = "#9333ea"    # 训练闭环
C_SUP = "#2563eb"     # 平台支撑
INK = "#0f172a"
SUB = "#475569"

# 状态徽标色
B_AZURE = "#2563eb"   # 蓝 = Azure 托管服务
B_DONE = "#16a34a"    # 绿 = 已有算法沿用(Phase1-3)
B_NEW = "#ea580c"     # 橙 = 本阶段新建(工程)
B_OPT = "#64748b"     # 灰 = 可选 / 端上 / 规划


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


def arrow(p1, p2, label="", color=INK, rad=0.0, off=(0, 0), ls="-", fs=9.0, lw=2.2):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=18,
        linewidth=lw, color=color, zorder=5,
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
ax.text(FIG_W / 2, 11.45, "视频理解 PoC · Phase 5 架构（基于 Azure 的全链路：端上推流 → AML 实时推理 → 拉流分发 / 闭环训练）",
        ha="center", va="center", fontsize=16.5, fontweight="bold", color=INK)
ax.text(FIG_W / 2, 11.02,
        "摄像头 / IoT Edge 推流(RTMP·SRT·WebRTC) → Azure 接入网关解码 → AML 在线端点(YOLO+门控+gpt-4o+ReID) → 事件 WebSocket 回传 / HLS 拉流 → 难样本闭环重训",
        ha="center", va="center", fontsize=10.2, color=SUB)
ax.text(FIG_W / 2, 10.66,
        "徽标：蓝=Azure 托管服务 · 绿=已有算法沿用(Phase1-3) · 橙=本阶段新建(工程) · 灰=可选/端上/规划",
        ha="center", va="center", fontsize=9.0, color="#94a3b8", style="italic")

# ============================================================
#  Band A — 端侧采集与推流（push / ingest）
# ============================================================
bandA = FancyBboxPatch(
    (0.25, 8.25), 16.9, 2.22,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    linewidth=1.6, edgecolor="#cbd5e1", facecolor="#faf5ff",
    linestyle=(0, (6, 4)), zorder=1,
)
ax.add_patch(bandA)
ax.text(0.45, 10.30, "A · 端侧采集与推流（push / ingest）",
        ha="left", va="center", fontsize=10.5, fontweight="bold", color="#6d28d9")

ay, ah = 8.45, 1.60
box(0.45, ay, 4.7, ah, [
    ("摄像头 / IoT 设备", 11.6, True, INK),
    ("Azure IoT Edge（可选端上）", 8.8, False, C_EDGE),
    ("端上 YOLO 初筛 → 省带宽/省 token", 8.2, False, SUB),
    ("H.264 编码", 8.2, False, SUB),
], C_EDGE, fc="#ffffff", badge="可选 · 端上", badge_color=B_OPT)

box(5.55, ay, 4.6, ah, [
    ("推流 RTMP / SRT / WebRTC", 11.0, True, INK),
    ("低延迟上行 · 鉴权 + TLS", 8.6, False, C_PUSH),
    ("摄像头/浏览器把流「推」上云", 8.2, False, SUB),
    ("（替代上传文件）", 8.2, False, SUB),
], C_PUSH, fc="#fff7ed", badge="本阶段新建", badge_color=B_NEW)

box(10.55, ay, 6.55, ah, [
    ("接入网关 · 解码抽帧", 11.6, True, INK),
    ("MediaMTX / LiveKit 容器 @ AKS / Container Apps", 8.6, False, C_GATE_IN),
    ("收流 → 解码 → 抽帧 640px（采样 1~2s）", 8.2, False, SUB),
    ("※ Azure Media Services 已退役，故自建接入", 8.0, True, "#b45309"),
], C_GATE_IN, fc="#ecfeff", badge="Azure 托管 · 自建", badge_color=B_AZURE)

arrow((5.15, ay + ah / 2), (5.55, ay + ah / 2), "推流", C_PUSH, off=(0, 0.28), fs=8.6)
arrow((10.15, ay + ah / 2), (10.55, ay + ah / 2), "接收", C_GATE_IN, off=(0, 0.28), fs=8.6)

# A → B 全链路下行
arrow((8.55, 8.30), (8.55, 7.92), "解码后帧流进 AML（鉴权 · 内网）", C_AML,
      off=(0, 0.0), fs=8.8)

# ============================================================
#  Band B — Azure 云端实时推理（AML Online Endpoint）
# ============================================================
bandB = FancyBboxPatch(
    (0.25, 5.45), 16.9, 2.45,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    linewidth=1.9, edgecolor=C_AML, facecolor="#eff6ff",
    linestyle=(0, (6, 4)), zorder=1,
)
ax.add_patch(bandB)
ax.text(0.45, 7.68, "B · Azure 云端实时推理（Azure Machine Learning · Managed Online Endpoint）",
        ha="left", va="center", fontsize=10.5, fontweight="bold", color="#1d4ed8")

by, bh = 5.72, 1.72
box(0.45, by, 3.4, bh, [
    ("AML 在线端点", 11.8, True, INK),
    ("Managed Online Endpoint", 8.6, False, C_AML),
    ("容器化识别服务 (FastAPI)", 8.2, False, SUB),
    ("自动扩缩 · 鉴权 · 监控", 8.2, True, C_AML),
], C_AML, fc="#ffffff", badge="Azure 托管", badge_color=B_AZURE, lw=2.6)

box(4.07, by, 3.95, bh, [
    ("识别管线（粗筛→精筛）", 11.0, True, INK),
    ("YOLO 初筛 → 事件门控 → gpt-4o", 8.4, False, C_ALGO),
    ("命中才调 LLM · 省 token", 8.2, True, C_ALGO),
    ("Phase 1/2 逻辑直接沿用", 8.2, False, SUB),
], C_ALGO, fc="#f0fdf4", badge="已有沿用", badge_color=B_DONE)

box(8.24, by, 3.4, bh, [
    ("主体记忆 · ReID", 11.6, True, INK),
    ("ReID 向量库 (Azure AI Search)", 8.4, False, C_REID),
    ("跨帧/跨摄认出同一人/物", 8.2, False, SUB),
    ("Phase 3 规划机制", 8.2, False, SUB),
], C_REID, fc="#f0fdfa", badge="Phase3 规划", badge_color=B_OPT)

box(11.86, by, 2.6, bh, [
    ("结构化结果", 11.8, True, INK),
    ("JSON + 事件/告警", 8.4, False, C_ALGO),
    ("命中 → 推给出口", 8.2, False, SUB),
    ("可接业务 DB", 8.2, False, SUB),
], C_ALGO, fc="#f0fdf4", badge="已有沿用", badge_color=B_DONE)

box(14.66, by, 2.44, bh, [
    ("AML 模型注册", 11.0, True, INK),
    ("Model Registry", 8.6, False, C_REG),
    ("YOLO/LLM 版本", 8.2, False, SUB),
    ("一键切换 · 灰度", 8.2, True, C_REG),
], C_REG, fc="#eff6ff", badge="Azure 托管", badge_color=B_AZURE)

arrow((3.85, by + bh / 2), (4.07, by + bh / 2), "", C_ALGO)
arrow((8.02, by + bh / 2), (8.24, by + bh / 2), "", C_REID)
arrow((11.64, by + bh / 2), (11.86, by + bh / 2), "", C_ALGO)
# 模型注册 → 端点（供模型，虚线上行回流到识别端点）
arrow((15.40, by + bh), (3.20, by + bh), "Registry 供模型 / 热更新", C_REG,
      rad=0.16, ls=(0, (4, 3)), off=(0, 0.34), fs=8.0)

# B → C 全链路下行
arrow((8.55, 5.45), (8.55, 5.06), "事件 + 标注帧 → 出口", C_OUT, off=(0, 0.0), fs=8.8)

# ============================================================
#  Band C — 出口分发 · 拉流 · 训练闭环
# ============================================================
bandC = FancyBboxPatch(
    (0.25, 2.55), 16.9, 2.40,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    linewidth=1.6, edgecolor="#cbd5e1", facecolor="#f8fafc",
    linestyle=(0, (6, 4)), zorder=1,
)
ax.add_patch(bandC)
ax.text(0.45, 4.73, "C · 出口分发 · 拉流（pull / playback）· 训练闭环",
        ha="left", va="center", fontsize=10.5, fontweight="bold", color="#334155")

cy, ch = 2.82, 1.62
box(0.45, cy, 3.85, ch, [
    ("事件回传 → 前端", 11.4, True, INK),
    ("WebSocket / SignalR", 8.6, False, C_OUT),
    ("实时点亮架构图（本 demo）", 8.2, False, SUB),
    ("告警：Logic Apps / 邮件", 8.2, False, SUB),
], C_OUT, fc="#fff7ed", badge="本阶段新建", badge_color=B_NEW)

box(4.65, cy, 3.85, ch, [
    ("标注流 → 拉流播放", 11.4, True, INK),
    ("转封装 HLS / WebRTC", 8.6, False, C_OUT),
    ("叠加检测框的输出流", 8.2, False, SUB),
    ("拉流端：浏览器 / 大屏", 8.2, True, C_OUT),
], C_OUT, fc="#fff7ed", badge="本阶段新建", badge_color=B_NEW)

box(8.85, cy, 3.85, ch, [
    ("难样本闭环训练", 11.4, True, INK),
    ("难样本 → Blob 存储", 8.6, False, C_LOOP),
    ("AML Pipeline 重训 / 微调", 8.2, False, SUB),
    ("新模型回注 Registry", 8.2, True, C_LOOP),
], C_LOOP, fc="#faf5ff", badge="规划 · 闭环", badge_color=B_OPT)

box(13.05, cy, 4.05, ch, [
    ("平台支撑（横切）", 11.4, True, INK),
    ("Blob 存储 · Key Vault 密钥", 8.6, False, C_SUP),
    ("Azure Monitor / App Insights", 8.2, False, SUB),
    ("Entra ID 统一鉴权", 8.2, False, SUB),
], C_SUP, fc="#eff6ff", badge="Azure 托管", badge_color=B_AZURE)

arrow((4.30, cy + ch / 2), (4.65, cy + ch / 2), "", C_OUT)

# 闭环：C 难样本 → B 模型注册（向上回注）
arrow((12.70, cy + ch * 0.72), (14.95, by), "重训回注模型（闭环）", C_LOOP,
      rad=-0.22, ls=(0, (5, 3)), off=(0.65, 0.30), fs=8.2)

# ---------- 脚注 ----------
ax.text(0.30, 2.18,
        "落地三步：① 识别服务容器化 → AML 在线端点（最小改动，outcome 即上云）；② 接入 WebRTC/RTSP + WebSocket 回传（真·实时全链路）；③ 端上 IoT Edge 初筛 + 难样本闭环重训。",
        ha="left", va="center", fontsize=8.8, color="#64748b", style="italic")
ax.text(0.30, 1.90,
        "诚实评估：算法已就绪，瓶颈在工程——「推拉流接入 + AML 托管 + WebSocket 回传」三块；无不可逾越的技术障碍。AMS 已退役，推拉流需自建(MediaMTX/LiveKit)。",
        ha="left", va="center", fontsize=8.8, color="#94a3b8", style="italic")

out_dir = Path(__file__).resolve().parents[2] / "assets"
out_dir.mkdir(parents=True, exist_ok=True)
svg_path = out_dir / "architecture-phase5.svg"
png_path = out_dir / "architecture-phase5.png"
fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
fig.savefig(png_path, dpi=170, bbox_inches="tight", facecolor="white")
print(f"saved: {svg_path}")
print(f"saved: {png_path}")
