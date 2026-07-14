"""生成【最简 POC 部署方案】架构图（SVG + PNG）—— 修正布局版。

设计原则：
1. 代码是 monolithic FastAPI，不硬拆 CPU/GPU 微服务
2. Gallery 是 FAISS 文件不是数据库
3. POC 单用户短任务型，不做多副本 HPA
4. 只画代码真实支持的东西
"""
from __future__ import annotations
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

FIG_W, FIG_H = 20.0, 15.5
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

INK   = "#0f172a"
SUB   = "#475569"
C_USER    = "#ca8a04"
C_CODE    = "#7c3aed"
C_POD     = "#dc2626"
C_MODEL   = "#0891b2"
C_GALLERY = "#059669"
C_LLM     = "#2563eb"
C_OBS     = "#64748b"

F_UPLOAD  = "#7c3aed"
F_MOUNT_R = "#0891b2"
F_MOUNT_W = "#059669"
F_LLM     = "#2563eb"
F_OBS     = "#94a3b8"


def box(x, y, w, h, lines, color, fc="#ffffff", badge=None, badge_color=None, lw=1.8):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        linewidth=lw, edgecolor=color, facecolor=fc, zorder=3)
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
        ax.text(x + w / 2, y + 0.18, badge, ha="center", va="center",
                fontsize=8, fontweight="bold", color="#ffffff", zorder=6,
                bbox=dict(boxstyle="round,pad=0.24", fc=bc, ec="none"))


def group(x, y, w, h, title, color, subtitle=None):
    """标题浮在框顶边正上方；副标题另起一行在标题右侧。"""
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        linewidth=2.0, edgecolor=color, facecolor="#f8fafc", zorder=1)
    ax.add_patch(p)
    ax.text(x + 0.35, y + h, title, ha="left", va="center",
            fontsize=12.5, fontweight="bold", color=color, zorder=6,
            bbox=dict(boxstyle="round,pad=0.28", fc="#ffffff", ec=color, lw=1.4))
    if subtitle:
        # 副标题贴框内顶部，距顶边 -0.30
        ax.text(x + w - 0.35, y + h - 0.30, subtitle, ha="right", va="center",
                fontsize=9, color=SUB, zorder=6, style="italic")


def arrow(x1, y1, x2, y2, color, style="-", label=None, label_pos=0.5,
          label_offset=(0, 0.18), lw=1.8, curve=0.0):
    connectionstyle = f"arc3,rad={curve}" if curve else "arc3"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=15,
        color=color, linestyle=style, linewidth=lw,
        connectionstyle=connectionstyle, zorder=5))
    if label:
        mx = x1 + (x2 - x1) * label_pos + label_offset[0]
        my = y1 + (y2 - y1) * label_pos + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=8, color=color, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="#ffffff", ec=color, lw=0.8))


# ============================================================================
# 顶部标题（y=14.6~15.3）
# ============================================================================
ax.text(FIG_W / 2, FIG_H - 0.30, "视频理解 POC — 最简部署方案 (单 Pod on GPU Node)",
        ha="center", va="center", fontsize=18, fontweight="bold", color=INK)
ax.text(FIG_W / 2, FIG_H - 0.72,
        "Southeast Asia (Singapore)  ·  MCAPS 订阅  ·  Resource Group: videopoc-rg",
        ha="center", va="center", fontsize=11, color=SUB)
ax.text(FIG_W / 2, FIG_H - 1.10,
        "设计原则：代码是 monolithic → 不拆微服务；单用户短任务 → 不做 HPA；autoscale-to-zero → 零空闲成本",
        ha="center", va="center", fontsize=10, color=SUB, style="italic")

# ============================================================================
# ① 用户 (y=12.5~13.8)
# ============================================================================
box(0.6, 12.6, 3.6, 1.4,
    [("用户 / 浏览器", 11, True, INK),
     ("上传视频 (.mp4)", 9, False, SUB),
     ("→ POST /api/event-monitor/understand", 8.5, False, SUB)],
    C_USER, badge="Client", badge_color=C_USER)

# ============================================================================
# ② AKS 集群大组 (y=3.5~12.0)  左边
# ============================================================================
group(0.4, 3.5, 12.6, 8.5, "② AKS 集群 videopoc-aks (Kubernetes 1.29)",
      C_POD, subtitle="2 个 node pool")

# system pool 小框（放在 AKS 组左上）
box(0.7, 9.9, 3.8, 1.5,
    [("Node Pool: system", 10.5, True, INK),
     ("Standard_D2s_v5 × 1", 9, False, SUB),
     ("kube-system, CSI drivers", 8.5, False, SUB),
     ("~$75/月", 8.5, False, SUB)],
    "#64748b")

# gpu pool + Pod 子组
group(4.8, 3.9, 8.0, 7.5, "Node Pool: gpupool  (T4 GPU · 16 vCPU 已批)",
      "#dc2626", subtitle="NC4as_T4_v3 · Spot · min=0 max=1")

# Pod 大框
box(5.1, 5.5, 7.4, 5.4,
    [("Pod: video-poc-all-in-one", 12, True, "#b91c1c"),
     ("", 6, False, INK),
     ("FastAPI (uvicorn, port 8000)", 10, True, INK),
     ("├─ /event-monitor 页面（上传视频→端到端）", 9, False, INK),
     ("├─ /analyze-frame  路由（实时抽帧）", 9, False, INK),
     ("└─ 8 个 router + service（单进程 monolithic）", 9, False, INK),
     ("", 6, False, INK),
     ("推理模块（同进程 import，全走 CUDA T4）", 10, True, INK),
     ("├─ YOLOv8 + BoT-SORT/ByteTrack", 9, False, INK),
     ("├─ InsightFace buffalo_l  (512d 人脸)", 9, False, INK),
     ("├─ OSNet-AIN ReID  (4096d 人形)", 9, False, INK),
     ("├─ SkeletonGait++ / OpenGait  (4096d 步态)", 9, False, INK),
     ("├─ RapidOCR (PP-OCRv4)", 9, False, INK),
     ("└─ GFPGAN 超分 (可选)", 9, False, INK)],
    "#dc2626", fc="#fef2f2", lw=2.0)

# Service + Ingress（Pod 下面）
box(5.1, 4.15, 7.4, 1.15,
    [("Service (ClusterIP)  +  Ingress (Azure App Routing)", 10.5, True, INK),
     ("→ 公网 HTTPS  http://<external-ip>/event-monitor", 9, False, SUB)],
    "#dc2626", fc="#fff5f5")

# ============================================================================
# ③ 存储层 (y=3.5~12.0)  右边
# ============================================================================
group(13.3, 3.5, 6.3, 8.5, "③ Azure 存储 (Pod 挂载)",
      C_MODEL, subtitle="所有持久化数据")

# Blob Storage
box(13.55, 8.1, 5.8, 3.2,
    [("Azure Blob Storage (Standard_LRS)", 10.5, True, INK),
     ("videopocst<hash>", 8.5, False, SUB),
     ("", 6, False, INK),
     ("├─ models/   (只读 RO)", 9.5, False, INK),
     ("│   YOLOv8 · InsightFace · OSNet", 8, False, SUB),
     ("│   GFPGAN · RapidOCR · OpenGait", 8, False, SUB),
     ("├─ datasets/ (只读 RO)", 9.5, False, INK),
     ("│   ChokePoint · Market-1501", 8, False, SUB),
     ("├─ videos/   (读写 RW)  用户上传视频", 9.5, False, INK),
     ("└─ results/  (读写 RW)  JSON+关键帧+报告", 9.5, False, INK)],
    C_MODEL)

# Files Premium
box(13.55, 5.5, 5.8, 2.4,
    [("Azure Files Premium (Premium_LRS)", 10.5, True, INK),
     ("videopocfs<hash> · share: gallery", 8.5, False, SUB),
     ("SMB · POSIX 语义 · 支持多 Pod RWX", 8, False, SUB),
     ("", 6, False, INK),
     ("三个 FAISS 向量索引文件", 10, True, C_GALLERY),
     ("├─ person_gallery.faiss  (4096d 人形)", 9, False, INK),
     ("├─ face_gallery.faiss    (512d 人脸)", 9, False, INK),
     ("└─ gait_gallery.faiss    (4096d 步态)", 9, False, INK)],
    C_GALLERY)

# 注意说明
box(13.55, 3.7, 5.8, 1.55,
    [("澄清：我们没有传统数据库", 10, True, "#b91c1c"),
     ("无 PostgreSQL / Cosmos / MySQL", 8.5, False, SUB),
     ("Gallery = FAISS 向量索引文件 (Files)", 8.5, False, SUB),
     ("Session state = Pod 内存 (重启即丢)", 8.5, False, SUB)],
    "#b91c1c", fc="#fef2f2", lw=1.4)

# ============================================================================
# ④ 底部：外部依赖 & 平台支撑 (y=0.7~2.5)
# ============================================================================
group(0.4, 0.7, 19.2, 1.8, "④ 外部依赖 & 平台支撑（跨订阅 / 全托管，不属 videopoc-rg）",
      C_LLM)

# Azure OpenAI
box(0.7, 0.95, 4.5, 1.35,
    [("Azure OpenAI (外部订阅)", 10.5, True, INK),
     ("gpt-4o / gpt-4o-mini", 9, False, SUB),
     ("REST API + AAD Key", 8.5, False, SUB),
     ("事件精筛 + 场景理解", 8.5, False, SUB)],
    C_LLM)

# ACR
box(5.35, 0.95, 4.5, 1.35,
    [("ACR (容器镜像仓库)", 10.5, True, INK),
     ("videopocacr<hash>", 9, False, SUB),
     ("镜像: video-poc-gpu:latest", 8.5, False, SUB),
     ("Pod 启动时拉", 8.5, False, SUB)],
    C_CODE)

# LAW + AppInsights
box(10.0, 0.95, 4.5, 1.35,
    [("Log Analytics + AppInsights", 10.5, True, INK),
     ("videopoc-law · videopoc-ai", 9, False, SUB),
     ("容器日志 · APM · 事件/异常", 8.5, False, SUB)],
    C_OBS)

# 本地
box(14.65, 0.95, 4.7, 1.35,
    [("本地 (你的电脑)", 10.5, True, INK),
     ("源代码 → az acr build → 镜像", 9, False, SUB),
     ("upload-models.ps1 → Blob", 8.5, False, SUB),
     ("infra\\deploy.ps1 一键跑", 8.5, False, SUB)],
    "#0f766e")

# ============================================================================
# 数据流箭头
# ============================================================================

# 用户 → Ingress/Service (进入 Pod)
arrow(2.4, 12.6, 6.5, 5.3, F_UPLOAD, style="-",
      label="① 上传视频\n(HTTPS)", label_pos=0.35, curve=0.20, lw=2.2,
      label_offset=(0.5, 0.2))

# Pod → Blob (models/datasets 只读)
arrow(12.5, 9.5, 13.55, 10.3, F_MOUNT_R, style="--",
      label="② models/datasets\n只读 CSI 挂载", label_pos=0.5,
      label_offset=(-0.4, 0.15), curve=0.10, lw=1.7)

# Pod → Blob (videos/results 读写)
arrow(12.5, 8.5, 13.55, 8.7, F_MOUNT_W, style="--",
      label="③ videos/results\n读写", label_pos=0.5,
      label_offset=(-0.4, -0.10), curve=-0.05, lw=1.7)

# Pod → Files (gallery RWX)
arrow(12.5, 7.0, 13.55, 6.7, F_MOUNT_W, style="--",
      label="④ gallery (RWX)\nFAISS 向量库", label_pos=0.5,
      label_offset=(-0.3, -0.20), curve=-0.10, lw=1.7)

# Pod → Azure OpenAI
arrow(6.5, 4.15, 3.0, 2.30, F_LLM, style="-",
      label="⑤ LLM 调用\ngpt-4o", label_pos=0.5, curve=-0.15, lw=2.0,
      label_offset=(-0.5, 0.05))

# Pod → ACR (image pull)
arrow(8.0, 4.15, 7.6, 2.30, F_LLM, style="--",
      label="image pull", label_pos=0.5, label_offset=(0.30, 0.10), curve=0.10, lw=1.5)

# Pod → LAW (telemetry)
arrow(10.0, 4.15, 12.0, 2.30, F_OBS, style=":",
      label="logs / metrics", label_pos=0.5, label_offset=(0.5, 0.10), curve=-0.10, lw=1.6)

# ============================================================================
# 图例（顶部横向，标题栏下方）
# ============================================================================
legend_y = FIG_H - 1.55
ax.text(4.3, legend_y, "数据流：", fontsize=10, fontweight="bold", color=INK, va="center")

items = [
    (F_UPLOAD, "-",  "用户 HTTPS"),
    (F_MOUNT_R,"--", "只读挂载"),
    (F_MOUNT_W,"--", "读写挂载"),
    (F_LLM,    "-",  "外部 LLM"),
    (F_OBS,    ":",  "遥测"),
]
xcur = 5.3
for col, ls, txt in items:
    ax.plot([xcur, xcur + 0.55], [legend_y, legend_y],
            color=col, linestyle=ls, linewidth=2.4)
    ax.text(xcur + 0.65, legend_y, txt, fontsize=9, color=INK, va="center")
    xcur += 2.6

# ============================================================================
# 输出
# ============================================================================
out_dir = Path(__file__).resolve().parent.parent / "docs"
out_dir.mkdir(exist_ok=True)
svg_path = out_dir / "cloud-deploy-simple.svg"
png_path = out_dir / "cloud-deploy-simple.png"

plt.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.15)
plt.savefig(png_path, format="png", bbox_inches="tight", pad_inches=0.15, dpi=140)
print(f"[ok] wrote {svg_path}")
print(f"[ok] wrote {png_path}")
