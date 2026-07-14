"""生成【云上部署当前状态】架构 + 数据流 logic flow 图（SVG + PNG）。

本图只画本 Phase（Azure 上云 P0）已就绪的资源拓扑与运行时数据流向，不重复画本地
识别算法内部结构（那些请看 phase4-logic-flow.svg）。

四条 flow 用颜色区分：
- 蓝虚线：Build / Deploy（本地 → ACR / AKS，一次性）
- 紫虚线：Upload（本地脚本 → Blob，一次性）
- 绿实线：Runtime 数据流（用户 → Service → Pod → 挂载卷）
- 橙实线：外部依赖调用（Pod → Azure OpenAI）
- 灰点线：Telemetry / 监控（Pod → AppInsights → LAW）
"""
from __future__ import annotations
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

FIG_W, FIG_H = 21.5, 16.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ---------- 配色 ----------
C_LOCAL = "#0f766e"     # 本地 dev
C_RG    = "#1e40af"     # 资源组边框
C_ACR   = "#7c3aed"     # ACR
C_BLOB  = "#0891b2"     # Blob
C_FILES = "#0d9488"     # Files
C_AKS   = "#dc2626"     # AKS
C_POD   = "#ea580c"     # Pod / Deployment
C_OBS   = "#64748b"     # 监控
C_EXT   = "#ca8a04"     # 外部依赖（Azure OpenAI）
INK     = "#0f172a"
SUB     = "#475569"

# 数据流颜色
F_DEPLOY = "#2563eb"    # 蓝虚线 = build/deploy
F_UPLOAD = "#9333ea"    # 紫虚线 = upload data
F_RUN    = "#16a34a"    # 绿实线 = runtime
F_LLM    = "#ea580c"    # 橙实线 = external LLM
F_OBS    = "#94a3b8"    # 灰点线 = telemetry


def box(x, y, w, h, lines, color, fc="#ffffff", badge=None, badge_color=None, lw=1.8):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw, edgecolor=color, facecolor=fc, zorder=3,
    )
    ax.add_patch(p)
    line_h = 0.28
    block = line_h * len(lines)
    cursor = y + (h + block) / 2 - line_h / 2
    for text, size, bold, col in lines:
        ax.text(x + w / 2, cursor, text, ha="center", va="center",
                fontsize=size, fontweight=("bold" if bold else "normal"),
                color=col, zorder=4)
        cursor -= line_h
    if badge:
        bc = badge_color or color
        ax.text(x + w / 2, y + 0.15, badge, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="#ffffff", zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", fc=bc, ec="none"))
    return (x, y, w, h)


def group(x, y, w, h, title, color, subtitle=None, dashed=False):
    """带标题的分组容器框（大框，用于圈起来一堆小框）。

    标题浮在边框上（像 fieldset legend），子内容可占据整个框内空间。
    """
    ls = "--" if dashed else "-"
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        linewidth=2.0, edgecolor=color, facecolor="#f8fafc",
        linestyle=ls, zorder=1,
    )
    ax.add_patch(p)
    # 标题贴在框的顶边上（居中/略偏左）
    ax.text(x + 0.35, y + h, title, ha="left", va="center",
            fontsize=12, fontweight="bold", color=color, zorder=6,
            bbox=dict(boxstyle="round,pad=0.28", fc="#ffffff", ec=color, lw=1.4))
    if subtitle:
        # 副标题贴在框的顶边上，右侧
        ax.text(x + w - 0.35, y + h, subtitle, ha="right", va="center",
                fontsize=8.5, color=SUB, zorder=6, style="italic",
                bbox=dict(boxstyle="round,pad=0.20", fc="#f8fafc", ec="none"))


def arrow(x1, y1, x2, y2, color, style="-", label=None, label_pos=0.5,
          label_offset=(0, 0.18), lw=1.6, curve=0.0):
    connectionstyle = f"arc3,rad={curve}" if curve else "arc3"
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        color=color, linestyle=style, linewidth=lw,
        connectionstyle=connectionstyle, zorder=5,
    ))
    if label:
        mx = x1 + (x2 - x1) * label_pos + label_offset[0]
        my = y1 + (y2 - y1) * label_pos + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=8, color=color, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="#ffffff", ec=color, lw=0.8))


# ============================================================================
# 顶部标题
# ============================================================================
ax.text(FIG_W / 2, FIG_H - 0.35, "视频理解 POC — Azure 上云 P0 架构 & 运行时数据流",
        ha="center", va="center", fontsize=18, fontweight="bold", color=INK)
ax.text(FIG_W / 2, FIG_H - 0.75,
        "Region: Southeast Asia (Singapore)  |  Subscription: MCAPS-Hybrid-REQ-155066-2026  |  Resource Group: videopoc-rg",
        ha="center", va="center", fontsize=11, color=SUB)

# ============================================================================
# 上层：本地 (Windows 开发机)
# ============================================================================
group(0.4, 11.4, 19.7, 2.2, "① 本地 (Windows 开发机)",
      C_LOCAL, subtitle="源码 / 部署脚本 / 数据准备的源头，不承担线上流量")

# 源码
box(0.9, 12.1, 3.6, 0.95,
    [("源代码仓库", 10, True, INK),
     ("app/  experiment/  charts/", 8, False, SUB),
     ("infra/  docs/  scripts/", 8, False, SUB)],
    C_LOCAL)

# Docker 上下文
box(4.9, 12.1, 3.4, 0.95,
    [("Dockerfile.cpu / .gpu", 10, True, INK),
     ("镜像构建源（不打包权重）", 8, False, SUB),
     ("模型/数据走挂载", 8, False, SUB)],
    C_LOCAL)

# 部署脚本
box(8.7, 12.1, 4.2, 0.95,
    [("部署脚本（PowerShell）", 10, True, INK),
     ("deploy.ps1 · add-gpu-pool.ps1", 8, False, SUB),
     ("upload-models.ps1 · upload-datasets.ps1", 8, False, SUB)],
    C_LOCAL)

# CLI 工具
box(13.3, 12.1, 3.4, 0.95,
    [("CLI 工具链", 10, True, INK),
     ("az cli 2.87  ·  kubectl 1.36", 8, False, SUB),
     ("helm 4.2  ·  bicep", 8, False, SUB)],
    C_LOCAL)

# 本地数据集/权重
box(17.0, 12.1, 3.0, 0.95,
    [("本地数据 & 权重", 10, True, INK),
     ("data/  gfpgan/  models/", 8, False, SUB),
     ("待 upload 到 Blob", 8, False, SUB)],
    C_LOCAL, badge="源头", badge_color=C_LOCAL)

# ============================================================================
# 中层：Azure 资源组 videopoc-ea（大分组容器）
# ============================================================================
group(0.4, 3.0, 19.7, 8.0, "② Azure Resource Group：videopoc-rg (Southeast Asia)",
      C_RG,
      subtitle="所有资源属于同一个 RG，删 RG 即全部清零；停 AKS 保留数据 = 每月 ~$25")

# ---- 存储与镜像分组（左列） ----
group(0.8, 3.6, 5.4, 6.9, "存储 & 镜像仓库",
      C_ACR, subtitle="Pod 通过 CSI / OCI 挂载")

# ACR
box(1.0, 9.4, 5.0, 0.9,
    [("ACR（容器镜像仓库）", 10, True, INK),
     ("videopocacri6hd2v...", 8, False, SUB),
     ("Basic tier · 镜像 video-poc-cpu", 8, False, SUB)],
    C_ACR, badge="ACR", badge_color=C_ACR)

# Blob Storage 大容器 + 4 子容器
box(1.0, 6.2, 5.0, 3.0,
    [("Blob Storage (StorageV2)", 10, True, INK),
     ("videopocsti6hd2v...", 8, False, SUB),
     ("Standard_LRS  ·  Hot tier", 8, False, SUB),
     ("", 6, False, SUB),
     ("├─ models/     (模型权重 RO)", 8.5, False, INK),
     ("├─ datasets/   (数据集 RO)", 8.5, False, INK),
     ("├─ results/    (实验输出 RW)", 8.5, False, INK),
     ("└─ videos/     (视频源 RO)", 8.5, False, INK)],
    C_BLOB, badge="Blob", badge_color=C_BLOB)

# Files Premium
box(1.0, 3.8, 5.0, 2.2,
    [("Files Premium (FileStorage)", 10, True, INK),
     ("videopocfsi6hd2v...", 8, False, SUB),
     ("Premium_LRS  ·  100 GB 起", 8, False, SUB),
     ("", 6, False, SUB),
     ("└─ gallery share (FAISS 索引)", 8.5, False, INK),
     ("多 Pod POSIX 一致性 RWX", 8, False, SUB)],
    C_FILES, badge="Files", badge_color=C_FILES)

# ---- AKS 集群分组（中列大框）----
group(6.6, 3.6, 8.8, 6.9, "AKS 集群 (videopoc-aks-i6hd2v...)",
      C_AKS, subtitle="Kubernetes 1.29 · 双 node pool（GPU 池预留）")

# system pool
box(6.9, 9.0, 4.1, 1.3,
    [("Node Pool: system", 10, True, INK),
     ("Standard_D2s_v5  ·  1 node", 8, False, SUB),
     ("kube-system / coredns / csi-drivers", 8, False, SUB)],
    C_AKS)

# cpupool
box(11.1, 9.0, 4.1, 1.3,
    [("Node Pool: cpupool", 10, True, INK),
     ("Standard_D4s_v5  ·  autoscale 1-10", 8, False, SUB),
     ("承载 webapi Pod", 8, False, SUB)],
    C_AKS, badge="workload", badge_color=C_AKS)

# gpupool（T4 GPU 已批，可开）
box(6.9, 7.4, 8.3, 1.4,
    [("Node Pool: gpupool（T4 配额已批 · 16 vCPU）", 10, True, "#dc2626"),
     ("Standard_NC4as_T4_v3  ·  Spot  ·  min=0 max=3", 8, False, INK),
     ("add-gpu-pool.ps1 一键添加  ·  taint sku=gpu:NoSchedule", 8, False, SUB)],
    "#dc2626", fc="#fef2f2", badge="ready", badge_color="#dc2626", lw=1.6)

# namespace video-poc 内部
group(6.75, 3.8, 8.5, 3.4, "Namespace: video-poc",
      C_POD, subtitle="业务 workload")

# webapi Deployment
box(6.95, 5.6, 4.0, 1.4,
    [("Deployment: video-poc-webapi", 10, True, INK),
     ("FastAPI (uvicorn) + 识别 pipeline", 8, False, SUB),
     ("HPA 2-10 replicas  ·  CPU 4c/8Gi", 8, False, SUB),
     ("挂 /models /datasets /results /gallery", 8, False, SUB)],
    C_POD, badge="Pod", badge_color=C_POD)

# GPU Deployment (dashed, off)
box(11.05, 5.6, 4.0, 1.4,
    [("Deployment: video-poc-gpu（默认关）", 10, True, "#94a3b8"),
     ("--set gpu.enabled=true 启用", 8, False, "#94a3b8"),
     ("HPA 0-3  ·  nvidia.com/gpu: 1", 8, False, "#94a3b8"),
     ("nodeSelector workload=gpu", 8, False, "#94a3b8")],
    "#94a3b8", fc="#f1f5f9", lw=1.4)

# Service (LoadBalancer)
box(6.95, 4.0, 4.0, 1.3,
    [("Service: LoadBalancer", 10, True, INK),
     ("Azure LB · 公网 EXTERNAL-IP", 8, False, SUB),
     ("port 80 → target 8000", 8, False, SUB)],
    C_POD)

# ConfigMap/Secret
box(11.05, 4.0, 4.0, 1.3,
    [("Secret · ConfigMap", 10, True, INK),
     ("AZURE_OPENAI_KEY / API_KEY", 8, False, SUB),
     ("AppInsights conn string", 8, False, SUB)],
    C_POD)

# ---- 可观测性分组（右列） ----
group(15.6, 3.6, 4.4, 6.9, "监控 & 遥测",
      C_OBS, subtitle="集群 + 应用双层")

# Log Analytics
box(15.8, 8.6, 4.0, 1.6,
    [("Log Analytics Workspace", 10, True, INK),
     ("videopoc-law-i6hd2v...", 8, False, SUB),
     ("Container Insights 已装", 8, False, SUB),
     ("pay-per-GB", 8, False, SUB)],
    C_OBS, badge="LAW", badge_color=C_OBS)

# App Insights
box(15.8, 6.8, 4.0, 1.5,
    [("Application Insights", 10, True, INK),
     ("videopoc-ai-i6hd2v...", 8, False, SUB),
     ("APM · 事件 / 依赖 / 异常", 8, False, SUB)],
    C_OBS, badge="AI", badge_color=C_OBS)

# 后续可加：Azure Monitor Alerts
box(15.8, 5.2, 4.0, 1.3,
    [("(P1) Azure Monitor Alerts", 10, True, "#94a3b8"),
     ("告警规则 / Action Group", 8, False, "#94a3b8"),
     ("尚未配置", 8, False, "#94a3b8")],
    "#94a3b8", fc="#f1f5f9", lw=1.4)

# RBAC 说明
box(15.8, 3.9, 4.0, 1.0,
    [("RBAC (System-Assigned MI)", 9, True, INK),
     ("AKS→ACR: AcrPull", 7.5, False, SUB),
     ("AKS→Blob: Storage Blob Data Owner", 7.5, False, SUB),
     ("AKS→Files: Storage File Data SMB", 7.5, False, SUB)],
    C_OBS)

# ============================================================================
# 底层：外部依赖 + 用户
# ============================================================================
group(0.4, 0.4, 19.7, 2.3, "③ 外部依赖 & 客户端",
      C_EXT, subtitle="不在 videopoc-ea RG 内，跨订阅 / 跨 tenant")

# 客户端
box(0.9, 1.0, 4.0, 1.4,
    [("客户端 (浏览器 / API 调用方)", 10, True, INK),
     ("http://<EXTERNAL-IP>/docs", 8, False, SUB),
     ("上传视频 / 查询事件 / 拉结果", 8, False, SUB),
     ("后续可挂 Copilot Studio 前端", 8, False, "#94a3b8")],
    C_EXT, badge="User", badge_color=C_EXT)

# Azure OpenAI (external)
box(5.3, 1.0, 4.5, 1.4,
    [("Azure OpenAI（外部订阅）", 10, True, INK),
     ("gpt-4o / gpt-4o-mini", 8, False, SUB),
     ("事件精筛 + 场景理解", 8, False, SUB),
     ("REST API + AAD Key", 8, False, SUB)],
    C_EXT, badge="LLM", badge_color=C_EXT)

# GitHub / DevOps
box(10.2, 1.0, 4.5, 1.4,
    [("代码托管 (未接)", 10, True, "#94a3b8"),
     ("GitHub / Azure DevOps", 8, False, "#94a3b8"),
     ("(P1) CI 触发 ACR build", 8, False, "#94a3b8"),
     ("目前是本机 push", 8, False, "#94a3b8")],
    "#94a3b8", fc="#f1f5f9", lw=1.4)

# Copilot Studio 预留
box(15.1, 1.0, 4.9, 1.4,
    [("Copilot Studio Agent（P2 预留）", 10, True, "#94a3b8"),
     ("Custom Connector → /api/v1/copilot/*", 8, False, "#94a3b8"),
     ("对话式查询 · 告警交互 · Teams 发布", 8, False, "#94a3b8"),
     ("需先补齐异步 job API + 稳定端点", 8, False, "#94a3b8")],
    "#94a3b8", fc="#f1f5f9", lw=1.4)

# ============================================================================
# 数据流箭头
# ============================================================================

# —— ① Build / Deploy (蓝虚线，本地 → ACR/AKS) ——
arrow(6.6, 12.1, 3.5, 10.35, F_DEPLOY, style="--",
      label="az acr build\nDockerfile.cpu", label_pos=0.5, curve=0.15, lw=1.8)
arrow(10.8, 12.1, 11.0, 10.35, F_DEPLOY, style="--",
      label="helm upgrade\n→ Deployment", label_pos=0.5, curve=-0.10, lw=1.8)

# —— ② Upload data (紫虚线，本地 → Blob) ——
arrow(18.5, 12.1, 5.0, 8.0, F_UPLOAD, style="--",
      label="upload-models.ps1\nupload-datasets.ps1",
      label_pos=0.35, curve=0.25, lw=1.8)

# —— ③ Runtime: user → Service → Pod (绿实线) ——
arrow(2.9, 2.4, 8.5, 4.0, F_RUN, style="-",
      label="HTTPS 请求", label_pos=0.5, curve=0.20, lw=2.0)
arrow(8.95, 5.3, 8.95, 5.6, F_RUN, style="-", lw=2.0)

# —— ④ Pod → 挂载卷 (绿实线，短箭头表示 CSI 挂载) ——
arrow(6.95, 6.3, 6.2, 7.7, F_RUN, style="-",
      label="Blob CSI mount\n/models /datasets /results", label_pos=0.55,
      label_offset=(-1.0, 0.15), curve=-0.15, lw=1.8)
arrow(6.95, 6.0, 6.2, 4.9, F_RUN, style="-",
      label="Files CSI mount\n/gallery (RWX)", label_pos=0.55,
      label_offset=(-1.0, -0.10), curve=0.15, lw=1.8)

# —— ⑤ Pod → Azure OpenAI (橙实线) ——
arrow(9.0, 5.6, 7.5, 2.4, F_LLM, style="-",
      label="LLM 调用\ngpt-4o", label_pos=0.5, curve=0.20, lw=2.0)

# —— ⑥ Telemetry (灰点线，Pod → AI/LAW) ——
arrow(10.95, 6.3, 15.8, 7.4, F_OBS, style=":",
      label="metrics / logs / traces", label_pos=0.5,
      label_offset=(0, 0.2), curve=-0.10, lw=1.6)
arrow(10.95, 6.0, 15.8, 9.2, F_OBS, style=":", lw=1.6, curve=0.20)

# —— ⑦ AKS 从 ACR 拉镜像 (蓝虚线，集群内部) ——
arrow(6.0, 9.7, 6.9, 9.7, F_DEPLOY, style="--",
      label="image pull", label_pos=0.5, label_offset=(0, 0.22), lw=1.6)

# ============================================================================
# 图例
# ============================================================================
legend_y = 0.05
legend_x = 0.5
ax.text(legend_x, legend_y + 0.20, "数据流图例：", fontsize=9.5,
        fontweight="bold", color=INK, va="center")

items = [
    (F_DEPLOY, "--", "① Build / Deploy（一次性）"),
    (F_UPLOAD, "--", "② Upload 数据 / 权重"),
    (F_RUN,    "-",  "③ Runtime 请求 & 卷挂载"),
    (F_LLM,    "-",  "④ 外部 LLM 调用"),
    (F_OBS,    ":",  "⑤ Telemetry / 监控"),
]
xcur = legend_x + 2.0
for col, ls, txt in items:
    ax.plot([xcur, xcur + 0.5], [legend_y + 0.20, legend_y + 0.20],
            color=col, linestyle=ls, linewidth=2.2)
    ax.text(xcur + 0.6, legend_y + 0.20, txt, fontsize=8.5, color=INK, va="center")
    xcur += 3.5

# ============================================================================
# 输出
# ============================================================================
out_dir = Path(__file__).resolve().parent.parent / "docs"
out_dir.mkdir(exist_ok=True)
svg_path = out_dir / "cloud-architecture.svg"
png_path = out_dir / "cloud-architecture.png"

plt.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.15)
plt.savefig(png_path, format="png", bbox_inches="tight", pad_inches=0.15, dpi=140)
print(f"[ok] wrote {svg_path}")
print(f"[ok] wrote {png_path}")
