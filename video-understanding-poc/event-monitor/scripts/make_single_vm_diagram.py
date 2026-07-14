"""生成【单 VM GPU 部署方案】架构图（SVG + PNG）- 宽敞布局版

设计：更大画布 + 更多留白 + 无重叠
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

# 大画布 + 宽敞布局
FIG_W, FIG_H = 22.0, 20.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

INK   = "#0f172a"
SUB   = "#475569"
C_USER    = "#ca8a04"
C_CODE    = "#7c3aed"
C_VM      = "#dc2626"
C_MODEL   = "#0891b2"
C_LOCAL   = "#0d9488"
C_LLM     = "#2563eb"
C_OBS     = "#64748b"
C_GALLERY = "#059669"

F_UPLOAD  = "#9333ea"
F_DEPLOY  = "#2563eb"
F_MOUNT_R = "#0891b2"
F_LOCAL_W = "#059669"
F_LLM     = "#dc2626"


def box(x, y, w, h, lines, color, fc="#ffffff", lw=1.8):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.14",
        linewidth=lw, edgecolor=color, facecolor=fc, zorder=3)
    ax.add_patch(p)
    line_h = 0.32
    block = line_h * len(lines)
    natural_top = y + (h + block) / 2 - line_h / 2
    # 确保文字块顶部不超出框内 0.28（防挤压边框）
    max_top = y + h - 0.28
    top = min(natural_top, max_top)
    cursor = top
    for text, size, bold, col in lines:
        ax.text(x + w / 2, cursor, text, ha="center", va="center",
                fontsize=size, fontweight=("bold" if bold else "normal"),
                color=col, zorder=4)
        cursor -= line_h


def group(x, y, w, h, title, color):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.20",
        linewidth=2.2, edgecolor=color, facecolor="#f8fafc", zorder=1)
    ax.add_patch(p)
    ax.text(x + 0.4, y + h, title, ha="left", va="center",
            fontsize=14, fontweight="bold", color=color, zorder=6,
            bbox=dict(boxstyle="round,pad=0.32", fc="#ffffff", ec=color, lw=1.6))


def arrow(x1, y1, x2, y2, color, style="-", label=None, label_pos=0.5,
          label_offset=(0, 0.20), lw=2.0, curve=0.0):
    connectionstyle = f"arc3,rad={curve}" if curve else "arc3"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=17,
        color=color, linestyle=style, linewidth=lw,
        connectionstyle=connectionstyle, zorder=5))
    if label:
        mx = x1 + (x2 - x1) * label_pos + label_offset[0]
        my = y1 + (y2 - y1) * label_pos + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=9.5, color=color, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.24", fc="#ffffff", ec=color, lw=1.0))


# ============================================================================
# 顶部标题（y=18.9~19.7）
# ============================================================================
ax.text(FIG_W / 2, FIG_H - 0.35, "视频理解 POC — 单 VM GPU 部署方案（当前实际使用）",
        ha="center", va="center", fontsize=20, fontweight="bold", color=INK)
ax.text(FIG_W / 2, FIG_H - 0.90,
        "Southeast Asia (Singapore)  ·  MCAPS 订阅  ·  Resource Group: videopoc-rg",
        ha="center", va="center", fontsize=12, color=SUB)

# ============================================================================
# ① 本地（y=16.4~18.2 · 高度 1.8）
# ============================================================================
group(0.5, 16.4, 21.0, 1.8, "① 本地（你的 Windows 电脑）", C_LOCAL)

box(0.8, 16.6, 4.9, 1.35,
    [("源代码 & Dockerfile", 11.5, True, INK),
     ("app/  experiment/  infra/", 9.5, False, SUB),
     ("Dockerfile.gpu (CUDA 12.1)", 9.5, False, SUB)],
    C_LOCAL)

box(5.9, 16.6, 4.9, 1.35,
    [("本地模型 & 数据集", 11.5, True, INK),
     ("Market-1501 · ChokePoint(待补)", 9.5, False, SUB),
     ("scp 直传 VM OS 盘 ~/vp", 9.5, False, SUB)],
    C_LOCAL)

box(11.0, 16.6, 4.9, 1.35,
    [("CLI 工具", 11.5, True, INK),
     ("az cli · ssh · docker", 9.5, False, SUB),
     ("SSH 密钥 ~/.ssh/id_rsa", 9.5, False, SUB)],
    C_LOCAL)

box(16.1, 16.6, 5.2, 1.35,
    [("infra 脚本", 11.5, True, INK),
     ("quick-vm.ps1 (建 VM)", 9.5, False, SUB),
     ("scp 传数据 · ssh 运维", 9.5, False, SUB)],
    C_LOCAL)

# ============================================================================
# ② Azure Resource Group（y=2.7~15.8，高度 13.1，宽敞）
# ============================================================================
group(0.5, 2.7, 21.0, 13.1, "② Azure Resource Group：videopoc-rg (Southeast Asia)", C_VM)

# ---- 左半：GPU VM 大框 (y=3.0~15.0, x=0.9~13.4) ----
group(0.9, 3.0, 12.5, 12.0, "GPU VM：videopoc-gpu-vm  (NC4as_T4_v3 · Spot · 40.65.151.20)", C_VM)

# VM OS 层（y=13.5~14.6）
box(1.2, 13.55, 11.9, 1.1,
    [("Ubuntu 22.04  +  NVIDIA Driver 595  +  Docker 29  +  NVIDIA Container Toolkit", 10.5, True, INK),
     ("OS 盘 = Premium SSD (加速开机+模型加载) · Docker/容器开机自启", 9, False, SUB)],
    C_VM, fc="#fee2e2")

# Docker Container（大框，向下占据原挂载区）
box(1.2, 5.55, 11.9, 7.7,
    [("Docker Container：videopoc  (镜像 videopoc:gpu · 17GB)", 13, True, "#b91c1c"),
     ("在 VM 上 docker build · --restart unless-stopped 开机自启 · --env-file .env", 9.5, False, SUB),
     ("", 4, False, INK),
     ("FastAPI (gunicorn+uvicorn, 宿主 -p 8000:8000)", 11, True, INK),
     ("├─  /event-monitor 前端页（上传视频→端到端）", 9.5, False, INK),
     ("└─  8 个 router + service (monolithic 单进程)", 9.5, False, INK),
     ("", 4, False, INK),
     ("推理模块 (同进程 import, 走 CUDA T4 · 开机预热预加载)", 11, True, INK),
     ("YOLOv8 + BoT-SORT / ByteTrack", 9.5, False, INK),
     ("InsightFace buffalo_l / ArcFace  (512d 人脸) · AdaFace 待补", 9.5, False, INK),
     ("人形 ReID  (2048d, GPU)", 9.5, False, INK),
     ("步态 OpenGait  (权重待补 → 暂降级跳过)", 9.5, False, INK),
     ("RapidOCR + GFPGAN 超分 (可选)", 9.5, False, INK)],
    "#dc2626", fc="#fef2f2", lw=2.2)

# （挂载映射已并入右侧「持久化磁盘」模块，容器通过 docker -v 挂载）

# 公网访问（y=3.3~4.6）
box(1.2, 3.35, 11.9, 1.25,
    [("公网访问", 11, True, INK),
     ("22 (SSH)  ·  8000 (前端/API)   →   40.65.151.20   · NSG 双层放通(限 167.220/16)", 10, False, INK)],
    C_VM, fc="#fef2f2")

# ---- 右半：VM 持久化磁盘 (OS 盘 ~/vp)，通过 docker -v 挂进容器 ----
group(13.9, 3.0, 7.6, 12.0, "VM 持久化磁盘 · OS 盘 Premium SSD (~/vp)", C_MODEL)

# 数据集
box(14.2, 10.2, 7.0, 4.5,
    [("数据集    ~/vp/data  →  /data   (只读)", 11.5, True, C_MODEL),
     ("", 4, False, INK),
     ("Market-1501 (行人 ReID · 已就位)", 10.5, True, INK),
     ("  query · bounding_box_test / train", 9, False, SUB),
     ("  gt_bbox · gt_query", 9, False, SUB),
     ("", 4, False, INK),
     ("ChokePoint (门廊监控 · 待补)", 10.5, True, SUB)],
    C_MODEL)

# 模型权重 & 缓存
box(14.2, 6.4, 7.0, 3.5,
    [("模型权重 & 缓存    (只读)", 11.5, True, C_MODEL),
     ("~/vp/models→/models · ~/vp/apphome→/home/appuser", 8, False, SUB),
     ("", 4, False, INK),
     ("YOLOv8 检测权重 (ultralytics)", 9.5, False, INK),
     ("InsightFace buffalo_l 人脸 512d (601MB)", 9.5, False, INK),
     ("GFPGAN 超分权重 (首次自动下载)", 9.5, False, INK),
     ("ReID / torch hub 缓存", 9.5, False, INK)],
    C_MODEL, fc="#f0f9ff")

# 运行产出
box(14.2, 3.4, 7.0, 2.9,
    [("运行产出    (读写)", 11.5, True, C_GALLERY),
     ("", 4, False, INK),
     ("~/vp/gallery → /gallery", 10, True, INK),
     ("  FAISS 向量索引 (主体身份档案)", 9, False, SUB),
     ("~/vp/results → /results", 10, True, INK),
     ("  事件 JSON · 关键帧图 · 报告", 9, False, SUB)],
    C_GALLERY, fc="#f0fdf4")

# ============================================================================
# ③ 外部依赖 & 平台支撑（y=0.4~2.4）
# ============================================================================
group(0.5, 0.4, 21.0, 2.0, "③ 外部依赖 & 平台支撑", C_LLM)

box(0.8, 0.6, 4.9, 1.65,
    [("Azure OpenAI (外部订阅)", 11.5, True, INK),
     ("gpt-4o (api 2024-10-21)", 10, False, SUB),
     ("REST API + Key", 10, False, SUB),
     ("事件精筛 + 场景理解", 10, False, SUB),
     ("容器 --env-file .env 注入 Key", 10, False, SUB)],
    C_LLM)

box(5.9, 0.6, 4.9, 1.65,
    [("SSH 客户端 (你本机)", 11.5, True, INK),
     ("ssh azureuser@40.65.151.20", 10, False, SUB),
     ("~/.ssh/id_rsa (免密登录)", 10, False, SUB),
     ("scp 传数据集 · ssh 运维", 10, False, SUB)],
    C_LOCAL)

box(11.0, 0.6, 4.9, 1.65,
    [("成本控制", 11.5, True, INK),
     ("Spot 算力 ~\\$0.055/hr", 10, False, SUB),
     ("停机 deallocate = \\$0/hr (算力)", 10, False, SUB),
     ("每天8h≈\\$13/月 + Premium盘~\\$12/月", 9.5, True, "#059669"),
     ("(Spot 偶被抢占, 几率低)", 10, False, SUB)],
    C_OBS)

box(16.1, 0.6, 5.2, 1.65,
    [("未来演进（选做）", 11.5, True, INK),
     ("Blob CSI + AKS: 多用户", 10, False, SUB),
     ("Copilot Studio 前端接入", 10, False, SUB),
     ("Ingress + Custom Domain", 10, False, SUB),
     ("加 Log Analytics + AppInsights", 10, False, SUB)],
    "#94a3b8", fc="#f1f5f9")

# ============================================================================
# 数据流箭头（避开所有框中心）
# ============================================================================

# ① 本地代码 → VM  (从"源代码"框底 → VM OS 层顶部；镜像在 VM 上 build)
arrow(3.2, 16.6, 6.0, 14.65, F_DEPLOY, style="--",
      label="① scp 代码 · VM 上 build",
      label_pos=0.65, curve=0.10, lw=2.0, label_offset=(-2.7, 0.15))

# ② 本地数据 → VM 持久化磁盘  (从"本地模型&数据集"框底 → 磁盘面板顶)
arrow(8.4, 16.6, 17.7, 14.9, F_UPLOAD, style="--",
      label="② scp 数据/模型",
      label_pos=0.55, curve=0.25, lw=2.0, label_offset=(2.5, 0.2))

# ③ OS 盘目录 → 容器  (docker -v 把磁盘挂进容器)
arrow(14.2, 8.3, 13.1, 8.3, F_MOUNT_R, style="--",
      label="③ docker -v 挂载",
      label_pos=0.5, curve=0.0, lw=2.0, label_offset=(0, 0.30))

# ④ SSH 客户端 → VM 公网访问
arrow(7.0, 2.4, 7.0, 3.35, F_LOCAL_W, style="-",
      label="④ SSH / HTTP",
      label_pos=0.5, curve=0.0, lw=2.2, label_offset=(1.6, 0))

# ⑤ 容器 → Azure OpenAI  (事件理解调 gpt-4o)
arrow(2.3, 5.55, 2.3, 2.4, F_LLM, style="-",
      label="⑤ 容器调 gpt-4o",
      label_pos=0.15, curve=-0.28, lw=2.0, label_offset=(1.9, 0))

# ============================================================================
# 图例（放在 group ② 底部内侧，靠上防遮挡 group ③ 标题）
# ============================================================================
legend_y = 2.9
ax.text(3.0, legend_y, "数据流：", fontsize=11, fontweight="bold", color=INK, va="center")
items = [
    (F_DEPLOY, "--", "① scp 代码 (VM 上 build)"),
    (F_UPLOAD, "--", "② scp 数据/模型 (一次)"),
    (F_MOUNT_R,"--", "③ docker -v 挂载"),
    (F_LOCAL_W,"-",  "④ 用户 SSH/HTTP"),
    (F_LLM,    "-",  "⑤ 容器调 gpt-4o"),
]
xcur = 4.6
for col, ls, txt in items:
    ax.plot([xcur, xcur + 0.6], [legend_y, legend_y],
            color=col, linestyle=ls, linewidth=2.6)
    ax.text(xcur + 0.75, legend_y, txt, fontsize=10, color=INK, va="center")
    xcur += 3.35

# ============================================================================
# 输出
# ============================================================================
out_dir = Path(__file__).resolve().parent.parent / "docs" / "cloud-deploy"
out_dir.mkdir(parents=True, exist_ok=True)
svg_path = out_dir / "cloud-deploy-single-vm.svg"
png_path = out_dir / "cloud-deploy-single-vm.png"

plt.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.2)
plt.savefig(png_path, format="png", bbox_inches="tight", pad_inches=0.2, dpi=140)
print(f"[ok] wrote {svg_path}")
print(f"[ok] wrote {png_path}")
