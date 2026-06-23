# -*- coding: utf-8 -*-
"""生成视频理解 PoC 【Phase 4】事件理解 logic flow（两段选帧 + 可插拔事件提供器）。

输出：assets/architecture-phase4.{png}（与其它 phase 架构图同目录）。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(18, 19))
ax.set_xlim(0, 120); ax.set_ylim(0, 122); ax.invert_yaxis(); ax.axis("off")

BLUE=("#CFE4FA","#0078D4"); ORANGE=("#FFE9B0","#E8590C"); GREEN=("#DFF6DD","#107C10")
PURPLE=("#E8DAEF","#5C2D91"); TEAL=("#C5F0F5","#0C8599"); GRAY=("#E3E3E3","#495057"); RED=("#FDE7E9","#D13438")

def rbox(cx, cy, w, h, t, col, fs=10, lw=1.8, ls="-"):
    fc, ec = col
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.13,rounding_size=0.6", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(cx, cy, t, ha="center", va="center", fontsize=fs, color="#111")

def band(x, y, w, h, ec, label, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc="none", ec=ec, lw=1.4, ls="--"))
    ax.text(x + w/2, y + 2.2, label, ha="center", va="center", fontsize=fs, color=ec, fontweight="bold")

def arr(x1, y1, x2, y2, label=None, fs=9.5):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.7))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2, label, fontsize=fs, color="#333", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))

def badge(cx, cy, t, ec):
    ax.text(cx, cy, t, ha="center", va="center", fontsize=10.5, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc=ec, ec="none"))

ax.text(60, 3, "Phase 4 · 客户对齐：事件理解 logic flow（两段选帧 + 可插拔事件提供器）", ha="center", fontsize=15.5, fontweight="bold")
ax.text(60, 7.5, "业界 best practice：先定时均匀采样 → 各模型提特征 → 再内容感知关键帧选择(降图数) → 喂多模态 LLM。两次选帧都用最基础的方法即可。", ha="center", fontsize=10, color="#555")

rbox(60, 15, 38, 6, "① 视频流（实时摄像头 / 整段视频）", BLUE, 10.5)
rbox(60, 26, 64, 8, "【选帧①】定时均匀采样 ≈ 4 fps（基础方法）·（可选：画面静止跳过）", ORANGE, 10.5, lw=2.4)
badge(13, 26, "选帧①\n定时·均匀", "#E8590C")

band(7, 34, 106, 36, "#107C10", "② 事件提供器注册表（可插拔 · 对每帧提特征，4fps）")
rbox(31, 46, 34, 8, "人物身份 ★高频·本期做\n人脸 + 人形ReID + 步态(LiDAR可选)", PURPLE, 9.0)
rbox(66, 46, 26, 8, "宠物身份/事件\n宠物脸(UniFace 思路)", GREEN, 9.0)
rbox(96, 46, 26, 8, "车辆事件\n（驶入/离开）", GREEN, 9.2)
rbox(31, 57, 34, 8, "包裹/物品 ＋OCR\n投递·被取·移动·文字/车牌", GREEN, 9.0)
rbox(66, 57, 26, 8, "区域/越界/徘徊", GREEN, 9.2)
rbox(96, 57, 26, 8, "异常事件\n跌倒·火焰·玻璃破碎", GREEN, 9.2)
rbox(60, 66.5, 102, 5, "＋ 预留槽：新事件类型即插即用（统一 provider 接口：检测 → 特征 → 结构化信号）", GRAY, 9.3, ls="--")

rbox(60, 77.5, 92, 8,
     "③ 结构化事件信号（全部帧 · 文本紧凑 → 可全量带给 LLM）\n身份/轨迹 ＋ 目标级轨迹 ＋ 手物交互坐标（hand/dish traj…）", TEAL, 9.4)

rbox(60, 90, 96, 11,
     "【选帧②】内容感知关键帧选择 —— 把喂 LLM 的【图片数】砍下来（4fps 几百张 → 几十张）\n"
     "基础方法：相邻相似去重(灰度指纹/embedding 余弦) + 事件触发(有事件帧优先) + 每 track 最佳帧\n"
     "注意：保留时间顺序、别过度删（漏事件）", ORANGE, 9.6, lw=2.4)
badge(9, 90, "选帧②\n智能·降图数", "#E8590C")

rbox(60, 106, 86, 8.5,
     "④ 多模态 gpt-4o · 跨帧【事件理解】\n输入 = 少量关键帧(图片) + 全部结构化事件信号(文本)", RED, 10)
rbox(60, 118, 108, 5, "⑤ 描述性事件总结 ＋ 视频片段拼接 / 告警 / 检索（人/宠物/车/包裹/异常… 何时·何目标·发生了什么）", TEAL, 9.5)

arr(60, 18, 60, 22)
arr(60, 30, 60, 34)
arr(60, 69, 60, 73.3, "各 provider 汇总")
arr(60, 81.5, 60, 84.5)
arr(60, 95.5, 60, 101.7, "少量关键帧 + 全部信号")
arr(60, 110.3, 60, 115.5)

out_dir = Path(__file__).resolve().parents[2] / "assets"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "architecture-phase4.png"
plt.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
print("saved:", out)
