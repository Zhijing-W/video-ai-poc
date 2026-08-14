# -*- coding: utf-8 -*-
"""生成干净的 PoC 架构 SVG（手写布局，短箭头、无重叠）。"""
from __future__ import annotations

from pathlib import Path

W, H = 1480, 2680

# colors stroke, fill
BLUE = ("#0078D4", "#E8F3FC")
ORANGE = ("#F7630C", "#FFF4CE")
GREEN = ("#107C10", "#E9F7E9")
PURPLE = ("#5C2D91", "#F3EAF8")
TEAL = ("#0C8599", "#E0F7FA")
RED = ("#D13438", "#FDE7E9")
AMBER = ("#C19C00", "#FFF8E1")
GRAY = ("#605E5C", "#F5F5F5")


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


parts: list[str] = []
parts.append(
    f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#666666"/>
    </marker>
    <style>
      .title {{ font: 700 28px "Microsoft YaHei", "Segoe UI", sans-serif; fill: #111; }}
      .sub {{ font: 400 14px "Microsoft YaHei", "Segoe UI", sans-serif; fill: #444; }}
      .hint {{ font: 400 13px "Microsoft YaHei", "Segoe UI", sans-serif; fill: #666; }}
      .bt {{ font: 700 15px "Microsoft YaHei", "Segoe UI", sans-serif; fill: #111; }}
      .bd {{ font: 400 12.5px "Microsoft YaHei", "Segoe UI", sans-serif; fill: #333; }}
      .badge {{ font: 700 11px "Microsoft YaHei", "Segoe UI", sans-serif; }}
      .ct {{ font: 700 14px "Microsoft YaHei", "Segoe UI", sans-serif; }}
      .box {{ stroke-width: 2; }}
      .ctn {{ fill: none; stroke-width: 2.2; }}
      .arr {{ stroke: #666; stroke-width: 1.8; fill: none; marker-end: url(#arrow); }}
    </style>
  </defs>
  <rect width="100%" height="100%" fill="#ffffff"/>
'''
)


def text(x, y, s, cls="bd", anchor="middle"):
    parts.append(
        f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{esc(s)}</text>'
    )


def multilines(cx, cy, lines, *, title=True, line_h=18):
    """Centered multi-line block around (cx, cy)."""
    n = len(lines)
    total = (n - 1) * line_h
    y0 = cy - total / 2
    for i, line in enumerate(lines):
        cls = "bt" if (title and i == 0) else "bd"
        text(cx, y0 + i * line_h + 5, line, cls=cls)


def badge(x, y, label, on=True):
    # x,y = top-right inside box
    fill = "#DFF6DD" if on else "#FFF4CE"
    stroke = "#107C10" if on else "#C19C00"
    tw = 11 * len(label) + 16
    parts.append(
        f'<rect x="{x - tw:.1f}" y="{y:.1f}" width="{tw:.1f}" height="20" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<text x="{x - tw/2:.1f}" y="{y + 14.5:.1f}" class="badge" text-anchor="middle" '
        f'fill="{stroke}">{esc(label)}</text>'
    )


def rbox(x, y, w, h, lines, stroke, fill, badge_label=None, badge_on=True):
    parts.append(
        f'<rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
        f'fill="{fill}" stroke="{stroke}"/>'
    )
    multilines(x + w / 2, y + h / 2 + 2, lines, line_h=17)
    if badge_label:
        badge(x + w - 8, y + 8, badge_label, on=badge_on)
    return (x, y, w, h)


def ctn(x, y, w, h, title, stroke):
    parts.append(
        f'<rect class="ctn" x="{x}" y="{y}" width="{w}" height="{h}" rx="12" stroke="{stroke}"/>'
    )
    # title chip
    tw = min(len(title) * 14 + 24, w - 40)
    parts.append(
        f'<rect x="{x + 18}" y="{y - 12}" width="{tw}" height="24" rx="6" fill="#fff" stroke="{stroke}" stroke-width="1.4"/>'
    )
    parts.append(
        f'<text x="{x + 18 + tw/2:.1f}" y="{y + 5}" class="ct" text-anchor="middle" fill="{stroke}">{esc(title)}</text>'
    )


def v_arrow(x, y1, y2):
    """Vertical arrow from bottom of upper box to top of lower box."""
    if y2 - y1 < 8:
        return
    # leave small gaps so arrowheads don't sit inside boxes
    a, b = y1 + 2, y2 - 4
    if b <= a:
        return
    parts.append(f'<path class="arr" d="M {x:.1f} {a:.1f} L {x:.1f} {b:.1f}"/>')


def h_arrow(x1, y, x2):
    if abs(x2 - x1) < 8:
        return
    a, b = (x1 + 2, x2 - 4) if x2 > x1 else (x1 - 2, x2 + 4)
    parts.append(f'<path class="arr" d="M {a:.1f} {y:.1f} L {b:.1f} {y:.1f}"/>')


def link(x1, y1, x2, y2):
    """Straight connector between two points with short end gaps."""
    dx, dy = x2 - x1, y2 - y1
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 10:
        return
    ux, uy = dx / dist, dy / dist
    a = (x1 + ux * 3, y1 + uy * 3)
    b = (x2 - ux * 6, y2 - uy * 6)
    parts.append(f'<path class="arr" d="M {a[0]:.1f} {a[1]:.1f} L {b[0]:.1f} {b[1]:.1f}"/>')


def diamond(cx, cy, w, h, lines, stroke=ORANGE[0], fill=ORANGE[1]):
    pts = f"{cx},{cy - h/2} {cx + w/2},{cy} {cx},{cy + h/2} {cx - w/2},{cy}"
    parts.append(
        f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    )
    multilines(cx, cy + 2, lines, line_h=15)
    return cx, cy, w, h


# ===== Title =====
text(W / 2, 42, "事件监控台 · PoC 架构（已实现能力）", cls="title")
text(W / 2, 70, "视频流 + 关注点 → 分析参数 → 抽帧/检测跟踪 → 语义事件与分窗 → 三路身份 → 关键帧与场景 → Prompt → 双助手", cls="sub")
text(W / 2, 92, "绿标=默认开启　琥珀=默认关闭（可开）　场景文字在关键帧之后　对话基于本次结果追问", cls="hint")

# legend
def leg(x, y, stroke, fill, label):
    parts.append(f'<rect x="{x}" y="{y}" width="22" height="14" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    text(x + 30, y + 12, label, cls="hint", anchor="start")

leg(220, 108, "#107C10", "#DFF6DD", "默认开启")
leg(360, 108, "#C19C00", "#FFF4CE", "默认关闭（可开）")
leg(560, 108, "#0C8599", "#C5F0F5", "输出 / 页面")
leg(720, 108, "#D13438", "#FDE7E9", "大模型助手")
leg(900, 108, "#5C2D91", "#E8DAEF", "Prompt")

CX = W / 2
y = 145

# ===== 入口：视频流 + 关注点 → 分析参数设置 =====
rbox(CX - 320, y, 640, 78,
     ["输入：视频流 + 关注点（可选）", "样片/上传视频；关注点写入后续分析提示"],
     BLUE[0], BLUE[1], "默认开启", True)

y_set = y + 78 + 32
v_arrow(CX, y + 78 + 2, y_set - 2)

rbox(CX - 360, y_set, 720, 72,
     ["分析参数设置（设置单次生效）"],
     BLUE[0], BLUE[1], "已实现", True)

# ===== 主轴 ①-④ =====
def spine(y0, lines, col, h=78, badge="默认开启", on=True):
    rbox(280, y0, 920, h, lines, col[0], col[1], badge, on)
    return y0 + h

y = y_set + 72 + 30
v_arrow(CX, y_set + 72 + 2, y - 2)
y = spine(y, ["固定帧率抽帧（默认约每秒 2 帧）"], ORANGE, h=64)
yb = y
y = yb + 28
v_arrow(CX, yb + 2, y - 2)
y = spine(y, ["逐帧本地 CV：YOLO 检测 + BoT-SORT+ReID / ByteTrack → 稳定 track_id"], GREEN, h=64)
yb = y
y = yb + 28
v_arrow(CX, yb + 2, y - 2)
y = spine(y, ["生成语义事件信号", "新人出现 / 人离开 / 人数变化 / 新物体 / 认出熟人 → 切段与关键帧"], GREEN, h=72)
yb = y
y = yb + 28
v_arrow(CX, yb + 2, y - 2)
y = spine(y, ["切成若干「事件时间段」", "连续安静结束本段；单段过长截断"], ORANGE, h=64)
yb = y

# ===== 人物身份（Phase4 风格：三路 gallery + 合并 + 加权投票）=====
y = yb + 42
ID_H = 700
ctn(60, y, 1360, ID_H, "人物身份（三路分库：人脸 / 人形 / 步态 Gallery → 合并 → 加权置信）", PURPLE[0])
v_arrow(CX, yb, y)

fx, bx, gx = 90, 520, 970
fw, bw, gw = 390, 400, 400
iy = y + 36
ih = 78

rbox(fx, iy, fw, ih,
     ["人脸 · ArcFace（每 track 最佳脸帧）", "可选 AdaFace"],
     PURPLE[0], PURPLE[1], "默认关闭", False)
rbox(bx, iy, bw, ih,
     ["人形 ReID · DIFFER（每 track 最佳身体帧）", "可选 OSNet / CLIP-ReID / SigLIP2"],
     PURPLE[0], PURPLE[1], "默认开启", True)
rbox(gx, iy, gw, ih,
     ["步态 · SkeletonGait++（时间段内序列）", "姿态+剪影；无脸/背身兜底"],
     PURPLE[0], PURPLE[1], "默认关闭", False)

# decision diamonds
dq_h = 58
dqy = iy + ih + 18 + dq_h / 2
v_arrow(fx + fw / 2, iy + ih, dqy - dq_h / 2)
v_arrow(bx + bw / 2, iy + ih, dqy - dq_h / 2)
diamond(fx + fw / 2, dqy, 156, dq_h, ["人脸质量", "角度+模糊"])
diamond(bx + bw / 2, dqy, 168, dq_h, ["相似度裁决", "余弦阈值"])

# branch boxes
bh = 78
bry = dqy + dq_h / 2 + 16
rbox(fx, bry, 185, bh, ["clear", "入库 / 满权"], PURPLE[0], PURPLE[1])
rbox(fx + 200, bry, 190, bh, ["marginal / poor", "降权；只查不建"], PURPLE[0], PURPLE[1])
link(fx + fw / 2 - 28, dqy + dq_h / 2, fx + 92, bry)
link(fx + fw / 2 + 28, dqy + dq_h / 2, fx + 295, bry)

rbox(bx, bry, 185, bh,
     ["hit ≥0.60", "new <0.40"],
     PURPLE[0], PURPLE[1])
rbox(bx + 205, bry, 185, bh,
     ["grey 灰区", "0.40~0.60"],
     PURPLE[0], PURPLE[1])
link(bx + bw / 2 - 28, dqy + dq_h / 2, bx + 92, bry)
link(bx + bw / 2 + 28, dqy + dq_h / 2, bx + 297, bry)

# enhance / stitch
eh = 60
ery = bry + bh + 14
rbox(fx + 200, ery, 190, eh, ["糊脸超分(可选)", "原图才可入库"], PURPLE[0], PURPLE[1])
v_arrow(fx + 295, bry + bh, ery)
rbox(bx + 205, ery, 185, eh, ["轨迹缝合", "灰区并入相近主体"], PURPLE[0], PURPLE[1])
v_arrow(bx + 297, bry + bh, ery)

# three galleries
gh = 68
ggy = ery + eh + 16
rbox(fx, ggy, fw, gh, ["人脸 Gallery", "clear 建档；糊脸防污染"], PURPLE[0], PURPLE[1])
rbox(bx, ggy, bw, gh, ["人形 Gallery", "multi-shot 登记/命中"], PURPLE[0], PURPLE[1])
rbox(gx, ggy, gw, gh, ["步态 Gallery", "序列 → gait subject"], PURPLE[0], PURPLE[1])
v_arrow(fx + 92, bry + bh, ggy)
v_arrow(fx + 295, ery + eh, ggy)
v_arrow(bx + 92, bry + bh, ggy)
v_arrow(bx + 297, ery + eh, ggy)
v_arrow(gx + gw / 2, iy + ih, ggy)

# merge + confidence
mh = 68
my = ggy + gh + 18
for cx in (fx + fw / 2, bx + bw / 2, gx + gw / 2):
    v_arrow(cx, ggy + gh, my)
rbox(120, my, 1240, mh,
     ["跨路合并 → 统一 subject_id", "任一路相同则合并；输出统一身份编号"],
     PURPLE[0], PURPLE[1], "已实现", True)

fh = 100
fy = my + mh + 14
v_arrow(CX, my + mh, fy)
rbox(120, fy, 1240, fh,
     ["身份置信加权 + 归一化（非简单多数票）",
      "有效权重：人脸 0.5×(0.3+0.7×质量分) · 人形 0.3 · 步态 0.2（配置值不变）",
      "confidence = Σ(强度×权重)/Σ权重  ← 人脸降权后，人形/步态占比自动升高；多路同一人 +0.15"],
     PURPLE[0], PURPLE[1], "已实现", True)

id_bottom = y + ID_H

# ===== 关键帧 =====
y = id_bottom + 28
v_arrow(CX, id_bottom, y)
y = spine(
    y,
    [
        "挑选关键帧",
        "①有事件的帧优先  ②每人轨迹代表帧  ③相邻过相似去重  ④限制每段喂大模型图数",
    ],
    ORANGE,
    h=78,
)
yb = y

# ===== 场景 AFTER keyframe =====
y = yb + 36
ctn(80, y, 1320, 150, "场景信息（只在关键帧上补齐；不参与「是不是同一个人」）", GREEN[0])
v_arrow(CX, yb, y)

sy = y + 32
sh = 96
rbox(110, sy, 380, sh,
     ["场景文字 · RapidOCR", "关键帧读字：时间戳/车牌/单号"],
     GREEN[0], GREEN[1], "默认关闭", False)
rbox(550, sy, 380, sh,
     ["物体 / 包裹", "非人轨迹，与人物时间对齐"],
     GREEN[0], GREEN[1], "默认关闭", False)
rbox(990, sy, 370, sh,
     ["位置与走向", "框 / 中心 / 方向"],
     GREEN[0], GREEN[1], "默认开启", True)
sc_bottom = y + 150

# ===== Prompt =====
y = sc_bottom + 28
for cx in (110 + 190, 550 + 190, 990 + 185):
    v_arrow(cx, sc_bottom, y)
y = spine(y, ["Prompt（结构化事实）", "谁 / 何时 / 做什么 / 物体 / 画面文字 → 喂给大模型助手"], TEAL, h=72)
yb = y

# ===== 双助手 =====
y = yb + 32
v_arrow(CX, yb, y)
ctn(80, y, 1320, 190, "两类大模型助手（共用分析结果与 Prompt）", RED[0])

ay = y + 34
ah = 132
rbox(110, ay, 600, ah,
     ["事件分析助手",
      "自动生成逐段事件与整段故事",
      "读关键帧 + Prompt"],
     RED[0], RED[1], "默认开启", True)
rbox(770, ay, 590, ah,
     ["对话问答助手",
      "追问当前视频细节",
      "绑定本次分析；优先查 Prompt，不默认重跑 CV"],
     RED[0], RED[1], "默认开启", True)
h_arrow(710, ay + ah / 2, 770)

parts.append("</svg>\n")
svg = "".join(parts)

root = Path(__file__).resolve().parents[1]
docs = root / "docs"
docs.mkdir(parents=True, exist_ok=True)
for name in ("poc-architecture.svg", "phase4-logic-flow.svg"):
    p = docs / name
    p.write_text(svg, encoding="utf-8")
    print("wrote", p, "bytes", p.stat().st_size)

# also keep a simple PNG via cairosvg or skip if unavailable
try:
    import cairosvg
    for name in ("poc-architecture.png", "phase4-logic-flow.png"):
        out = docs / name
        cairosvg.svg2png(url=str(docs / "poc-architecture.svg"), write_to=str(out), dpi=140)
        print("png", out)
except Exception as e:
    print("png skip:", type(e).__name__, e)
    # fallback: matplotlib render of svg is hard; use pillow + svglib if present
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        drawing = svg2rlg(str(docs / "poc-architecture.svg"))
        for name in ("poc-architecture.png", "phase4-logic-flow.png"):
            renderPM.drawToFile(drawing, str(docs / name), fmt="PNG", dpi=120)
            print("png via svglib", name)
    except Exception as e2:
        print("png fallback skip:", type(e2).__name__, e2)
