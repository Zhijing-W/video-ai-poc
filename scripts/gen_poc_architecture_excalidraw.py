# -*- coding: utf-8 -*-
"""用 Excalidraw JSON 生成可读的 PoC 架构图（中文、流程对齐代码）。"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

# FluentUI-ish palette
C = {
    "blue_s": "#0078D4", "blue_f": "#CFE4FA",
    "orange_s": "#F7630C", "orange_f": "#FFF4CE",
    "green_s": "#107C10", "green_f": "#DFF6DD",
    "purple_s": "#5C2D91", "purple_f": "#E8DAEF",
    "teal_s": "#0C8599", "teal_f": "#C5F0F5",
    "red_s": "#D13438", "red_f": "#FDE7E9",
    "amber_s": "#C19C00", "amber_f": "#FFF4CE",
    "gray_s": "#605E5C", "gray_f": "#F3F2F1",
    "ink": "#000000",
}

elements: list[dict] = []


def nid(prefix: str = "e") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def add(el: dict) -> str:
    if "id" not in el:
        el["id"] = nid(el.get("type", "e")[:3])
    # defaults
    el.setdefault("x", 0)
    el.setdefault("y", 0)
    el.setdefault("angle", 0)
    el.setdefault("strokeColor", C["ink"])
    el.setdefault("backgroundColor", "transparent")
    el.setdefault("fillStyle", "solid")
    el.setdefault("strokeWidth", 1.5)
    el.setdefault("strokeStyle", "solid")
    el.setdefault("roughness", 0)
    el.setdefault("opacity", 100)
    el.setdefault("groupIds", [])
    el.setdefault("frameId", None)
    el.setdefault("roundness", {"type": 3} if el["type"] in {"rectangle", "diamond"} else None)
    el.setdefault("seed", abs(hash(el["id"])) % 2_000_000_000)
    el.setdefault("versionNonce", abs(hash(el["id"] + "v")) % 2_000_000_000)
    el.setdefault("isDeleted", False)
    el.setdefault("boundElements", None)
    el.setdefault("updated", 1)
    el.setdefault("link", None)
    el.setdefault("locked", False)
    elements.append(el)
    return el["id"]


def text(x, y, s, *, size=16, w=None, h=None, align="left", bold=False):
    lines = s.count("\n") + 1
    # CJK ~ size px wide; use generous metrics
    est_w = w or max(len(line) for line in s.split("\n")) * size * 0.95
    est_h = h or int(size * 2.2 * lines)
    tid = add({
        "type": "text",
        "x": x, "y": y,
        "width": est_w, "height": est_h,
        "text": s,
        "originalText": s,
        "fontSize": size,
        "fontFamily": 2,  # Helvetica - better CJK fallback often
        "textAlign": align,
        "verticalAlign": "top",
        "strokeColor": C["ink"],
        "backgroundColor": "transparent",
        "containerId": None,
        "autoResize": True,
        "lineHeight": 1.25,
    })
    return tid


def box(x, y, w, h, title, body, *, stroke, fill, bid=None, badge=None):
    """Leaf box with title+body as bound-ish centered multi-line text."""
    lines = [title]
    if body:
        lines.extend(body.split("\n") if isinstance(body, str) else body)
    if badge:
        lines.append(f"【{badge}】")
    content = "\n".join(lines)
    box_id = bid or nid("box")
    # text first to measure-ish, then box
    t_id = nid("txt")
    # estimate text size
    line_count = content.count("\n") + 1
    t_w = w - 24
    t_h = max(h - 16, int(15 * 2.15 * line_count))
    add({
        "type": "rectangle",
        "id": box_id,
        "x": x, "y": y, "width": w, "height": h,
        "strokeColor": stroke,
        "backgroundColor": fill,
        "fillStyle": "solid",
        "strokeWidth": 2,
        "boundElements": [{"id": t_id, "type": "text"}],
    })
    add({
        "type": "text",
        "id": t_id,
        "x": x + 12, "y": y + 8,
        "width": t_w, "height": t_h,
        "text": content,
        "originalText": content,
        "fontSize": 15,
        "fontFamily": 2,
        "textAlign": "center",
        "verticalAlign": "middle",
        "containerId": box_id,
        "strokeColor": C["ink"],
        "backgroundColor": "transparent",
        "autoResize": False,
        "lineHeight": 1.25,
    })
    return box_id


def container(x, y, w, h, title, stroke):
    cid = nid("ctn")
    tid = nid("ctt")
    add({
        "type": "rectangle",
        "id": cid,
        "x": x, "y": y, "width": w, "height": h,
        "strokeColor": stroke,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2.5,
        "strokeStyle": "solid",
        "boundElements": None,
    })
    # title label floating on top border
    text(x + 16, y - 12, title, size=16, w=min(w - 32, 900), h=36)
    return cid


def arrow_v(x, y1, y2, *, color="#666666"):
    """Vertical arrow from y1 to y2 at x."""
    h = y2 - y1
    if h <= 0:
        return None
    return add({
        "type": "arrow",
        "x": x, "y": y1,
        "width": 0, "height": h,
        "strokeColor": color,
        "strokeWidth": 2,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "points": [[0, 0], [0, h]],
        "lastArrowhead": "arrow",
        "startArrowhead": None,
        "elbowed": False,
    })


def arrow_between(x1, y1, x2, y2, *, color="#666666"):
    """Simple two-point arrow."""
    return add({
        "type": "arrow",
        "x": x1, "y": y1,
        "width": x2 - x1, "height": y2 - y1,
        "strokeColor": color,
        "strokeWidth": 2,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "points": [[0, 0], [x2 - x1, y2 - y1]],
        "lastArrowhead": "arrow",
        "startArrowhead": None,
        "elbowed": False,
    })


# ---------------- layout constants ----------------
# canvas content roughly 0..1500 x
CX = 760
LEFT = 80
GAP = 28
BW = 360  # branch box width
BH = 110
MW = 720  # main spine width
MH = 88

# Title
text(CX - 280, 20, "事件监控台 · PoC 架构（已实现）", size=28, w=700, h=50, align="center")
text(
    CX - 420, 60,
    "样片/上传 → 同步分析 → 抽帧跟踪 → 切事件段 → 认人融合 → 选关键帧 → 场景文字/物体 → 大模型 → 页面展示",
    size=15, w=1000, h=40, align="center",
)
text(
    CX - 380, 92,
    "绿标=默认开启　琥珀=已实现默认关闭　场景文字在关键帧之后，不参与「是不是同一个人」",
    size=14, w=900, h=36, align="center",
)

y = 150

# ---- 入口三块 + 串行说明 ----
b_in = box(LEFT, y, 300, 100,
           "选择样片或上传视频",
           "监控台首页入口\n支持常见视频格式",
           stroke=C["blue_s"], fill=C["blue_f"], badge="默认开启")
b_focus = box(LEFT + 340, y, 280, 100,
              "分析关注点（可选）",
              "一句话说明重点\n写入大模型提示",
              stroke=C["blue_s"], fill=C["blue_f"], badge="可选")
b_dry = box(LEFT + 660, y, 280, 100,
            "仅本地分析",
            "先不调用大模型\n需要时再补理解",
            stroke=C["blue_s"], fill=C["blue_f"], badge="默认关闭")
b_lock = box(LEFT + 980, y, 260, 100,
             "同时只跑一路",
             "避免算力争抢",
             stroke=C["teal_s"], fill=C["teal_f"], badge="已实现")
y1_bottom = y + 100

y = y1_bottom + GAP
b_set = box(LEFT + 80, y, 1200, 100,
            "本次分析参数（只对这一次生效）",
            "开关：人形 / 人脸 / 步态 / 场景文字 / 物体　｜　可选模型：跟踪器、人脸、清晰化、人形特征\n"
            "其它：抽帧频率、每段最多关键帧、单段最长约30秒、同人合并松紧",
            stroke=C["blue_s"], fill=C["blue_f"], badge="已实现")
# arrows into settings - short vertical from each top box center
for bx, bw in [(LEFT, 300), (LEFT + 340, 280), (LEFT + 660, 280), (LEFT + 980, 260)]:
    arrow_v(bx + bw / 2, y1_bottom + 2, y - 2, color="#888888")

y2_bottom = y + 100
y = y2_bottom + GAP
b_api = box(LEFT + 80, y, 1200, 95,
            "同步分析接口",
            "一次请求跑完整段视频并返回结果　｜　可先本地分析再补大模型\n"
            "另有：样片列表、可用模型列表、健康检查",
            stroke=C["teal_s"], fill=C["teal_f"], badge="默认开启")
arrow_v(CX, y2_bottom + 2, y - 2)

# ---- 主轴 ①-④ ----
def spine(y0, title, body, stroke, fill, badge="默认开启", h=MH):
    bid = box(CX - MW / 2, y0, MW, h, title, body, stroke=stroke, fill=fill, badge=badge)
    return bid, y0 + h


y = y + 95 + GAP
arrow_v(CX, y - GAP + 2, y - 2)
b_fps, yb = spine(y, "① 按固定帧率抽帧（默认约每秒2帧）",
                  "整段视频变成有序图片，供跟踪与认人（智能抽帧未接入主流程）",
                  C["orange_s"], C["orange_f"])
y = yb + GAP
arrow_v(CX, yb + 2, y - 2)
b_mot, yb = spine(y, "② 行人检测 + 多目标跟踪",
                  "跨帧保持同一跟踪编号；默认带外观辅助，遮挡交叉更稳",
                  C["green_s"], C["green_f"])
y = yb + GAP
arrow_v(CX, yb + 2, y - 2)
b_evt, yb = spine(y, "③ 生成语义事件信号",
                  "新人出现、人离开、人数变化、新物体、认出熟人 → 用来切段和挑关键帧",
                  C["green_s"], C["green_f"])
y = yb + GAP
arrow_v(CX, yb + 2, y - 2)
b_win, yb = spine(
    y,
    "④ 切成若干「事件时间段」",
    "连续一段时间没人/没活动 → 结束本段；单段太长（约30秒）→ 强制截断\n全程安静时整段仍作为一段，方便描述空场景",
    C["orange_s"], C["orange_f"], h=100,
)

# ---- ⑤ 身份识别 container ----
y = yb + GAP + 20
ct_h = 320
container(LEFT, y, 1400, ct_h,
          "⑤ 人物身份识别（在跟踪轨迹上做；人脸/人形各选最佳帧；步态用时间段内序列）",
          C["purple_s"])
arrow_v(CX, yb + 2, y + 8)

iy = y + 40
b_body = box(LEFT + 40, iy, 420, 120,
             "人形特征识别",
             "每条轨迹选「身体最清晰」一帧\n提特征并查询/登记身份库\n默认 OSNet 路线，可换模型",
             stroke=C["purple_s"], fill=C["purple_f"], badge="默认开启")
b_face = box(LEFT + 500, iy, 420, 120,
             "人脸识别",
             "另选「脸最合适」候选帧（不复用身体帧）\n默认 ArcFace，可切 AdaFace\n可选清晰化；侧脸几何线索可开",
             stroke=C["purple_s"], fill=C["purple_f"], badge="默认关闭")
b_gait = box(LEFT + 960, iy, 400, 120,
             "步态识别",
             "不是单帧，而是时间段内走路序列\n姿态+剪影提特征\n无脸/背身时补充身份",
             stroke=C["purple_s"], fill=C["purple_f"], badge="默认关闭")

iy2 = iy + 120 + 24
b_gal = box(LEFT + 120, iy2, 520, 110,
            "本次分析身份库（仅本次有效）",
            "同一次分析里记住见过的人\n分析结束清空，不跨视频长期存",
            stroke=C["purple_s"], fill=C["purple_f"], badge="默认开启")
b_fuse = box(LEFT + 720, iy2, 560, 110,
             "身份汇总到「同一个人」",
             "轨迹断裂可缝合；人脸/人形/步态对上可合并\n输出统一身份编号 + 可信度",
             stroke=C["purple_s"], fill=C["purple_f"], badge="默认开启")

# clean vertical arrows identity row1 -> row2
arrow_v(LEFT + 40 + 210, iy + 120 + 2, iy2 - 2, color=C["purple_s"])
arrow_v(LEFT + 500 + 210, iy + 120 + 2, iy2 - 2, color=C["purple_s"])
arrow_v(LEFT + 960 + 200, iy + 120 + 2, iy2 - 2, color=C["purple_s"])

id_bottom = y + ct_h

# ---- ⑥ 关键帧 ----
y = id_bottom + GAP
arrow_v(CX, id_bottom + 2, y - 2)
b_kf, yb = spine(
    y,
    "⑥ 为每个事件时间段挑选关键帧（准备给大模型看的图）",
    "优先有事件的帧、每人代表帧，去掉过于相似的相邻帧\n目的：少给大模型几张图——不是丢掉前面的跟踪/认人结果",
    C["orange_s"], C["orange_f"], h=100,
)

# ---- ⑦ 场景信息 AFTER keyframes ----
y = yb + GAP + 18
ct2_h = 180
container(LEFT, y, 1400, ct2_h,
          "⑦ 场景信息（只在关键帧上补齐；不参与「是不是同一个人」）",
          C["green_s"])
arrow_v(CX, yb + 2, y + 8)

sy = y + 42
b_ocr = box(LEFT + 40, sy, 420, 115,
            "场景文字识别",
            "只对选出的关键帧读字\n时间戳、车牌、运单号等\n场景说明，不代表某个人",
            stroke=C["green_s"], fill=C["green_f"], badge="默认关闭")
b_obj = box(LEFT + 500, sy, 420, 115,
            "物体/包裹说明",
            "本段非人物体轨迹\n行李、车辆等变化\n与人物时间对齐",
            stroke=C["green_s"], fill=C["green_f"], badge="默认关闭")
b_sp = box(LEFT + 960, sy, 400, 115,
           "人物位置与走动方向",
           "关键帧上的框、中心、走向\n页面可叠加显示",
           stroke=C["green_s"], fill=C["green_f"], badge="默认开启")
sc_bottom = y + ct2_h

# ---- ⑧ 打包 ----
y = sc_bottom + GAP
# three short verticals into pack from scene boxes
for bx, bw in [(LEFT + 40, 420), (LEFT + 500, 420), (LEFT + 960, 400)]:
    arrow_v(bx + bw / 2, sc_bottom - 20, y - 2, color=C["green_s"])
# also from identity fuse conceptually already via spine - main arrow from center of scene container top already had arrow into container; pack gets from scene bottoms

b_pack, yb = spine(
    y,
    "⑧ 打成结构化结果包（每一事件时间段一份）",
    "时间范围、关键帧、已融合人物身份、位置说明、场景文字、物体说明 → 交给大模型",
    C["teal_s"], C["teal_f"], h=95,
)

# ---- ⑨ ⑩ 大模型 ----
y = yb + GAP
arrow_v(CX, yb + 2, y - 2)
b_llm = box(CX - MW / 2, y, 340, 110,
            "⑨ 大模型逐段理解",
            "看关键帧 + 读身份/场景文字\n输出本段事件与告警等级",
            stroke=C["red_s"], fill=C["red_f"], badge="默认开启")
b_all = box(CX - MW / 2 + 380, y, 340, 110,
            "⑩ 整段故事串联",
            "多段收成完整叙述\n总览、要点、通知文案",
            stroke=C["red_s"], fill=C["red_f"], badge="默认开启")
# horizontal link between llm and overall
arrow_between(CX - MW / 2 + 340 + 2, y + 55, CX - MW / 2 + 380 - 2, y + 55, color=C["red_s"])
llm_bottom = y + 110

# ---- 11 展示 ----
y = llm_bottom + GAP + 16
container(LEFT, y, 1400, 160, "11. 事件监控台展示", C["teal_s"])
arrow_v(CX - 180, llm_bottom + 2, y + 6, color=C["red_s"])
arrow_v(CX + 180, llm_bottom + 2, y + 6, color=C["red_s"])

uy = y + 40
box(LEFT + 40, uy, 300, 100, "原始结果下载", "完整 JSON\n含配置与各阶段耗时",
    stroke=C["teal_s"], fill=C["teal_f"], badge="默认开启")
box(LEFT + 380, uy, 300, 100, "人物身份卡片", "头像、各路命中\n融合可信度",
    stroke=C["teal_s"], fill=C["teal_f"], badge="默认开启")
box(LEFT + 720, uy, 300, 100, "事件时间线", "关键帧预览\n可看位置框",
    stroke=C["teal_s"], fill=C["teal_f"], badge="默认开启")
box(LEFT + 1060, uy, 300, 100, "进度与工具", "耗时、配置摘要\n补做大模型",
    stroke=C["teal_s"], fill=C["teal_f"], badge="默认开启")

doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "poc-architecture-generator",
    "elements": elements,
    "appState": {
        "viewBackgroundColor": "#ffffff",
        "gridSize": 20,
    },
    "files": {},
}

root = Path(__file__).resolve().parents[1]
out_ex = root / "docs" / "poc-architecture.excalidraw"
out_ex.parent.mkdir(parents=True, exist_ok=True)
out_ex.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out_ex)
print("elements", len(elements))
