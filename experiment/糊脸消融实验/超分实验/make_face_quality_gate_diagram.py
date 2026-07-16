"""生成人脸质量判定与身份匹配的自上而下学术架构图。"""
from __future__ import annotations

import html
import json
from pathlib import Path

import cairosvg


HERE = Path(__file__).resolve().parent
SVG = HERE / "当前人脸质量门控分类规则.svg"
PNG = HERE / "当前人脸质量门控分类规则.png"
EXCALIDRAW = HERE / "当前人脸质量门控分类规则.excalidraw"

W, H = 1700, 2520

COLOR = {
    "blue": ("#0078D4", "#EAF4FC"),
    "purple": ("#5C2D91", "#F4ECFA"),
    "orange": ("#F7630C", "#FFF4CE"),
    "green": ("#107C10", "#EAF6EA"),
    "teal": ("#0C8599", "#E6F7F9"),
    "red": ("#D13438", "#FDE7E9"),
    "gray": ("#605E5C", "#F3F2F1"),
}


def esc(value: str) -> str:
    return html.escape(value)


def rect(x, y, w, h, stroke, fill, sw=2, rx=12, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
    )


def txt(x, y, values, size=16, weight=400, anchor="middle", gap=24, color="#111111"):
    spans = []
    for index, value in enumerate(values):
        spans.append(f'<tspan x="{x}" dy="{0 if index == 0 else gap}">{esc(value)}</tspan>')
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{color}">{"".join(spans)}</text>'
    )


def box(x, y, w, h, family, title, details=(), title_size=21, detail_size=15, dash=None):
    stroke, fill = COLOR[family]
    result = [rect(x, y, w, h, stroke, fill, 2.2, 14, dash)]
    result.append(txt(x + w / 2, y + 34, [title], title_size, 700))
    if details:
        result.append(txt(x + w / 2, y + 67, list(details), detail_size, 400, "middle", 23))
    return "".join(result)


def container(x, y, w, h, family, title, subtitle):
    stroke, _ = COLOR[family]
    return (
        rect(x, y, w, h, stroke, "transparent", 2.8, 20)
        + txt(x + w / 2, y + 36, [title], 22, 700)
        + txt(x + w / 2, y + 64, [subtitle], 14, 400, "middle", color="#444444")
    )


def diamond(cx, cy, w, h, family, lines):
    stroke, fill = COLOR[family]
    points = f"{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}"
    return (
        f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="2.2"/>'
        + txt(cx, cy - 8, lines, 16, 700, "middle", 21)
    )


def connector(points, color="#605E5C", dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    coords = " ".join(f"{x},{y}" for x, y in points)
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        f'stroke-width="2.2"{dash_attr}/>'
    )


def arrow(points, label=None, label_xy=None, color="#605E5C", dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    coords = " ".join(f"{x},{y}" for x, y in points)
    result = (
        f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.4" '
        f'marker-end="url(#arrow)"{dash_attr}/>'
    )
    if label:
        x, y = label_xy or points[len(points) // 2]
        result += txt(x, y, [label], 14, 600, "middle", color="#333333")
    return result


def curve_arrow(x1, y1, c1x, c1y, c2x, c2y, x2, y2, color="#605E5C", dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="M {x1} {y1} C {c1x} {c1y}, {c2x} {c2y}, {x2} {y2}" '
        f'fill="none" stroke="{color}" stroke-width="2.4" '
        f'marker-end="url(#arrow)"{dash_attr}/>'
    )


def junction(x, y, family="gray"):
    stroke, fill = COLOR[family]
    return f'<circle cx="{x}" cy="{y}" r="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'


svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    """
    <defs>
      <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3"
              orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#605E5C"/>
      </marker>
    </defs>
    <rect width="100%" height="100%" fill="#ffffff"/>
    <style>text { font-family: "Microsoft YaHei", "Segoe UI", sans-serif; }</style>
    """,
    txt(W / 2, 48, ["人脸质量判定与身份匹配目标架构"], 32, 700),
    txt(
        W / 2,
        82,
        ["质量先于身份：完成质量分类后，才决定是否调用产品ArcFace / AdaFace"],
        16,
        400,
        "middle",
        color="#444444",
    ),
]

# Main detection spine
svg.append(box(610, 125, 480, 100, "blue", "① 人物轨迹采样帧", ["输入：RGB人物图像"]))
svg.append(arrow([(850, 225), (850, 270)]))
svg.append(box(555, 270, 590, 145, "blue", "② SCRFD人脸检测", [
    "输出：人脸框bbox、5点关键点、检测分数det_score",
    "当前确认阈值：det_score ≥ 0.5",
]))
svg.append(arrow([(850, 415), (850, 455)]))
svg.append(diamond(850, 515, 220, 120, "blue", ["检测到", "人脸？"]))

# None branch
svg.append(arrow([(960, 515), (1230, 515)], "否", (1090, 494), COLOR["red"][0]))
svg.append(box(1230, 440, 390, 155, "gray", "none：人脸路线不可用", [
    "不运行CR-FIQA和产品ArcFace",
    "人物轨迹仍然保留",
    "转人形、步态和事件理解",
]))

# Alignment
svg.append(arrow([(850, 575), (850, 620)], "是", (875, 602)))
svg.append(box(620, 620, 460, 125, "blue", "③ 五点关键点对齐", [
    "输出同一张112×112对齐人脸",
    "供质量分支与身份分支共用",
]))

# Parallel quality block
svg.append(container(70, 800, 1560, 390, "purple", "④ 并行质量评估", "相同输入、独立计算；两条分支在质量合成前汇合"))
svg.append(box(140, 875, 620, 245, "purple", "CR-FIQA质量分支", [
    "112×112对齐人脸 → 独立iResNet50 CNN",
    "→ 512维内部质量embedding → 质量回归头",
    "→ 连续身份识别可用性分数q",
    "质量embedding在产品中丢弃，不进入人脸库",
]))
svg.append(box(940, 875, 620, 245, "orange", "显式规则分支", [
    "检测可信度：det_score    尺寸：最小边 / 面积",
    "姿态：yaw / pitch        清晰度：拉普拉斯方差",
    "输出：small_face / blur / extreme_pose等子标签",
]))
svg.append(curve_arrow(850, 745, 800, 815, 560, 825, 450, 875))
svg.append(curve_arrow(850, 745, 900, 815, 1140, 825, 1250, 875))

# Quality merge
svg.append(box(500, 1210, 700, 190, "green", "⑤ 质量合成与动作生成", [
    "规则硬失败或q < 0.3 → poor",
    "规则全部清晰且q ≥ 0.6 → clear；其余 → marginal",
    "同时输出：defects、can_enroll、can_match、can_superres",
]))
svg.append(curve_arrow(450, 1120, 500, 1165, 610, 1180, 700, 1210))
svg.append(curve_arrow(1250, 1120, 1200, 1165, 1090, 1180, 1000, 1210))

# Category block
svg.append(container(70, 1450, 1560, 390, "green", "⑥ 检测到人脸后的分类结果", "none已在SCRFD失败分支结束；三类结果统一进入动作字段"))
category_centers = [330, 850, 1370]
svg.append(box(125, 1550, 410, 190, "green", "clear", [
    "can_enroll=true",
    "can_match=true",
    "can_superres=false",
]))
svg.append(box(645, 1550, 410, 190, "orange", "marginal", [
    "can_match=true，按q降低身份权重",
    "有模糊 / 小脸标签时可超分",
]))
svg.append(box(1165, 1550, 410, 190, "red", "poor + defects", [
    "模糊 / 小脸：可低权重匹配或超分",
    "极端姿态：can_match=false",
]))
svg.append(curve_arrow(850, 1400, 730, 1450, 470, 1485, 330, 1550))
svg.append(arrow([(850, 1400), (850, 1550)]))
svg.append(curve_arrow(850, 1400, 970, 1450, 1230, 1485, 1370, 1550))
svg.append(box(260, 1770, 1180, 80, "green", "统一动作字段", [
    "分类结果 + defects → can_enroll / can_match / can_superres / match_weight",
], 19, 15))
for center in category_centers:
    svg.append(arrow([(center, 1740), (center, 1770)]))

# Match gate
svg.append(arrow([(850, 1850), (850, 1890)]))
svg.append(diamond(850, 1950, 240, 125, "green", ["can_match", "= true？"]))
svg.append(arrow([(730, 1950), (410, 1950)], "否", (570, 1928), COLOR["red"][0]))
svg.append(box(85, 1880, 325, 150, "gray", "停止人脸身份匹配", [
    "不调用产品ArcFace / AdaFace",
    "保留人物与事件信息",
    "转人形 / 步态路线",
]))

# Super-resolution decision
svg.append(arrow([(850, 2012), (850, 2050)], "是", (875, 2035)))
svg.append(diamond(850, 2110, 270, 120, "purple", ["can_superres", "= true？"]))
svg.append(arrow([(715, 2110), (470, 2110)], "否：使用原图", (595, 2088)))
svg.append(box(155, 2050, 315, 125, "teal", "原始对齐人脸", [
    "不执行超分",
]))
svg.append(arrow([(985, 2110), (1170, 2110)], "是", (1080, 2088), COLOR["purple"][0]))
svg.append(box(1170, 2025, 390, 175, "purple", "可选超分与复评", [
    "GFPGAN / CodeFormer",
    "修复后重新计算CR-FIQA",
    "检查身份embedding漂移",
], dash="8 5"))

# Identity merge and matching
svg.append(connector([(312, 2175), (312, 2240), (850, 2240)]))
svg.append(connector([(1365, 2200), (1365, 2240), (850, 2240)]))
svg.append(junction(850, 2240, "teal"))
svg.append(arrow([(850, 2240), (850, 2280)]))
svg.append(box(520, 2280, 660, 130, "teal", "⑦ 产品ArcFace / AdaFace身份分支", [
    "输出512维身份embedding → 与人脸库计算余弦相似度 → 固定阈值决定是否复用身份",
]))

svg.append(txt(
    W / 2,
    2475,
    ["CR-FIQA的512维内部特征只服务质量回归；产品ArcFace / AdaFace的512维特征只服务身份匹配，两者不可混用。"],
    15,
    600,
    "middle",
    color="#444444",
))

svg.append("</svg>")
svg_text = "".join(svg)
SVG.write_text(svg_text, encoding="utf-8")
cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), write_to=str(PNG), output_width=W, output_height=H)


# Editable Excalidraw source with centered text and vertical order.
elements = []


def ex_box(eid, x, y, w, h, family, value, font=17):
    tid = f"{eid}_text"
    stroke, fill = COLOR[family]
    elements.append(
        {
            "type": "rectangle",
            "id": eid,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "strokeColor": stroke,
            "backgroundColor": fill,
            "fillStyle": "solid",
            "strokeWidth": 2,
            "roundness": {"type": 3},
            "boundElements": [{"type": "text", "id": tid}],
        }
    )
    elements.append(
        {
            "type": "text",
            "id": tid,
            "x": x + 10,
            "y": y + 10,
            "width": w - 20,
            "height": max(h - 20, font * 2.5 * (value.count("\n") + 1)),
            "text": value,
            "fontSize": font,
            "fontFamily": 2,
            "strokeColor": "#000000",
            "textAlign": "center",
            "verticalAlign": "middle",
            "containerId": eid,
        }
    )


def ex_arrow(eid, x1, y1, x2, y2):
    elements.append(
        {
            "type": "arrow",
            "id": eid,
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "strokeColor": "#605E5C",
            "strokeWidth": 2,
            "points": [[0, 0], [x2 - x1, y2 - y1]],
        }
    )


elements.append(
    {
        "type": "text",
        "id": "title",
        "x": 250,
        "y": 20,
        "width": 1200,
        "height": 80,
        "text": "人脸质量判定与身份匹配目标架构",
        "fontSize": 30,
        "fontFamily": 2,
        "strokeColor": "#000000",
        "textAlign": "center",
        "verticalAlign": "top",
    }
)
ex_box("input", 600, 120, 500, 100, "blue", "人物轨迹采样帧")
ex_box("scrfd", 550, 280, 600, 130, "blue", "SCRFD检测\nbbox / 5点关键点 / det_score")
ex_box("align", 620, 500, 460, 110, "blue", "5点对齐\n112×112人脸")
ex_box("fiqa", 140, 760, 620, 210, "purple", "CR-FIQA质量分支\niResNet50 → 质量embedding → q")
ex_box("rules", 940, 760, 620, 210, "orange", "规则分支\ndet / size / yaw / pitch / Laplacian / defects")
ex_box("fusion", 500, 1080, 700, 170, "green", "质量合成\nclear / marginal / poor\n输出can_enroll / can_match / can_superres")
ex_box("categories", 250, 1390, 1200, 180, "green", "分类结果\nclear：建档+匹配\nmarginal：降权匹配\npoor：按defects决定")
ex_box("actions", 300, 1630, 1100, 100, "green", "统一动作字段\ncan_enroll / can_match / can_superres / match_weight")
ex_box("gate", 600, 1810, 500, 130, "green", "can_match=true？")
ex_box("stop", 80, 1810, 360, 130, "gray", "否：停止人脸匹配\n转人形/步态")
ex_box("sr", 970, 2020, 460, 160, "purple", "can_superres=true\n→ 超分并重新FIQA")
ex_box("arc", 520, 2260, 660, 140, "teal", "产品ArcFace / AdaFace\n→ 身份embedding → 查询人脸库")

ex_arrow("a1", 850, 220, 850, 280)
ex_arrow("a2", 850, 410, 850, 500)
ex_arrow("a3", 850, 610, 450, 760)
ex_arrow("a4", 850, 610, 1250, 760)
ex_arrow("a5", 450, 970, 700, 1080)
ex_arrow("a6", 1250, 970, 1000, 1080)
ex_arrow("a7", 850, 1250, 850, 1390)
ex_arrow("a8", 850, 1570, 850, 1630)
ex_arrow("a9", 850, 1730, 850, 1810)
ex_arrow("a10", 600, 1875, 440, 1875)
ex_arrow("a11", 850, 1940, 850, 2260)
ex_arrow("a12", 1100, 1940, 1200, 2020)
ex_arrow("a13", 1200, 2180, 950, 2260)

EXCALIDRAW.write_text(
    json.dumps(
        {
            "type": "excalidraw",
            "version": 2,
            "source": "copilot",
            "elements": elements,
            "appState": {"viewBackgroundColor": "#ffffff"},
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(SVG)
print(PNG)
print(EXCALIDRAW)
