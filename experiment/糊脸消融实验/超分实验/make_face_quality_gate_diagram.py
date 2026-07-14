"""生成当前人脸质量门控分类规则图，而非处理流程图。"""
from __future__ import annotations

import html
import json
from pathlib import Path

import cairosvg


HERE = Path(__file__).resolve().parent
SVG = HERE / "当前人脸质量门控分类规则.svg"
PNG = HERE / "当前人脸质量门控分类规则.png"
EXCALIDRAW = HERE / "当前人脸质量门控分类规则.excalidraw"

W, H = 1900, 1360

COLORS = {
    "blue": ("#0078D4", "#CFE4FA"),
    "orange": ("#F7630C", "#FFF4CE"),
    "purple": ("#5C2D91", "#E8DAEF"),
    "green": ("#107C10", "#DFF6DD"),
    "red": ("#D13438", "#FDE7E9"),
    "gray": ("#605E5C", "#F3F2F1"),
    "teal": ("#0C8599", "#C5F0F5"),
}


def esc(value: str) -> str:
    return html.escape(value)


def rect(x, y, w, h, stroke, fill="transparent", sw=2, rx=16, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
    )


def text(x, y, values, size=16, weight=400, anchor="start", gap=25, color="#111111"):
    spans = []
    for index, value in enumerate(values):
        spans.append(f'<tspan x="{x}" dy="{0 if index == 0 else gap}">{esc(value)}</tspan>')
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{color}">{"".join(spans)}</text>'
    )


def card(x, y, w, h, family, title, rows, dash=None, title_size=19, row_size=14):
    stroke, fill = COLORS[family]
    out = [rect(x, y, w, h, stroke, fill, 2, 14, dash)]
    out.append(text(x + w / 2, y + 31, [title], title_size, 700, "middle"))
    if rows:
        out.append(text(x + 15, y + 62, rows, row_size, 400, "start", 23))
    return "".join(out)


def section(x, y, w, h, family, title, subtitle):
    stroke, _ = COLORS[family]
    return (
        rect(x, y, w, h, stroke, "transparent", 3, 20)
        + text(x + 22, y + 35, [title], 22, 700)
        + text(x + 22, y + 61, [subtitle], 13, 400, color="#444444")
    )


parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    """
    <rect width="100%" height="100%" fill="#FFFFFF"/>
    <style>text { font-family: "Microsoft YaHei", "Segoe UI", sans-serif; }</style>
    """,
    text(W / 2, 48, ["当前人脸质量门控：输入信号、阈值、总体等级与子标签"], 31, 700, "middle"),
    text(
        W / 2,
        82,
        ["依据 event-monitor/app/face.py 与 app/core/config.py 当前实现；虚线紫框表示已预留但尚未生效"],
        14,
        400,
        "middle",
        color="#444444",
    ),
]

# Signals
parts.append(section(35, 115, 1830, 455, "blue", "① 质量判定输入维度", "同一张检测到的人脸同时计算；每个维度独立产生状态或问题标签"))

xs = [60, 355, 650, 945, 1240, 1535]
width = 270
height = 360
parts.append(card(xs[0], 185, width, height, "blue", "人脸检测可信度", [
    "没有检测到脸",
    "→ 总体等级 none",
    "",
    "det_score < 0.5",
    "→ blur_bad = true",
    "→ poor候选",
    "",
    "det_score ≥ 0.5",
    "→ 检测可信",
]))
parts.append(card(xs[1], 185, width, height, "orange", "人脸尺寸", [
    "最小边 < 28像素",
    "→ size_bad = true",
    "→ poor",
    "→ 子标签 too_small",
    "",
    "最小边 ≥ 28像素",
    "→ 尺寸通过",
    "",
    "面积还参与quality_score",
]))
parts.append(card(xs[2], 185, width, height, "orange", "水平转头 yaw", [
    "|yaw| ≤ 25°",
    "→ 角度清晰",
    "",
    "25° < |yaw| < 80°",
    "→ 非清晰但未硬失败",
    "→ 可能进入 marginal",
    "",
    "|yaw| ≥ 80°",
    "→ poor / yaw_extreme",
]))
parts.append(card(xs[3], 185, width, height, "orange", "抬头低头 pitch", [
    "|pitch| ≤ 20°",
    "→ 角度清晰",
    "",
    "-35° < pitch < -20°",
    "或 20° < pitch < 50°",
    "→ 可能进入 marginal",
    "",
    "pitch ≤ -35° → pitch_down",
    "pitch ≥ 50° → pitch_up",
]))
parts.append(card(xs[4], 185, width, height, "teal", "拉普拉斯清晰度", [
    "blur_var < 15",
    "→ blur_bad = true",
    "→ poor / too_blurry",
    "",
    "15 ≤ blur_var < 60",
    "→ 不算硬失败",
    "→ 但不满足清晰条件",
    "",
    "blur_var ≥ 60",
    "→ 清晰度通过",
]))
parts.append(card(xs[5], 185, width, height, "purple", "公开FIQA（未生效）", [
    "配置：FACE_FIQA_BACKEND",
    "当前默认 off",
    "_deep_fiqa_score返回None",
    "",
    "计划阈值：",
    "FIQA < 0.3 → poor信号",
    "0.3～0.6 → 非清晰",
    "FIQA ≥ 0.6 → 清晰",
    "",
    "应在5点对齐脸上推理",
], dash="9 6"))

# Combination rule
parts.append(section(35, 600, 1830, 220, "orange", "② 总体等级组合规则", "总体等级是单值；具体问题应保留为可同时存在的子标签"))
parts.append(card(70, 670, 510, 115, "red", "poor：任一硬失败", [
    "yaw_bad OR pitch_bad OR blur_bad OR size_bad",
], title_size=18))
parts.append(card(695, 670, 510, 115, "green", "clear：全部清晰", [
    "angle_clear AND blur_clear AND NOT size_bad",
], title_size=18))
parts.append(card(1320, 670, 510, 115, "orange", "marginal：其余情况", [
    "没有硬失败，但至少一个维度未达到clear",
], title_size=18))

# Categories and labels
parts.append(section(35, 850, 1830, 360, "green", "③ 最终四类输出与子标签", "当前category只有一个值；建议defects保留所有同时存在的问题"))
category_x = [65, 505, 945, 1385]
category_w = 410
parts.append(card(category_x[0], 920, category_w, 245, "green", "clear", [
    "检测到脸",
    "尺寸 ≥ 28像素",
    "|yaw| ≤ 25°",
    "|pitch| ≤ 20°",
    "blur_var ≥ 60",
    "若FIQA启用：FIQA ≥ 0.6",
    "",
    "当前动作：可建档、可匹配",
]))
parts.append(card(category_x[1], 920, category_w, 245, "orange", "marginal", [
    "没有触发任何poor硬条件",
    "但角度或清晰度未达到clear",
    "",
    "典型示例：",
    "yaw=40°，其他条件正常",
    "blur_var=35，其他条件正常",
    "",
    "当前动作：匹配时降权",
]))
parts.append(card(category_x[2], 920, category_w, 245, "red", "poor + 子标签", [
    "任意硬条件失败即进入poor",
    "",
    "建议子标签可同时存在：",
    "too_small",
    "yaw_extreme",
    "pitch_down / pitch_up",
    "too_blurry / low_detection",
    "fiqa_low（接入后）",
]))
parts.append(card(category_x[3], 920, category_w, 245, "gray", "none", [
    "人脸检测结果为空",
    "",
    "不代表人物不存在",
    "只表示人脸路线不可用",
    "",
    "人物轨迹、人形ReID、步态",
    "和事件理解继续运行",
]))

# Notes
parts.append(card(35, 1240, 890, 85, "gray", "当前实现限制", [
    "reason只按优先级保留一个：too_small → yaw → pitch → blur，会隐藏同时存在的问题。",
], title_size=16, row_size=13))
parts.append(card(975, 1240, 890, 85, "teal", "连续quality_score的用途", [
    "det_score × 正脸度 × 尺寸项 × 清晰度项；用于最佳脸选择和融合降权，不直接替代category。",
], title_size=16, row_size=13))
parts.append("</svg>")

svg_text = "".join(parts)
SVG.write_text(svg_text, encoding="utf-8")
cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), write_to=str(PNG), output_width=W, output_height=H)


# Simplified editable source
elements = []


def ex_box(eid, x, y, w, h, stroke, fill, value, font=16):
    tid = f"{eid}_text"
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
    count = value.count("\n") + 1
    elements.append(
        {
            "type": "text",
            "id": tid,
            "x": x + 12,
            "y": y + 12,
            "width": w - 24,
            "height": max(h - 24, font * 2.5 * count),
            "text": value,
            "fontSize": font,
            "fontFamily": 2,
            "strokeColor": "#000000",
            "textAlign": "center",
            "verticalAlign": "middle",
            "containerId": eid,
        }
    )


elements.append(
    {
        "type": "text",
        "id": "title",
        "x": 300,
        "y": 30,
        "width": 1200,
        "height": 80,
        "text": "当前人脸质量门控：信号、阈值、总体等级与子标签",
        "fontSize": 30,
        "fontFamily": 2,
        "strokeColor": "#000000",
        "textAlign": "center",
        "verticalAlign": "top",
    }
)

signal_texts = [
    ("det", "检测可信度\n无脸→none\ndet<0.5→poor信号", "blue"),
    ("size", "尺寸\n最小边<28→poor\ntoo_small", "orange"),
    ("yaw", "yaw\n≤25°清晰\n25°～80°中间\n≥80° poor", "orange"),
    ("pitch", "pitch\n≤20°清晰\n低头≤-35° poor\n抬头≥50° poor", "orange"),
    ("lap", "拉普拉斯\n<15 poor\n15～60中间\n≥60清晰", "teal"),
    ("fiqa", "FIQA（计划）\n<0.3 poor\n0.3～0.6中间\n≥0.6清晰", "purple"),
]
for idx, (eid, value, family) in enumerate(signal_texts):
    stroke, fill = COLORS[family]
    ex_box(eid, 50 + idx * 295, 150, 260, 250, stroke, fill, value)

ex_box("poor", 120, 500, 480, 120, COLORS["red"][0], COLORS["red"][1], "poor\n任一硬失败：yaw / pitch / blur / size")
ex_box("clear", 710, 500, 480, 120, COLORS["green"][0], COLORS["green"][1], "clear\n角度清晰 + 清晰度清晰 + 尺寸通过")
ex_box("marginal", 1300, 500, 480, 120, COLORS["orange"][0], COLORS["orange"][1], "marginal\n没有硬失败，但至少一项未达到clear")

ex_box("c1", 70, 760, 390, 260, COLORS["green"][0], COLORS["green"][1], "clear\n可建档、可匹配\n不需要超分")
ex_box("c2", 510, 760, 390, 260, COLORS["orange"][0], COLORS["orange"][1], "marginal\n无硬失败但非全清晰\n匹配时降权")
ex_box("c3", 950, 760, 390, 260, COLORS["red"][0], COLORS["red"][1], "poor + defects\n小脸 / 极端姿态\n模糊 / 低检测分\n标签允许同时存在")
ex_box("c4", 1390, 760, 390, 260, COLORS["gray"][0], COLORS["gray"][1], "none\n没有检测到脸\n只关闭人脸路线\n其他模态继续")

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
