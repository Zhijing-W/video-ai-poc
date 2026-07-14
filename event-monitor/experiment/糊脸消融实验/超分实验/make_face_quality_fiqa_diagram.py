"""生成人脸质量分流与FIQA接入图：SVG、PNG及Excalidraw可编辑源。"""
from __future__ import annotations

import html
import json
from pathlib import Path

import cairosvg


HERE = Path(__file__).resolve().parent
SVG_PATH = HERE / "人脸质量分流与FIQA接入.svg"
PNG_PATH = HERE / "人脸质量分流与FIQA接入.png"
EXCALIDRAW_PATH = HERE / "人脸质量分流与FIQA接入.excalidraw"

W, H = 1900, 1450

COLORS = {
    "blue": ("#0078D4", "#CFE4FA"),
    "orange": ("#F7630C", "#FFF4CE"),
    "purple": ("#5C2D91", "#E8DAEF"),
    "green": ("#107C10", "#DFF6DD"),
    "teal": ("#0C8599", "#C5F0F5"),
    "red": ("#D13438", "#FDE7E9"),
    "gray": ("#605E5C", "#F3F2F1"),
}


def esc(value: str) -> str:
    return html.escape(value)


def rect(x, y, w, h, stroke, fill="transparent", sw=2, rx=16, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
    )


def lines(x, y, values, size=16, weight=400, anchor="start", gap=25, color="#111111"):
    spans = []
    for index, value in enumerate(values):
        dy = 0 if index == 0 else gap
        spans.append(f'<tspan x="{x}" dy="{dy}">{esc(value)}</tspan>')
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{color}">{"".join(spans)}</text>'
    )


def leaf(x, y, w, h, family, title, details, title_size=19, detail_size=14, dash=None):
    stroke, fill = COLORS[family]
    out = [rect(x, y, w, h, stroke, fill, 2, 14, dash)]
    out.append(lines(x + w / 2, y + 31, [title], title_size, 700, "middle"))
    if details:
        out.append(lines(x + 18, y + 60, details, detail_size, 400, "start", 22))
    return "".join(out)


def container(x, y, w, h, family, title, subtitle=None):
    stroke, _ = COLORS[family]
    out = [rect(x, y, w, h, stroke, "transparent", 3, 20)]
    out.append(lines(x + 20, y + 34, [title], 22, 700))
    if subtitle:
        out.append(lines(x + 20, y + 60, [subtitle], 13, 400, color="#444444"))
    return "".join(out)


def arrow(x1, y1, x2, y2, label=None, color="#605E5C", dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    path = (
        f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{color}" '
        f'stroke-width="2.5" marker-end="url(#arrow)"{dash_attr}/>'
    )
    if not label:
        return path
    tx, ty = (x1 + x2) / 2, (y1 + y2) / 2 - 8
    return path + lines(tx, ty, [label], 13, 600, "middle", color="#333333")


def poly_arrow(points, label=None, label_xy=None, color="#605E5C", dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    coords = " ".join(f"{x},{y}" for x, y in points)
    path = (
        f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" '
        f'marker-end="url(#arrow)"{dash_attr}/>'
    )
    if not label:
        return path
    tx, ty = label_xy or points[len(points) // 2]
    return path + lines(tx, ty, [label], 13, 600, "middle", color="#333333")


svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    """
    <defs>
      <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3"
              orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#605E5C"/>
      </marker>
      <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.13"/>
      </filter>
    </defs>
    <rect width="100%" height="100%" fill="#FFFFFF"/>
    <style>text { font-family: "Microsoft YaHei", "Segoe UI", sans-serif; }</style>
    """,
    lines(W / 2, 48, ["人脸质量分流、超分动作与FIQA接入位置"], 31, 700, "middle"),
    lines(
        W / 2,
        80,
        ["唯一质量评估源 → 多维问题标签 → 动作决策；none只关闭人脸路线，不丢弃人物轨迹"],
        15,
        400,
        "middle",
        color="#444444",
    ),
]

# Stage 1
svg.append(container(40, 120, 340, 500, "blue", "① 人物帧与人脸检测", "当前已实现"))
svg.append(leaf(70, 190, 280, 95, "blue", "人物轨迹采样帧", ["原始监控人物crop / 全帧中的人物区域"]))
svg.append(arrow(210, 285, 210, 320))
svg.append(leaf(70, 320, 280, 115, "blue", "SCRFD人脸检测", ["输出bbox、5点关键点、det_score", "没有检测到脸 → face_status=unavailable"]))
svg.append(arrow(210, 435, 210, 475, "检测成功"))
svg.append(leaf(70, 475, 280, 110, "blue", "5点对齐人脸", ["对齐为112×112", "供ArcFace / AdaFace / FIQA / 超分使用"]))

# No-face branch
svg.append(leaf(40, 680, 340, 170, "gray", "未检测到脸（none）", [
    "不建人脸参考特征",
    "不做人脸身份匹配",
    "当前人脸裁剪超分无法运行",
    "人物轨迹仍保留",
]))
svg.append(poly_arrow([(70, 380), (25, 380), (25, 735), (40, 735)], color=COLORS["red"][0]))

# Stage 2
svg.append(container(420, 120, 650, 650, "orange", "② 唯一的人脸质量评估", "总体等级 + 可同时存在的问题标签"))
svg.append(leaf(450, 190, 285, 180, "orange", "当前规则信号", [
    "人脸尺寸：bbox宽高、面积",
    "清晰度：拉普拉斯方差",
    "检测可信度：det_score",
    "姿态：5点近似yaw / pitch",
    "正脸程度：frontalness",
]))
svg.append(leaf(755, 190, 285, 180, "purple", "公开FIQA（计划接入）", [
    "CR-FIQA / SER-FIQ / OFIQ",
    "输入：检测并对齐后的人脸crop",
    "输出：身份识别可用性总分",
    "当前_deep_fiqa_score仍返回None",
], dash="9 6"))
svg.append(arrow(592, 370, 592, 415))
svg.append(arrow(897, 370, 897, 415))
svg.append(leaf(485, 415, 520, 155, "orange", "统一质量合成 assess_quality()", [
    "总体等级：clear / marginal / poor",
    "问题标签：blur / small_face / extreme_yaw / extreme_pitch / occlusion ...",
    "动作字段：can_enroll / can_match / can_superres",
]))
svg.append(arrow(745, 570, 745, 615))
svg.append(leaf(485, 615, 520, 120, "orange", "保留原始质量证据", [
    "原图分桶始终不因超分改变",
    "用于实验分析、门控解释和坏案例追踪",
]))
svg.append(poly_arrow([(350, 530), (400, 530), (400, 280), (450, 280)], "检测结果", (400, 260)))

# Stage 3
svg.append(container(1110, 120, 750, 650, "green", "③ 根据同一质量结果选择动作", "不是第二套质量评分"))
rows = [
    ("clear", "建档：是", "匹配：是", "超分：否", "green"),
    ("marginal + blur", "建档：谨慎/否", "匹配：降权", "超分：可尝试", "orange"),
    ("poor + blur/small_face", "建档：否", "匹配：低权重", "超分：可尝试", "orange"),
    ("poor + extreme_pose", "建档：否", "匹配：谨慎/拒绝", "超分：否", "red"),
    ("none", "人脸不建档", "人脸不匹配", "人脸裁剪超分不可用", "gray"),
]
start_y = 190
for idx, (quality, enroll, match, sr, family) in enumerate(rows):
    y = start_y + idx * 103
    svg.append(leaf(1140, y, 690, 84, family, quality, [f"{enroll}    {match}    {sr}"], 18, 14))
svg.append(arrow(1005, 492, 1110, 445, "质量结果与标签"))

# Stage 4
svg.append(container(420, 810, 1440, 540, "teal", "④ 身份特征与多模态兜底", "超分只是一种可选处理动作"))
svg.append(leaf(450, 885, 350, 145, "teal", "原图识别路径", [
    "clear或不需要超分",
    "对齐脸 → ArcFace / AdaFace",
    "得到512维人脸特征",
]))
svg.append(leaf(835, 855, 420, 205, "purple", "超分识别路径", [
    "can_superres=true",
    "对齐脸 → GFPGAN / CodeFormer",
    "超分后重新计算FIQA与清晰度",
    "检查修复前后身份特征是否漂移",
    "通过后再提ArcFace / AdaFace特征",
], dash="9 6"))
svg.append(leaf(1290, 885, 520, 145, "teal", "人脸参考库查询", [
    "人脸特征只与人脸特征比较",
    "达到固定身份匹配阈值 → 候选统一主体ID",
    "低于阈值 → 人脸路线拒绝复用",
]))
svg.append(poly_arrow([(800, 955), (815, 955), (815, 820), (1545, 820), (1545, 885)]))
svg.append(arrow(1255, 955, 1290, 955))
svg.append(arrow(1465, 1030, 1465, 1095))
svg.append(leaf(1080, 1095, 730, 160, "green", "统一主体层融合", [
    "人脸候选 + 人形ReID候选 + 步态候选",
    "各模态先在自己的向量库内比较，再映射到统一subject_id",
    "多路一致才加成；高置信冲突进入灰区并禁止写库",
]))
svg.append(leaf(450, 1100, 570, 150, "gray", "人脸none或人脸路线拒绝", [
    "继续人物跟踪、人形ReID、步态识别",
    "身份仍未知时输出“未识别人物”",
    "事件理解仍描述其进入、停留、拿取物品等行为",
]))
svg.append(poly_arrow([(210, 850), (210, 1175), (450, 1175)], "人脸不可用", (330, 1158)))
svg.append(arrow(1020, 1175, 1080, 1175, "其他模态"))

# FIQA load location note
svg.append(leaf(40, 900, 340, 375, "purple", "FIQA加载位置（建议）", [
    "配置入口：",
    "app/core/config.py",
    "FACE_FIQA_BACKEND",
    "",
    "模型加载：",
    "app/face.py新增_ensure_fiqa()",
    "进程内懒加载一次，不逐帧重复加载",
    "",
    "推理调用：",
    "_deep_fiqa_score(aligned_face)",
    "由assess_quality()消费",
    "",
    "当前状态：接口已留，模型未实现",
], dash="9 6"))
svg.append(lines(210, 1305, ["虚线紫框 = 计划接入 / 尚未实现"], 13, 600, "middle", color="#5C2D91"))

svg.append(lines(W / 2, 1405, [
    "关键原则：FIQA负责“身份识别可用性总分”；规则分项负责解释原因；动作字段负责决定建档、匹配与超分。"
], 15, 600, "middle", color="#333333"))
svg.append("</svg>")

SVG_PATH.write_text("".join(svg), encoding="utf-8")
cairosvg.svg2png(bytestring="".join(svg).encode("utf-8"), write_to=str(PNG_PATH), output_width=W, output_height=H)


# A simplified editable Excalidraw source with the same major stages.
elements = []


def ex_rect(eid, x, y, w, h, stroke, fill, text, font=17):
    text_id = f"{eid}_text"
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
            "boundElements": [{"type": "text", "id": text_id}],
        }
    )
    line_count = text.count("\n") + 1
    elements.append(
        {
            "type": "text",
            "id": text_id,
            "x": x + 12,
            "y": y + 12,
            "width": w - 24,
            "height": max(h - 24, font * 2.5 * line_count),
            "text": text,
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
        "x": 410,
        "y": 30,
        "width": 1000,
        "height": 80,
        "text": "人脸质量分流、超分动作与FIQA接入位置",
        "fontSize": 30,
        "fontFamily": 2,
        "strokeColor": "#000000",
        "textAlign": "center",
        "verticalAlign": "top",
    }
)
ex_rect("detect", 60, 160, 300, 160, COLORS["blue"][0], COLORS["blue"][1], "SCRFD检测 + 5点对齐\n无脸：仅关闭人脸路线")
ex_rect("rules", 430, 130, 320, 210, COLORS["orange"][0], COLORS["orange"][1], "当前规则信号\n尺寸 / 拉普拉斯 / det_score\n5点yaw、pitch、正脸度")
ex_rect("fiqa", 790, 130, 330, 210, COLORS["purple"][0], COLORS["purple"][1], "公开FIQA（计划）\nCR-FIQA / SER-FIQ / OFIQ\n当前接口仍返回None")
ex_rect("quality", 570, 410, 430, 180, COLORS["orange"][0], COLORS["orange"][1], "唯一质量评估 assess_quality\n总体等级 + 多个问题标签\ncan_enroll / can_match / can_superres")
ex_rect("actions", 1190, 130, 560, 460, COLORS["green"][0], COLORS["green"][1], "动作分流\nclear：建档+匹配，不超分\nmarginal+blur：降权匹配，可超分\npoor+blur/small：不建档，可超分\npoor+extreme_pose：拒绝超分\nnone：人脸不可用，转人形/步态")
ex_rect("original", 480, 720, 340, 150, COLORS["teal"][0], COLORS["teal"][1], "原图路径\n对齐脸 → ArcFace/AdaFace")
ex_rect("sr", 870, 690, 390, 210, COLORS["purple"][0], COLORS["purple"][1], "超分路径\nGFPGAN/CodeFormer\n超分后重新FIQA\n检查身份漂移后再提特征")
ex_rect("match", 1320, 720, 390, 150, COLORS["teal"][0], COLORS["teal"][1], "人脸库查询\n达到固定阈值才复用统一主体")
ex_rect("fallback", 480, 1010, 530, 170, COLORS["gray"][0], COLORS["gray"][1], "人脸none/拒绝\n人物轨迹保留\n人形ReID、步态和事件理解继续")
ex_rect("fusion", 1120, 1010, 590, 170, COLORS["green"][0], COLORS["green"][1], "统一主体层融合\n三种特征分别查询各自向量库\n映射到统一subject_id后融合")
ex_rect("load", 60, 720, 320, 300, COLORS["purple"][0], COLORS["purple"][1], "FIQA加载位置\nconfig.py: FACE_FIQA_BACKEND\nface.py: _ensure_fiqa()\n_deep_fiqa_score()\nassess_quality()消费")

ex_arrow("a1", 360, 240, 430, 240)
ex_arrow("a2", 750, 240, 790, 240)
ex_arrow("a3", 700, 340, 750, 410)
ex_arrow("a4", 1000, 500, 1190, 400)
ex_arrow("a5", 1380, 590, 650, 720)
ex_arrow("a6", 1460, 590, 1060, 690)
ex_arrow("a7", 1260, 795, 1320, 795)
ex_arrow("a8", 1515, 870, 1415, 1010)
ex_arrow("a9", 1010, 1095, 1120, 1095)

EXCALIDRAW_PATH.write_text(
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

print(SVG_PATH)
print(PNG_PATH)
print(EXCALIDRAW_PATH)
