# -*- coding: utf-8 -*-
"""16:9 defense derivatives: continuous LTR half-chains (upper/lower).

Does NOT modify docs/poc-architecture.svg.
Overwrites:
  docs/defense/poc-architecture-16x9-overview.svg/.png  (page 20 upper half)
  docs/defense/poc-architecture-16x9-detail.svg/.png    (page 21 lower half)
"""
from __future__ import annotations

from pathlib import Path

W, H = 1920, 1080

INK = "#201F1E"
SUB = "#605E5C"
MUTED = "#8A8886"
MS_BLUE = "#0078D4"
MS_BLUE_SOFT = "#DEECF9"
MS_BLUE_PALE = "#EFF6FC"
TEAL = "#0C8599"
TEAL_SOFT = "#E0F7FA"
PURPLE = "#5C2D91"
PURPLE_SOFT = "#F3EAF8"
GREEN = "#107C10"
GREEN_SOFT = "#E9F7E9"
ORANGE = "#C19C00"
ORANGE_SOFT = "#FFF4CE"
WHITE = "#FFFFFF"
GRAY_SOFT = "#F3F2F1"


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


class Svg:
    def __init__(self):
        self.parts: list[str] = []
        self.parts.append(
            f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <marker id="arr" markerWidth="10" markerHeight="10" refX="9" refY="4" orient="auto">
      <path d="M0,0 L9,4 L0,8 Z" fill="{MS_BLUE}"/>
    </marker>
    <style>
      .t1 {{ font: 700 36px "Segoe UI","Microsoft YaHei",sans-serif; fill: {INK}; }}
      .t2 {{ font: 400 17px "Segoe UI","Microsoft YaHei",sans-serif; fill: {SUB}; }}
      .t3 {{ font: 600 14px "Segoe UI","Microsoft YaHei",sans-serif; fill: {MUTED}; }}
      .bt {{ font: 700 20px "Segoe UI","Microsoft YaHei",sans-serif; fill: {INK}; }}
      .bd {{ font: 400 15px "Segoe UI","Microsoft YaHei",sans-serif; fill: {SUB}; }}
      .bs {{ font: 400 14px "Segoe UI","Microsoft YaHei",sans-serif; fill: {MUTED}; }}
      .badge {{ font: 700 13px "Segoe UI","Microsoft YaHei",sans-serif; }}
      .flow {{ font: 600 15px "Segoe UI","Microsoft YaHei",sans-serif; fill: {MS_BLUE}; }}
      .note {{ font: 400 14px "Segoe UI","Microsoft YaHei",sans-serif; fill: {SUB}; }}
    </style>
  </defs>
  <rect width="100%" height="100%" fill="{WHITE}"/>
'''
        )

    def text(self, x, y, s, cls="bd", anchor="start"):
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{esc(s)}</text>'
        )

    def badge(self, x, y, label, kind="on"):
        if kind == "on":
            fill, stroke = GREEN_SOFT, GREEN
        elif kind == "off":
            fill, stroke = ORANGE_SOFT, ORANGE
        else:
            fill, stroke = MS_BLUE_SOFT, MS_BLUE
        tw = max(56, 13 * len(label) + 20)
        self.parts.append(
            f'<rect x="{x - tw:.1f}" y="{y:.1f}" width="{tw:.1f}" height="24" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>'
        )
        self.parts.append(
            f'<text x="{x - tw/2:.1f}" y="{y + 16.5:.1f}" class="badge" text-anchor="middle" fill="{stroke}">{esc(label)}</text>'
        )

    def card(self, x, y, w, h, title, line, stroke, fill, badge=None, badge_kind="on"):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2.2"/>'
        )
        # title + one detail line, vertically centered-ish
        self.text(x + w / 2, y + h / 2 - 8, title, "bt", "middle")
        self.text(x + w / 2, y + h / 2 + 18, line, "bd", "middle")
        if badge:
            self.badge(x + w - 12, y + 10, badge, badge_kind)
        return x, y, w, h

    def subcard(self, x, y, w, h, title, line, stroke, fill, badge=None, badge_kind="off"):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>'
        )
        self.text(x + w / 2, y + h / 2 - 6, title, "bt", "middle")
        self.text(x + w / 2, y + h / 2 + 16, line, "bs", "middle")
        if badge:
            self.badge(x + w - 10, y + 8, badge, badge_kind)

    def h_arrow(self, x1, y, x2):
        if x2 <= x1 + 4:
            return
        self.parts.append(
            f'<path d="M {x1:.1f} {y:.1f} L {x2:.1f} {y:.1f}" stroke="{MS_BLUE}" stroke-width="2.4" fill="none" marker-end="url(#arr)"/>'
        )

    def finish(self) -> str:
        self.parts.append("</svg>\n")
        return "".join(self.parts)


def page_upper() -> str:
    """Page 20: video -> person features. Pure LTR."""
    g = Svg()
    g.text(56, 52, "最终定版 PoC（上）：视频到人物特征", "t1")
    g.text(56, 82, "第 6 部分 · 前半链路（仅左→右）　终点：轨迹级人物特征　不进入融合 / OCR / 大模型", "t2")
    g.text(56, 108, "阅读方向：全程从左到右一条主链", "flow")

    # 6 equal-ish step cards on one row
    # margins: left 48, right 48, gaps 22 between cards, arrows ~28 wide inside gap
    n = 6
    margin_l, margin_r = 40, 40
    gap = 30  # includes arrow space
    usable = W - margin_l - margin_r - gap * (n - 1)
    cw = usable / n
    ch = 280
    y = 200
    cy = y + ch / 2

    steps = [
        (["视频输入 / 本次参数", "样片或上传；开关仅本次生效"], MS_BLUE, MS_BLUE_SOFT, "入口", "on"),
        (["固定帧率抽帧", "默认约 2 fps 有序帧"], ORANGE, ORANGE_SOFT, "默开", "on"),
        (["检测与多目标跟踪", "YOLO + ByteTrack / BoT-SORT"], GREEN, GREEN_SOFT, "默开", "on"),
        (["切事件时间段", "安静结束；过长截断"], ORANGE, ORANGE_SOFT, "默开", "on"),
        (["轨迹门控与选帧", "过短/过糊跳过；身体/脸各选最佳帧"], PURPLE, PURPLE_SOFT, "默开", "on"),
        (["人形特征提取", "轨迹最佳身体帧提特征"], PURPLE, PURPLE_SOFT, "默开", "on"),
    ]

    xs = []
    for i, (lines, st, fl, badge, bk) in enumerate(steps):
        x = margin_l + i * (cw + gap)
        g.card(x, y, cw, ch, lines[0], lines[1], st, fl, badge, bk)
        xs.append(x)
        if i < n - 1:
            g.h_arrow(x + cw + 2, cy, x + cw + gap - 2)

    # optional face/gait under the last feature stage only (supplement below, same column)
    last_x = xs[-1]
    g.subcard(last_x, y + ch + 22, cw, 110, "人脸 / 步态（可选）", "脸另选最佳帧；步态用序列 · 默认关", PURPLE, "#FBF7FD", "默关", "off")

    # continuity footer
    g.text(56, 990, "本页终点：人物特征已提取（尚未融合、未进场景旁路、未调用大模型）", "note")
    g.text(56, 1020, "下一页从「多线索身份融合」继续 →　　派生自 docs/poc-architecture.svg · 未改原图", "t3")
    g.text(W - 56, 1020, "（上）1 / 2", "t3", "end")
    return g.finish()


def page_lower() -> str:
    """Page 21: fusion -> AI output. Pure LTR, continues from page 20."""
    g = Svg()
    g.text(56, 52, "最终定版 PoC（下）：证据融合到 AI 输出", "t1")
    g.text(56, 82, "第 6 部分 · 后半链路（仅左→右）　承接上一页终点：在人物特征之后继续", "t2")
    g.text(56, 108, "阅读方向：全程从左到右一条主链　|　场景证据在关键帧之后，OCR 不参与「是否同一人」", "flow")

    n = 7
    margin_l, margin_r = 36, 36
    gap = 24
    usable = W - margin_l - margin_r - gap * (n - 1)
    cw = usable / n
    ch = 250
    y = 190
    cy = y + ch / 2

    steps = [
        (["多线索身份融合", "轨迹级聚合；三路可合并同一人"], PURPLE, PURPLE_SOFT, "默开", "on"),
        (["本次身份库匹配", "Gallery 登记/命中；统一编号"], PURPLE, PURPLE_SOFT, "默开", "on"),
        (["关键帧筛选", "事件帧优先；限量喂大模型"], ORANGE, ORANGE_SOFT, "默开", "on"),
        (["场景证据（关键帧后）", "OCR / 物体 / 位置走向"], GREEN, GREEN_SOFT, "旁路", "on"),
        (["结构化结果 + CSV", "谁/何时/做什么/物体/文字"], TEAL, TEAL_SOFT, "默开", "on"),
        (["Azure OpenAI 理解", "关键帧 + 事实表 → 事件叙述"], "#A4262C", "#FDE7E9", "默开", "on"),
        (["最终事件 JSON/报告", "逐段事件 · 整段总结 · 告警"], TEAL, TEAL_SOFT, "默开", "on"),
    ]

    xs = []
    for i, (lines, st, fl, badge, bk) in enumerate(steps):
        x = margin_l + i * (cw + gap)
        g.card(x, y, cw, ch, lines[0], lines[1], st, fl, badge, bk)
        xs.append((x, cw))
        if i < n - 1:
            g.h_arrow(x + cw + 2, cy, x + cw + gap - 2)

    # supplements under specific cards only (same column, no direction change)
    sy = y + ch + 20
    sh = 100
    x0, w0 = xs[0]
    g.subcard(x0, sy, w0, sh, "融合口径", "人脸/人形/步态线索聚合", PURPLE, "#FBF7FD")
    x1, w1 = xs[1]
    g.subcard(x1, sy, w1, sh, "Session-only", "本次有效；结束清空", PURPLE, "#FBF7FD")
    x3, w3 = xs[3]
    g.subcard(x3, sy, w3, sh, "OCR 边界", "不参与是否同一人的融合", GREEN, "#F6FBF6", "默关", "off")
    x4, w4 = xs[4]
    g.subcard(x4, sy, w4, sh, "事实表用途", "扁平字段供事件理解引用", TEAL, "#F3FBFC")

    g.text(56, 990, "本页起点承接上一页「人形/身份特征」→ 融合 → 身份库 → 关键帧 → 场景旁路 → CSV → LLM → 事件报告", "note")
    g.text(56, 1020, "已去掉：监控台 UI 分叉、对话入口、运行约束灰区、部署模块　　派生自 docs/poc-architecture.svg · 未改原图", "t3")
    g.text(W - 56, 1020, "（下）2 / 2", "t3", "end")
    return g.finish()


def write_png(svg_path: Path, png_path: Path) -> None:
    import os
    import subprocess
    import uuid

    edge_candidates = [
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
    ]
    edge = next((p for p in edge_candidates if Path(p).exists()), None)
    if not edge:
        print("no browser for png", png_path.name)
        return

    html_path = Path(os.environ.get("TEMP", ".")) / f"render-{uuid.uuid4().hex}.html"
    user = Path(os.environ.get("TEMP", ".")) / f"edge-prof-{uuid.uuid4().hex}"
    user.mkdir(parents=True, exist_ok=True)
    svg_uri = svg_path.resolve().as_uri()
    html_path.write_text(
        f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<style>html,body{{margin:0;background:#fff}}img{{width:1920px;height:1080px;display:block}}</style>
</head><body><img src='{svg_uri}' width='1920' height='1080'/></body></html>""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            edge,
            "--headless=new",
            "--disable-gpu",
            "--allow-file-access-from-files",
            f"--user-data-dir={user}",
            "--window-size=1920,1080",
            f"--screenshot={png_path}",
            html_path.resolve().as_uri(),
        ],
        check=False,
        capture_output=True,
    )
    try:
        html_path.unlink(missing_ok=True)
    except Exception:
        pass
    print("png", png_path, "exists" if png_path.exists() else "MISSING", png_path.stat().st_size if png_path.exists() else 0)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    original = root / "docs" / "poc-architecture.svg"
    assert original.exists(), original
    out = root / "docs" / "defense"
    out.mkdir(parents=True, exist_ok=True)

    overview = out / "poc-architecture-16x9-overview.svg"
    detail = out / "poc-architecture-16x9-detail.svg"
    overview.write_text(page_upper(), encoding="utf-8")
    detail.write_text(page_lower(), encoding="utf-8")
    print("wrote", overview)
    print("wrote", detail)
    print("original_bytes", original.stat().st_size)

    write_png(overview, out / "poc-architecture-16x9-overview.png")
    write_png(detail, out / "poc-architecture-16x9-detail.png")


if __name__ == "__main__":
    main()
