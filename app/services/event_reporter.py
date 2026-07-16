"""身份感知的多帧事件理解（Phase 4 · Step 23 / 3.4）—— 本阶段灵魂。

定位：客户对齐的核心。把**一个事件窗的关键帧（多帧图）** + **结构化身份上下文** 一起喂多模态
大模型，让它做**跨帧事件理解**——输出"谁、何时、做了什么、是否异常"的描述性叙述。

与客户一致的关键约定：**身份由外部（传统 CV：人脸/人形ReID/步态+库比对）给定，模型看图但不
重新认人**；模型的职责是"基于身份 + 多帧画面，理解并整合成事件"，而非逐帧"画面里有个人"。

本模块只负责"**理解一个窗**"（LLM 调用这一段），是干净可复用的核心；**流式开/关窗的编排**
（什么时候攒够一个事件窗、何时冲刷）在上层（demo / 编排）做，不在这里。

复用：`openai_client.get_client`（Azure OpenAI 客户端）+
`identity.identity_context.format_identity_grounding`（身份上下文文本）。模型名可配置（`EVENT_LLM_DEPLOYMENT`，
默认回退主部署），以后指向 gpt-4.1/更强只改配置。
"""
from __future__ import annotations

import sys
import time

from openai import RateLimitError

from ..core.config import settings
from ..openai_client import get_client, parse_json
from ..utils.image_utils import image_to_data_uri
from ..identity.identity_context import format_identity_grounding

EVENT_SYSTEM = (
    "你是监控视频的事件理解助手。下面给你一段监控视频里**按时间顺序的若干关键帧**，以及画面中"
    "人物的**身份信息**（由外部传统 CV 已识别好）。\n"
    "请把两件事分清：\n"
    "1）【谁 = 身份】已由外部给定，你**不要重新做人脸/人形识别去猜这是谁**；同一 track 或同一"
    "『主体#/库内身份』在多帧中就是同一个人，直接采信。\n"
    "2）【做了什么 = 事件】你**必须仔细观察每一帧画面**，看懂画面里实际发生了什么——人物的动作、"
    "姿态、移动方向、与物体/他人的交互、场景与物体变化等。这部分**完全依赖你对图像的视觉理解**，"
    "不能只凭身份文字臆测、更不能编造画面里没有的情节。\n"
    "请把『谁（来自身份）』和『在做什么（来自你看图）』结合起来，叙述这段时间的跨帧事件，"
    "只根据可见信息，不臆造画面之外的内容。\n"
    "若另外提供了【画面文字（OCR）】，那是画面里出现的**场景级文字**（如时间戳、车牌、包裹单号），"
    "可用来补全事件的**时间/物件**线索；但它**不代表任何人的身份**，不要据此推断『谁』。\n"
    "若另外提供了【画面中的物体】，那是 YOLO 检出的**场景级物体**（包裹/行李/车辆等）及其轨迹，"
    "可用来理解放下、取走、搬运、到达、离开等事件；**若疑似快递/包裹，请结合画面识别其品牌或 logo"
    "（如 Amazon / UPS / FedEx）**。物体同样**不代表人物身份**。"
)


def _frame_to_data_uri(image) -> str:
    """把一帧统一成 data URI：已是 data URI 直接用；是路径则读盘转 data URI。"""
    if isinstance(image, str) and image.startswith("data:"):
        return image
    return image_to_data_uri(image)  # 文件路径


def _create_with_retry(client, **kwargs):
    """调用 chat.completions.create，撞 429 限流时按 Retry-After / 指数退避重试。

    低配额的 Azure OpenAI 部署常因"单次请求预留 token 超过每分钟配额"直接 429。这里不一撞就
    挂：读响应里的 Retry-After（没有则指数退避），重试 settings.event_llm_max_retries 次。
    仍失败才抛出，让上层决定（降关键帧数 / 降 max_tokens / 换部署）。
    """
    max_retries = max(0, settings.event_llm_max_retries)
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            if attempt >= max_retries:
                raise
            retry_after = None
            try:
                retry_after = float(exc.response.headers.get("retry-after"))
            except Exception:  # noqa: BLE001
                retry_after = None
            wait = retry_after if retry_after else min(2 ** attempt + 1, 30)
            print(
                f"[event_reporter] 429 限流，{wait:.0f}s 后重试"
                f"（第 {attempt + 1}/{max_retries} 次）...",
                file=sys.stderr,
            )
            time.sleep(wait)


def understand_event(
    frames: list[dict],
    identity: str | list[dict] | None = None,
    objective: str | None = None,
    model: str | None = None,
    scene_context: str | None = None,
    object_context: str | None = None,
) -> dict:
    """对一个事件窗做身份感知的跨帧事件理解。

    Args:
        frames: 该窗的关键帧（按时间序），每项 {"image": data URI 或 文件路径, "timestamp": 文本/秒}。
        identity: 身份上下文——可传已格式化文本，或传 per-person 记录列表（内部会格式化）。
        objective: 可选关注点（如"留意陌生人/包裹被取走"），写进 prompt。
        model: 覆盖部署名（默认 settings.event_llm_deployment 或主部署）。
        scene_context: 可选**场景级**文字上下文（OCR 读出的时间戳/车牌/单号等，LANE D）。与人物身份
            **并列**注入，**不代表任何人的身份**；由 LLM 自行决定时间/物件线索如何配给事件。
        object_context: 可选**场景级**物体上下文（YOLO 检出的包裹/行李/车辆 + 轨迹，LANE D）。同样与
            身份**并列**注入、**不代表身份**；含"疑似包裹则看图认品牌/logo"的提示。

    Returns:
        dict（结构化事件理解）：
          events:[{time, subject, action, abnormal}], summary,
          subjects_involved:[...], alert_level, notification
    """
    if not frames:
        return {"events": [], "summary": "（无关键帧）", "subjects_involved": [],
                "alert_level": "normal", "notification": ""}

    # 身份上下文：列表则格式化成文本
    if isinstance(identity, list):
        identity_text = format_identity_grounding(identity)
    else:
        identity_text = identity or ""

    schema = (
        "请严格输出 JSON（不要多余文字），字段：\n"
        "{\n"
        '  "events": [\n'
        '    {"time": "事件大致时间/对应帧时间戳", "subject": "涉及的身份（如 主体#3 / 员工A123 / 未识别人物）",\n'
        '     "action": "该主体在这段时间做了什么（跨帧叙述）", "abnormal": true 或 false}\n'
        "  ],\n"
        '  "summary": "用 1-3 句话总结这段时间发生了什么",\n'
        '  "subjects_involved": ["涉及到的身份列表"],\n'
        '  "alert_level": "normal | attention | alert",\n'
        '  "notification": "给值班人员的一句话通知"\n'
        "}"
    )
    prompt = (
        "下面是一个事件窗内、按时间顺序抽取的若干关键帧。请结合给出的人物身份，"
        "以及每个关键帧的 bbox/center 坐标 grounding，理解并叙述这段时间发生的跨帧事件。"
        "坐标用于把主体绑定到画面位置、移动方向和相互关系；不要把它当成让你重新检测的任务。\n" + schema
    )
    if objective:
        prompt += f"\n\n特别关注：{objective}"

    content: list[dict] = [{"type": "text", "text": prompt}]
    if identity_text:
        content.append({"type": "text", "text": identity_text})
    if scene_context:
        content.append({"type": "text", "text": scene_context})
    if object_context:
        content.append({"type": "text", "text": object_context})

    detail = settings.event_frame_detail
    content.append({"type": "text", "text": "\n【以下为该事件窗的关键帧（按时间顺序）】"})
    for i, f in enumerate(frames, 1):
        ts = f.get("timestamp")
        label = f"[关键帧 {i}" + (f" @ {ts}]" if ts is not None else "]")
        content.append({"type": "text", "text": label})
        content.append(
            {"type": "image_url", "image_url": {"url": _frame_to_data_uri(f["image"]), "detail": detail}}
        )

    deployment = model or settings.event_llm_deployment or settings.azure_openai_deployment
    client = get_client()
    resp = _create_with_retry(
        client,
        model=deployment,
        messages=[
            {"role": "system", "content": EVENT_SYSTEM},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=settings.event_llm_max_tokens,
    )
    result = parse_json(resp.choices[0].message.content or "{}")
    result.setdefault("events", [])
    result.setdefault("alert_level", "normal")
    result["_model"] = deployment
    result["_frames"] = len(frames)
    return result


# ============ 跨窗整段事件总结（Phase 4 · E）============
# 长视频被切成多个事件窗，各窗已"看图"理解过。这里再做一道**纯文本**整合（无图片、便宜）：
# 把各窗叙述 + 身份名册喂给 LLM，串成整段视频的一个连贯事件故事——靠 ReID 身份把"同一主体在
# 不同窗"关联起来（主体#1 在 0:05 和 1:10 都出现）。

WINDOW_SUMMARY_SYSTEM = (
    "你是监控视频的整段事件总结助手。下面给你同一段视频里【按时间顺序的若干事件窗】"
    "（每个窗已由多模态模型理解过：时间段、涉及的人物身份、发生了什么、告警级别），"
    "以及全程出现过的【人物身份名册】。请把这些窗整合成【整段视频的一个连贯事件故事】："
    "同一身份（主体#/库内身份）在不同窗里是同一个人，按时间把他们的行为串起来；"
    "只依据给定信息，不要臆造画面之外的内容。"
)

WINDOW_SUMMARY_SCHEMA = (
    "请严格输出 JSON（不要多余文字），字段：\n"
    "{\n"
    '  "overall_summary": "用 2-4 句话总结整段视频发生了什么（贯穿各窗）",\n'
    '  "story": [\n'
    '    {"time": "时间/时间段", "subject": "涉及身份（如 主体#1）", "action": "做了什么（跨窗连贯叙述）"}\n'
    "  ],\n"
    '  "subjects": ["涉及到的身份及其一句话概括，如 主体#1：闯入并翻找后离开"],\n'
    '  "overall_alert_level": "normal | attention | alert（整段最高告警级别）",\n'
    '  "notification": "给值班人员的一句话整段通知"\n'
    "}"
)


def _build_roster(windows: list[dict]) -> str:
    """身份名册：每个主体出现在哪些窗的时间段（让模型跨窗认出同一人）。"""
    appear: dict[str, list[str]] = {}
    for w in windows:
        tr = w.get("time_range", ["", ""])
        for p in w.get("people", []):
            sid = p.get("subject_id")
            label = f"主体#{sid}" if sid is not None else f"track {p.get('track_id')}"
            appear.setdefault(label, []).append(f"{tr[0]}~{tr[1]}")
    if not appear:
        return "（无可用身份）"
    return "\n".join(f"- {label}：出现于 {', '.join(rs)}" for label, rs in appear.items())


def _windows_to_text(windows: list[dict]) -> str:
    """把各窗的事件理解结果压成紧凑文本时间线（喂给整段总结，无图片）。"""
    lines: list[str] = []
    for w in windows:
        ev = w.get("event") or {}
        tr = w.get("time_range", ["", ""])
        lines.append(f"[窗{w.get('window_index')} {tr[0]}~{tr[1]}] 告警={ev.get('alert_level', 'normal')}")
        if ev.get("summary"):
            lines.append(f"  概述：{ev['summary']}")
        for e in ev.get("events", []):
            flag = "⚠" if e.get("abnormal") else ""
            lines.append(f"  - {e.get('time')} {e.get('subject')}：{flag}{e.get('action')}")
    return "\n".join(lines) or "（无事件窗）"


def summarize_event_windows(windows: list[dict], model: str | None = None) -> dict:
    """把若干事件窗整合成整段视频的连贯事件故事（纯文本调用）。

    Args:
        windows: analyze_event_stream 产出的窗列表（每项含 time_range / people / event）。
        model: 覆盖部署名（默认主部署）。

    Returns:
        dict：overall_summary, story[{time,subject,action}], subjects[], overall_alert_level, notification。
        无任何已理解的窗时返回 {}（上层据此跳过）。
    """
    ev_windows = [w for w in windows if w.get("event")]
    if not ev_windows:
        return {}

    roster = _build_roster(windows)
    timeline = _windows_to_text(windows)
    prompt = (
        WINDOW_SUMMARY_SCHEMA
        + "\n\n【人物身份名册（同一身份跨窗即同一人）】\n" + roster
        + "\n\n【各事件窗（按时间顺序）】\n" + timeline
    )

    deployment = model or settings.event_llm_deployment or settings.azure_openai_deployment
    client = get_client()
    resp = _create_with_retry(
        client,
        model=deployment,
        messages=[
            {"role": "system", "content": WINDOW_SUMMARY_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=settings.event_llm_max_tokens,
    )
    result = parse_json(resp.choices[0].message.content or "{}")
    result.setdefault("story", [])
    result.setdefault("overall_alert_level", "normal")
    result["_model"] = deployment
    result["_windows"] = len(ev_windows)
    return result


__all__ = ["understand_event", "summarize_event_windows"]
