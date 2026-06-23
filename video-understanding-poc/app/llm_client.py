"""调用 Azure OpenAI vision 模型，对关键帧做理解并返回结构化 JSON。"""
from __future__ import annotations

import json

from openai import AzureOpenAI

from .core.config import settings
from .utils.image_utils import image_to_data_uri
from .video_processor import Frame

SYSTEM_PROMPT = "你是一个严谨的视频理解助手，只根据图片中可见的信息作答，不臆测画面之外的内容。"

USER_PROMPT = """下面是从同一个视频中按时间顺序抽取的关键帧（含时间戳）。
请只根据这些图片中可见的信息进行总结。

请严格输出 JSON，字段如下（不要输出多余文字）：
{
  "summary": "用 1-3 句话总结视频内容",
  "detected_objects": ["可见对象，如 person, package, pet, vehicle"],
  "possible_events": ["可能发生的事件，如 person_appears, object_moved"],
  "notification": "面向用户的一句话通知",
  "confidence": "low | medium | high",
  "evidence": [{"timestamp": "对应帧时间", "observation": "该帧可见信息"}],
  "limitations": "信息不足时说明限制"
}"""


def _client() -> AzureOpenAI:
    settings.require_openai()
    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )


def summarize_frames(frames: list[Frame]) -> dict:
    """把关键帧发给 vision 模型，返回解析后的 dict。"""
    client = _client()

    content: list[dict] = [{"type": "text", "text": USER_PROMPT}]
    for f in frames:
        content.append({"type": "text", "text": f"[帧 {f.frame_id} @ {f.timestamp}]"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_uri(f.local_path), "detail": "low"},
            }
        )

    resp = client.chat.completions.create(
        model=settings.azure_openai_deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=1200,
    )

    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 兜底：模型偶尔包了 ```json 围栏
        cleaned = raw.strip().strip("`")
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
        return json.loads(cleaned)


# ============ 实时单帧分析 + 比对（Phase 2 实时监控用）============

REALTIME_SYSTEM = (
    "你是一个专业的实时监控视频分析助手。只根据当前画面中可见的信息作答，"
    "不臆测画面之外的内容。判定要克制：证据不足时降低置信度或判为不匹配。"
)


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip().strip("`")
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
        return json.loads(cleaned)


def _format_detections(detections: list[dict] | None, img_w: int | None, img_h: int | None) -> str:
    """把 YOLO 检测结果格式化成给 LLM 的 grounding 文本（坐标 + 置信度）。"""
    if not detections:
        return ""
    dims = f"（图像尺寸约 {img_w}×{img_h} 像素，坐标原点在左上角）" if img_w and img_h else ""
    lines = []
    for i, d in enumerate(detections, 1):
        box = d.get("box") or []
        box_str = ", ".join(str(v) for v in box)
        lines.append(f"  #{i} {d.get('label')}  置信度{d.get('confidence')}  box=[{box_str}]")
    return (
        "\n\n【YOLO 物体检测结果（已为你定位，请基于这些框分析框内细节，不要凭空臆测）"
        + dims
        + "】\n"
        + "\n".join(lines)
        + "\n注意：这些框只提供【位置与类别】，不含颜色/外观判断。颜色、衣着、姿态等请你"
        "直接看图自行判断（你的视觉判断为准），不要假设框一定准确。"
    )


def analyze_single_frame(
    image_data_uri: str,
    target: str | None = None,
    reference_image: str | None = None,
    detections: list[dict] | None = None,
    img_w: int | None = None,
    img_h: int | None = None,
) -> dict:
    """对一张实时画面做理解 + 可选目标比对，返回结构化 dict。

    Args:
        image_data_uri: 当前画面（浏览器 canvas 导出的 data URI，如 data:image/jpeg;base64,...）。
        target: 比对目标的文字描述（如 "穿红色外套的人" / "快递包裹"）。
        reference_image: 比对参考图（data URI），与当前画面做相似判定。
        detections: YOLO 检测框列表 [{label, confidence, box:[x1,y1,x2,y2]}]，作为
            grounding 锚点注入提示词，让 LLM 基于确定的框描述细节（接 DB/人脸/人形比对）。
        img_w, img_h: 当前画面像素尺寸，帮助 LLM 理解 box 坐标系。

    Returns:
        dict：scene / detected_objects / subjects[] / match{...} / alert_level / notification。
    """
    client = _client()

    has_target = bool(target) or bool(reference_image)
    target_desc = target or "（见参考目标图）"
    has_dets = bool(detections)

    subjects_schema = (
        '''
  "subjects": [
    {
      "ref": "对应上方 YOLO 检测编号，如 #1",
      "label": "类别，如 person",
      "box": [x1, y1, x2, y2],
      "appearance": "该框内目标的外观描述（如 上衣红色、背黑色包、面向镜头）",
      "attributes": ["可用于后续比对/检索的离散标签，如 male, backpack, facing_camera"]
    }
  ],'''
        if has_dets
        else ""
    )

    prompt = f"""你正在分析一段监控视频的【当前画面】。
请只根据可见信息，严格输出 JSON（不要多余文字），字段如下：
{{
  "scene": "用一句话描述画面整体情况",
  "detected_objects": ["可见对象，如 person, package, vehicle, pet"],{subjects_schema}
  "match": {{
    "is_match": {"true 或 false（当前画面是否出现比对目标）" if has_target else "null（本次无比对目标）"},
    "confidence": "low | medium | high",
    "target": "{target_desc if has_target else ''}",
    "reason": "判定理由（指出画面中的具体依据）"
  }},
  "alert_level": "normal | attention | alert（是否需要告警）",
  "notification": "面向值班人员的一句话通知"
}}"""
    if has_dets:
        prompt += _format_detections(detections, img_w, img_h)
        prompt += (
            "\n\n要求：对每个 YOLO 检测到的关键目标（尤其 person），在 subjects 中给出一条记录，"
            "appearance 和 attributes 要具体、可用于后续人脸/人形比对与数据库检索；box 直接沿用上方坐标。"
        )
    if has_target:
        prompt += f"\n\n比对任务：判断【当前画面】中是否出现目标「{target_desc}」。命中才把 is_match 设为 true。"

    content: list[dict] = [{"type": "text", "text": prompt}]
    if reference_image:
        content.append({"type": "text", "text": "【比对参考目标图】"})
        content.append(
            {"type": "image_url", "image_url": {"url": reference_image, "detail": "low"}}
        )
    content.append({"type": "text", "text": "【当前监控画面】"})
    content.append(
        {"type": "image_url", "image_url": {"url": image_data_uri, "detail": "low"}}
    )

    resp = client.chat.completions.create(
        model=settings.azure_openai_deployment,
        messages=[
            {"role": "system", "content": REALTIME_SYSTEM},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=900,
    )
    return _parse_json(resp.choices[0].message.content or "{}")


# ============ 目标编译（LLM监工级联 · ① 把自然语言报警条件编译成可执行规则）============

COMPILE_SYSTEM = (
    "你是一个监控规则编译器。你的任务是把用户的自然语言报警条件，编译成一台廉价检测器"
    "（YOLO 物体检测 + 简单颜色判断）能否独立执行的结构化规则。要诚实：YOLO 只认物体类别"
    "（不懂颜色/姿态/动作/持物/身份），颜色可由廉价 CV 补充，其余复杂语义只能交给大模型。"
)


def compile_target(
    target: str,
    class_names: list[str],
    reference_image: str | None = None,
) -> dict:
    """把自然语言报警条件编译成 YOLO 可执行规则。

    Args:
        target: 用户输入，如 "出现红色汽车就报警"、"有人摔倒"、"穿红色外套的人"。
        class_names: YOLO 支持的合法类别清单（COCO 80 类），LLM 只能从中选 yolo_class。
        reference_image: 可选参考图（暂用于辅助理解，不强制）。

    Returns:
        dict:
          yolo_class: 选中的 YOLO 类别（必须来自 class_names），无合适项为 null
          attribute: {type:"color", value:"red", region:"whole|upper|lower"} 或 null
          can_yolo_handle: YOLO(+颜色) 能否独立判断（否→每帧回落 gpt-4o）
          summary: 一句话说明编译结果
    """
    client = _client()
    classes_str = ", ".join(class_names)
    prompt = f"""请把下面的监控报警条件编译成结构化规则，严格输出 JSON（不要多余文字）：

报警条件：「{target}」

可选的 YOLO 类别（yolo_class 只能从中选，没有合适的填 null）：
{classes_str}

输出字段：
{{
  "yolo_class": "最匹配的 YOLO 类别，或 null",
  "attribute": {{
    "type": "color",
    "value": "目标颜色英文小写，如 red/blue/black",
    "region": "whole（整体，如车）| upper（上半身，如上衣）| lower（下半身）"
  }} 或 null（无颜色等附加属性时）,
  "can_yolo_handle": true 或 false,
  "summary": "一句话说明：将如何监视该目标"
}}

判定 can_yolo_handle 的规则：
- 若目标 = 某个 YOLO 类别（可叠加颜色判断）→ true（如"红色汽车"=car+red、"行人"=person）
- 若目标依赖 YOLO 无法判断的语义（姿态如摔倒、动作如打架、持物如拿刀、身份如某个人脸）→ false
- yolo_class 为 null 时，can_yolo_handle 必须为 false"""

    content: list[dict] = [{"type": "text", "text": prompt}]
    if reference_image:
        content.append({"type": "text", "text": "【参考目标图（辅助理解）】"})
        content.append(
            {"type": "image_url", "image_url": {"url": reference_image, "detail": "low"}}
        )

    resp = client.chat.completions.create(
        model=settings.azure_openai_deployment,
        messages=[
            {"role": "system", "content": COMPILE_SYSTEM},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=400,
    )
    plan = _parse_json(resp.choices[0].message.content or "{}")

    # 兜底校验：yolo_class 必须合法，否则强制回落 LLM
    yc = plan.get("yolo_class")
    if yc not in class_names:
        plan["yolo_class"] = None
        plan["can_yolo_handle"] = False
    if not plan.get("yolo_class"):
        plan["can_yolo_handle"] = False
    return plan


# ============ 末尾总结（实时整段分析跑完后，把逐帧事件归纳成一段总结）============
# 纯文本廉价调用：输入实时管线累积的逐帧事件时间线，输出整体总结 JSON。

SUMMARY_SYSTEM = (
    "你是视频理解助手。下面给你一段监控视频按时间顺序的事件记录（纯文字），"
    "请据此做整体总结。不要臆造记录之外的信息。"
)


def summarize_events(events: list[dict]) -> dict:
    """把实时整段分析累积的逐帧事件，归纳成整体总结（纯文本廉价调用）。

    Args:
        events: [{timestamp, observation, alert_level}, ...]，由前端实时管线累积。

    Returns:
        dict: summary / detected_objects / possible_events / overall_alert_level /
        notification / confidence。
    """
    client = _client()
    timeline = "\n".join(
        f"- {e.get('timestamp', '?')} [{e.get('alert_level', 'normal')}] {e.get('observation', '')}"
        for e in events
    ) or "（无事件）"
    prompt = f"""一段监控视频按时间顺序的事件记录如下：
{timeline}

请基于以上记录，严格输出 JSON（不要多余文字）：
{{
  "summary": "用 1-3 句话总结整段视频发生了什么",
  "detected_objects": ["全程出现过的关键对象，如 person, vehicle, package"],
  "possible_events": ["归纳出的关键事件，如 person_appears, object_moved"],
  "overall_alert_level": "normal | attention | alert（整段最高告警级别）",
  "notification": "给值班人员的一句话总结通知",
  "confidence": "low | medium | high"
}}"""

    resp = client.chat.completions.create(
        model=settings.azure_openai_deployment,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=600,
    )
    return _parse_json(resp.choices[0].message.content or "{}")
