"""LLM 调用服务：统一转发到现有 llm_client。"""
from __future__ import annotations

from ..llm_client import (
    analyze_single_frame,
    compile_target,
    summarize_events,
    summarize_frames,
)
from ..video_processor import Frame


def summarize_video_frames(frames: list[Frame]) -> dict:
    return summarize_frames(frames)


def summarize_session_events(events: list[dict]) -> dict:
    """末尾总结：把实时整段分析累积的逐帧事件归纳成整体总结。"""
    return summarize_events(events)


def analyze_frame_content(
    image_data_uri: str,
    target: str | None = None,
    reference_image: str | None = None,
    detections: list[dict] | None = None,
    img_w: int | None = None,
    img_h: int | None = None,
) -> dict:
    return analyze_single_frame(
        image_data_uri,
        target=target,
        reference_image=reference_image,
        detections=detections,
        img_w=img_w,
        img_h=img_h,
    )


def compile_target_rule(
    target: str,
    available_classes: list[str],
    reference_image: str | None = None,
) -> dict:
    return compile_target(target, available_classes, reference_image)
