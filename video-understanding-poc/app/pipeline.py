"""端到端处理管线（被竖切脚本和 FastAPI 共用）。

流程：视频 → ffmpeg 抽帧 → Azure OpenAI vision → 结构化 JSON。
把这段核心逻辑集中在这里，避免脚本和 Web 层各写一份。
"""
from __future__ import annotations

import time
from pathlib import Path

from .core.config import settings
from .services.llm_service import summarize_video_frames
from .video_processor import extract_frames, extract_frames_smart, frames_as_dicts


def analyze_video(video_path: str | Path, out_dir: str | Path) -> dict:
    """对一个视频做完整理解（供竖切 CLI 脚本用），返回统一结构 payload。

    Args:
        video_path: 输入视频路径。
        out_dir: 输出目录（帧会放在 out_dir/frames）。

    Returns:
        dict，字段：video / model_deployment / frame_selection /
        frames_used / elapsed_seconds / llm_result。
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"

    # Step 7：智能抽帧（场景突变 + 定时兜底），可由 SMART_FRAMES 关闭回退定时抽帧。
    if settings.smart_frames:
        frames = extract_frames_smart(video_path, frames_dir)
        frame_selection = {
            "method": "smart",
            "scene_threshold": settings.scene_threshold,
            "fallback_interval_seconds": settings.fallback_interval_seconds,
        }
    else:
        frames = extract_frames(video_path, frames_dir)
        frame_selection = {
            "method": "fixed",
            "interval_seconds": settings.frame_interval_seconds,
        }

    t0 = time.time()
    result = summarize_video_frames(frames)
    elapsed = round(time.time() - t0, 2)

    return {
        "video": str(video_path),
        "model_deployment": settings.azure_openai_deployment,
        "frame_selection": frame_selection,
        "frames_used": frames_as_dicts(frames),
        "elapsed_seconds": elapsed,
        "llm_result": result,
    }
