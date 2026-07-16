"""Event Monitor API: video stream to event-window timeline.

与 monitor（逐帧分析）不同：本页是"**视频流 → 事件窗时间线**"范式。后端只做两件事：
  - `GET  /api/event-monitor/samples`：列出 data/samples 下的样片。
  - `POST /api/event-monitor/understand`：处理样片或上传视频。
    `event_analysis_pipeline.analyze_event_stream`（同步，几十秒~1分钟），返回事件窗 JSON（含关键帧缩略图）。

故意做成**同步**：PoC 演示，处理完一次性返回，前端转圈等待即可，省掉 job 轮询的复杂度。
"""
from __future__ import annotations

import asyncio
import uuid
from copy import deepcopy
from pathlib import Path

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .. import body_reid as reid_mod
from ..core.config import ALLOWED_VIDEO_SUFFIXES, DATA_DIR, OUTPUT_DIR, settings
from ..event_analysis_pipeline import analyze_event_stream
from ..services.event_reporter import summarize_event_windows, understand_event

router = APIRouter(prefix="/api/event-monitor", tags=["event-monitor"])

SAMPLES_DIR = DATA_DIR / "samples"
OUT_DIR = OUTPUT_DIR / "event-monitor"
_RUN_LOCK = asyncio.Lock()


@router.get("/samples")
def list_samples() -> dict:
    """列出可选样片（data/samples 下的视频文件）。"""
    items = []
    if SAMPLES_DIR.exists():
        for p in sorted(SAMPLES_DIR.iterdir()):
            if p.suffix.lower() in ALLOWED_VIDEO_SUFFIXES:
                items.append({"name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)})
    return {"samples": items}


@router.post("/complete")
def complete_from_dry_run(body: dict = Body(...)) -> dict:
    """把已有 dry-run 结果继续送进 LLM，不重新跑抽帧/检测/跟踪/ReID。"""
    payload = deepcopy(body.get("payload") or {})
    objective = body.get("objective") or None
    if not payload.get("windows"):
        raise HTTPException(400, "没有可继续理解的 dry-run windows")

    for w in payload["windows"]:
        if w.get("event"):
            continue
        keyframes = w.get("keyframes") or []
        if not keyframes:
            raise HTTPException(400, "dry-run 结果里没有关键帧图片，请重新跑一次 dry-run")
        frames = [
            {"image": k.get("image"), "timestamp": k.get("timestamp")}
            for k in keyframes
            if k.get("image")
        ]
        w["event"] = understand_event(frames, w.get("identity_context") or "", objective=objective)

    payload["dry_run"] = False
    payload["model"] = settings.event_llm_deployment or settings.azure_openai_deployment
    if settings.event_overall_summary:
        try:
            payload["overall"] = summarize_event_windows(payload["windows"]) or None
        except Exception as exc:  # 总结失败不影响逐窗事件结果
            payload["overall"] = {"error": str(exc)}
    return payload


@router.post("/understand")
async def understand(
    sample: str | None = Form(None),
    file: UploadFile | None = File(None),
    fps: float = Form(2.0),
    max_keyframes: int = Form(8),
    objective: str | None = Form(None),
    with_face: bool = Form(False),
    with_gait: bool = Form(False),
    with_ocr: bool = Form(False),
    with_objects: bool = Form(False),
    dry_run: bool = Form(False),
    # ---- 本次请求覆盖的可插拔开关（设置面板传来；留空=用默认，仅本次生效不持久）----
    face_rec_backend: str | None = Form(None),   # arcface | adaface
    face_superres: str | None = Form(None),      # off | gfpgan
    face_3d_cue: bool | None = Form(None),
    reid_backend: str | None = Form(None),       # auto | osnet | resnet50 | coarse
    reid_decision_top_k: int | None = Form(None),
    reid_consistency_enabled: bool | None = Form(None),
    reid_vote_score_thresh: float | None = Form(None),
    reid_consistency_ratio: float | None = Form(None),
    reid_top1_margin: float | None = Form(None),
    track_backend: str | None = Form(None),      # bytetrack | botsort | botsort_reid
    max_window_seconds: float | None = Form(None),
    stitch_thresh: float | None = Form(None),
) -> dict:
    """对"样片或上传视频"跑端到端事件理解，返回事件窗时间线。

    设置面板的模型/能力开关随本请求传入，用 settings.override 临时覆盖、仅本次生效。
    """
    run_id = uuid.uuid4().hex[:12]
    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 解析视频来源：上传优先，否则用样片名。
    if file is not None and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            raise HTTPException(400, f"不支持的视频格式：{suffix}")
        video_path = run_dir / f"input{suffix}"
        video_path.write_bytes(await file.read())
    elif sample:
        # 防路径穿越：只取文件名，限定在 samples 目录内
        video_path = SAMPLES_DIR / Path(sample).name
        if not video_path.exists():
            raise HTTPException(404, f"样片不存在：{sample}")
    else:
        raise HTTPException(400, "请选择样片或上传视频")

    overrides = {
        "face_rec_backend": (face_rec_backend or None),
        "face_superres": (face_superres or None),
        "face_3d_cue": face_3d_cue,
        "reid_backend": (reid_backend or None),
        "reid_decision_top_k": reid_decision_top_k,
        "reid_consistency_enabled": reid_consistency_enabled,
        "reid_vote_score_thresh": reid_vote_score_thresh,
        "reid_consistency_ratio": reid_consistency_ratio,
        "reid_top1_margin": reid_top1_margin,
        "track_backend": (track_backend or None),
    }
    async with _RUN_LOCK:
        try:
            with settings.override(**overrides):
                if reid_backend:
                    reid_mod.reset_backend()
                config_used = {
                    "with_face": with_face,
                    "with_gait": with_gait,
                    "with_ocr": with_ocr,
                    "with_objects": with_objects,
                    "face_rec_backend": settings.face_rec_backend,
                    "face_superres": settings.face_superres,
                    "face_3d_cue": settings.face_3d_cue,
                    "reid_backend": settings.reid_backend,
                    "reid_decision_top_k": settings.reid_decision_top_k,
                    "reid_consistency_enabled": settings.reid_consistency_enabled,
                    "reid_vote_score_thresh": settings.reid_vote_score_thresh,
                    "reid_consistency_ratio": settings.reid_consistency_ratio,
                    "reid_top1_margin": settings.reid_top1_margin,
                    "track_backend": settings.track_backend,
                }
                payload = await run_in_threadpool(
                    analyze_event_stream,
                    video_path,
                    run_dir,
                    fps=fps,
                    run_llm=not dry_run,
                    with_face=with_face,
                    with_gait=with_gait,
                    with_ocr=with_ocr,
                    with_objects=with_objects,
                    objective=objective or None,
                    max_keyframes=max_keyframes,
                    max_window_seconds=max_window_seconds,
                    stitch_thresh=stitch_thresh,
                    include_keyframe_images=True,
                    session_id=f"event-monitor-{run_id}",
                )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"事件理解失败：{exc}") from exc
        finally:
            if reid_backend:
                reid_mod.reset_backend()

    payload["run_id"] = run_id
    payload["config_used"] = config_used
    return payload
