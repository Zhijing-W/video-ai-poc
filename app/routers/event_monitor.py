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
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .. import body_reid as reid_mod
from .. import face as face_mod
from ..core.config import ALLOWED_VIDEO_SUFFIXES, DATA_DIR, OUTPUT_DIR, settings
from ..event_analysis_pipeline import EventAnalysisRunError, analyze_event_stream
from ..event_monitor_i18n import normalize_report_language, resolve_report_language
from ..identity.gallery_seed import GallerySeedError, load_gallery_seed
from ..services.event_chat import (
    chat_about_run,
    export_run_prompt,
    persist_run_snapshot,
)
from ..services.event_reporter import summarize_event_windows, understand_event
from ..services.llm_models import model_catalog, resolve_model

router = APIRouter(prefix="/api/event-monitor", tags=["event-monitor"])

SAMPLES_DIR = DATA_DIR / "samples"
OUT_DIR = OUTPUT_DIR / "event-monitor"
_RUN_LOCK = asyncio.Lock()
_STARTUP_FACE_SUPERRES = settings.face_superres
_STARTUP_CODEFORMER_FIDELITY = settings.face_codeformer_fidelity
_STARTUP_REID_BACKEND = reid_mod.validate_backend(settings.reid_backend)
_INLINE_IMAGE_PREFIXES = (
    "data:image/jpeg;base64,",
    "data:image/png;base64,",
    "data:image/webp;base64,",
)


def _inline_keyframe_image(value: object) -> str:
    if not isinstance(value, str) or not value.startswith(_INLINE_IMAGE_PREFIXES):
        raise HTTPException(400, "dry-run keyframes must be inline image data URIs")
    return value


class RunChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    model: str | None = None
    language: str | None = None


@router.get("/samples")
def list_samples() -> dict:
    """列出可选样片（data/samples 下的视频文件）。"""
    items = []
    if SAMPLES_DIR.exists():
        for p in sorted(SAMPLES_DIR.iterdir()):
            if p.suffix.lower() in ALLOWED_VIDEO_SUFFIXES:
                item = {
                    "name": p.name,
                    "size_mb": round(p.stat().st_size / 1e6, 1),
                }
                try:
                    seed = load_gallery_seed(p)
                    if seed is not None:
                        item["gallery_seed"] = seed.public_dict()
                except GallerySeedError as exc:
                    item["gallery_seed_error"] = str(exc)
                items.append(item)
    return {"samples": items}


@router.get("/superres-backends")
def list_superres_backends() -> dict:
    return {
        "default": _STARTUP_FACE_SUPERRES,
        "backends": ["off", *face_mod.available_superres_backends()],
        "metadata": {
            "codeformer": {
                "requires_fidelity": True,
                "fidelity_default": _STARTUP_CODEFORMER_FIDELITY,
                "fidelity_min": 0.0,
                "fidelity_max": 1.0,
            },
        },
    }


@router.get("/reid-backends")
def list_reid_backends() -> dict:
    return {
        "default": _STARTUP_REID_BACKEND,
        "backends": ["auto", *reid_mod.available_backends()],
        "metadata": reid_mod.backend_metadata(),
    }


@router.get("/llm-models")
def list_llm_models(language: str | None = None) -> dict:
    return model_catalog(locale=normalize_report_language(language))


@router.post("/complete")
def complete_from_dry_run(body: dict = Body(...)) -> dict:
    """把已有 dry-run 结果继续送进 LLM，不重新跑抽帧/检测/跟踪/ReID。"""
    payload = deepcopy(body.get("payload") or {})
    objective = body.get("objective") or None
    requested_language = normalize_report_language(body.get("language"))
    report_language = requested_language or normalize_report_language(
        payload.get("report_language")
    )
    try:
        selection = resolve_model(
            "analysis",
            body.get("analysis_model"),
            locale=report_language,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    if not payload.get("windows"):
        raise HTTPException(400, "没有可继续理解的 dry-run windows")

    for w in payload["windows"]:
        if w.get("event"):
            continue
        keyframes = w.get("keyframes") or []
        if not keyframes:
            raise HTTPException(400, "dry-run 结果里没有关键帧图片，请重新跑一次 dry-run")
        frames = [
            {
                "image": _inline_keyframe_image(k.get("image")),
                "timestamp": k.get("timestamp"),
            }
            for k in keyframes
        ]
        w["event"] = understand_event(
            frames,
            w.get("identity_context") or "",
            objective=objective,
            model=selection.deployment,
            model_name=selection.model,
            scene_context=w.get("scene_context") or None,
            object_context=w.get("object_context") or None,
            language=report_language,
            window=w,
        )

    payload["dry_run"] = False
    payload["model"] = selection.model
    payload["llm_selection"] = selection.public_dict()
    payload["report_language"] = resolve_report_language(report_language)
    if settings.event_overall_summary:
        try:
            payload["overall"] = (
                summarize_event_windows(
                    payload["windows"],
                    model=selection.deployment,
                    model_name=selection.model,
                    language=report_language,
                )
                or None
            )
        except Exception as exc:  # 总结失败不影响逐窗事件结果
            payload["overall"] = {"error": str(exc)}
    if payload.get("run_id"):
        persist_run_snapshot(payload)
    return payload


@router.post("/runs/{run_id}/chat")
def chat_with_run(run_id: str, body: RunChatRequest) -> dict:
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "问题不能为空")
    try:
        selection = resolve_model(
            "chat",
            body.model,
            prompt=question,
            locale=normalize_report_language(body.language),
        )
        return chat_about_run(run_id, question, selection)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"大模型问答失败：{exc}") from exc


@router.get("/runs/{run_id}/prompt")
def get_run_prompt(run_id: str, format: str = "json") -> Response:
    """Return one persisted run's canonical JSON or exact compact LLM evidence."""
    try:
        content, extension = export_run_prompt(run_id, format)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    media_type = "application/json; charset=utf-8" if extension == "json" else (
        "text/tab-separated-values; charset=utf-8"
    )
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": (
                f'attachment; filename="event-monitor_{run_id}_prompt.{extension}"'
            ),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/understand")
async def understand(
    sample: str | None = Form(None),
    file: UploadFile | None = File(None),
    fps: float = Form(2.0),
    tracking_fps: float | None = Form(None),
    max_keyframes: int = Form(8),
    objective: str | None = Form(None),
    with_body: bool = Form(True),
    with_face: bool = Form(False),
    with_gait: bool = Form(False),
    with_ocr: bool = Form(False),
    with_objects: bool = Form(False),
    dry_run: bool = Form(False),
    analysis_model: str | None = Form(None),
    # ---- 本次请求覆盖的可插拔开关（设置面板传来；留空=用默认，仅本次生效不持久）----
    face_rec_backend: str | None = Form(None),   # arcface | adaface
    face_superres: str | None = Form(None),      # off | registered backend
    face_codeformer_fidelity: float | None = Form(None),
    face_3d_cue: bool | None = Form(None),
    reid_backend: str | None = Form(None),       # auto | registered backend
    reid_decision_top_k: int | None = Form(None),
    reid_consistency_enabled: bool | None = Form(None),
    reid_vote_score_thresh: float | None = Form(None),
    reid_consistency_ratio: float | None = Form(None),
    reid_top1_margin: float | None = Form(None),
    track_backend: str | None = Form(None),      # bytetrack | botsort | botsort_reid
    max_window_seconds: float | None = Form(None),
    stitch_thresh: float | None = Form(None),
    language: str | None = Form(None),
) -> dict:
    """对"样片或上传视频"跑端到端事件理解，返回事件窗时间线。

    设置面板的模型/能力开关随本请求传入，用 settings.override 临时覆盖、仅本次生效。
    """
    requested_face_superres = face_superres or None
    if not 0.5 <= fps <= 8:
        raise HTTPException(400, "fps 必须在 [0.5, 8] 范围内")
    effective_tracking_fps = float(
        settings.event_tracking_fps
        if tracking_fps is None
        else tracking_fps
    )
    if not 1 <= effective_tracking_fps <= 30:
        raise HTTPException(400, "tracking_fps 必须在 [1, 30] 范围内")
    if effective_tracking_fps < fps:
        raise HTTPException(400, "tracking_fps 不能低于语义 fps")
    if (
        face_codeformer_fidelity is not None
        and not 0.0 <= face_codeformer_fidelity <= 1.0
    ):
        raise HTTPException(400, "face_codeformer_fidelity 必须在 [0, 1] 范围内")
    explicit_face_superres = None
    if requested_face_superres is not None:
        try:
            explicit_face_superres = face_mod.validate_superres_backend(
                requested_face_superres
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    try:
        selected_reid_backend = reid_mod.validate_backend(
            reid_backend if reid_backend is not None else settings.reid_backend
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    report_language = normalize_report_language(language)
    try:
        analysis_selection = resolve_model(
            "analysis",
            analysis_model,
            require_deployment=not dry_run,
            locale=report_language,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    run_id = uuid.uuid4().hex[:12]
    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 解析视频来源：上传优先，否则用样片名。
    gallery_seed = None
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
        try:
            gallery_seed = load_gallery_seed(video_path)
        except GallerySeedError as exc:
            raise HTTPException(400, f"样片 gallery manifest 无效：{exc}") from exc
    else:
        raise HTTPException(400, "请选择样片或上传视频")

    async with _RUN_LOCK:
        selected_face_superres = explicit_face_superres
        if selected_face_superres is None and with_face:
            try:
                selected_face_superres = face_mod.validate_superres_backend(
                    settings.face_superres
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        overrides = {
            "face_rec_backend": (face_rec_backend or None),
            "face_superres": selected_face_superres,
            "face_codeformer_fidelity": face_codeformer_fidelity,
            "face_3d_cue": face_3d_cue,
            "reid_backend": selected_reid_backend,
            "reid_decision_top_k": reid_decision_top_k,
            "reid_consistency_enabled": reid_consistency_enabled,
            "reid_vote_score_thresh": reid_vote_score_thresh,
            "reid_consistency_ratio": reid_consistency_ratio,
            "reid_top1_margin": reid_top1_margin,
            "track_backend": (track_backend or None),
            "event_llm_deployment": analysis_selection.deployment,
        }
        try:
            with settings.override(**overrides):
                if selected_reid_backend != settings.reid_backend or reid_backend:
                    reid_mod.reset_backend()
                config_used = {
                    "with_body": with_body,
                    "with_face": with_face,
                    "with_gait": with_gait,
                    "with_ocr": with_ocr,
                    "with_objects": with_objects,
                    "face_rec_backend": settings.face_rec_backend,
                    "face_superres": settings.face_superres,
                    "face_codeformer_fidelity": settings.face_codeformer_fidelity,
                    "face_3d_cue": settings.face_3d_cue,
                    "reid_backend": settings.reid_backend,
                    "reid_decision_top_k": settings.reid_decision_top_k,
                    "reid_consistency_enabled": settings.reid_consistency_enabled,
                    "reid_vote_score_thresh": settings.reid_vote_score_thresh,
                    "reid_consistency_ratio": settings.reid_consistency_ratio,
                    "reid_top1_margin": settings.reid_top1_margin,
                    "track_backend": settings.track_backend,
                    "tracking_fps": effective_tracking_fps,
                    "semantic_fps": fps,
                    "analysis_model_requested": analysis_selection.requested,
                    "analysis_model_selected": analysis_selection.selected,
                    "gallery_seed": (
                        gallery_seed.public_dict()
                        if gallery_seed is not None
                        else None
                    ),
                }
                payload = await run_in_threadpool(
                    analyze_event_stream,
                    video_path,
                    run_dir,
                    fps=fps,
                    tracking_fps=effective_tracking_fps,
                    run_llm=not dry_run,
                    with_body=with_body,
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
                    report_language=report_language,
                    llm_model=analysis_selection.deployment,
                    llm_model_name=analysis_selection.model,
                    gallery_seed=gallery_seed,
                )
        except EventAnalysisRunError as exc:
            if isinstance(exc.__cause__, GallerySeedError):
                raise HTTPException(
                    400,
                    f"样片 gallery 建档失败：{exc.__cause__}",
                ) from exc
            timing = exc.body_reid_timing
            error_config = dict(config_used)
            error_config["reid_backend_effective"] = timing.get("backend")
            error_config["reid_device"] = timing.get("device")
            raise HTTPException(
                500,
                detail={
                    "message": f"事件理解失败：{exc}",
                    "body_reid_timing": timing,
                    "config_used": error_config,
                },
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"事件理解失败：{exc}") from exc
        finally:
            if selected_reid_backend != _STARTUP_REID_BACKEND or reid_backend:
                reid_mod.reset_backend()

    payload["run_id"] = run_id
    payload["model"] = analysis_selection.model
    payload["llm_selection"] = analysis_selection.public_dict()
    payload["report_language"] = resolve_report_language(
        payload.get("report_language") or report_language
    )
    timing = payload.get("body_reid_timing") or {}
    config_used["reid_backend_effective"] = (
        payload.get("reid_backend") or timing.get("backend")
    )
    config_used["reid_device"] = (
        (payload.get("runtime") or {}).get("reid_device")
        or timing.get("device")
    )
    payload["config_used"] = config_used
    persist_run_snapshot(payload)
    return payload
