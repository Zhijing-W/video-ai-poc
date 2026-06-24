"""事件监控页路由（Phase 4 · Step 26）—— 身份感知·多帧事件理解的 Web 入口。

与 monitor（逐帧分析）不同：本页是"**视频流 → 事件窗时间线**"范式。后端只做两件事：
  - `GET  /eventmonitor/samples`  ：列出 data/samples 下的样片，供前端下拉选择。
  - `POST /eventmonitor/understand`：接收"选中的样片"或"上传的视频" + 参数，跑
    `event_pipeline.analyze_event_stream`（同步，几十秒~1分钟），返回事件窗 JSON（含关键帧缩略图）。

故意做成**同步**：PoC 演示，处理完一次性返回，前端转圈等待即可，省掉 job 轮询的复杂度。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..core.config import ALLOWED_VIDEO_SUFFIXES, BASE_DIR, DATA_DIR
from ..event_pipeline import analyze_event_stream

router = APIRouter(tags=["eventmonitor"])

SAMPLES_DIR = DATA_DIR / "samples"
OUT_DIR = BASE_DIR / "out" / "eventmonitor"


@router.get("/eventmonitor/samples")
def list_samples() -> dict:
    """列出可选样片（data/samples 下的视频文件）。"""
    items = []
    if SAMPLES_DIR.exists():
        for p in sorted(SAMPLES_DIR.iterdir()):
            if p.suffix.lower() in ALLOWED_VIDEO_SUFFIXES:
                items.append({"name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)})
    return {"samples": items}


@router.post("/eventmonitor/understand")
async def understand(
    sample: str | None = Form(None),
    file: UploadFile | None = File(None),
    fps: float = Form(2.0),
    max_keyframes: int = Form(8),
    objective: str | None = Form(None),
    with_face: bool = Form(False),
    dry_run: bool = Form(False),
) -> dict:
    """对"样片或上传视频"跑端到端事件理解，返回事件窗时间线。"""
    # 解析视频来源：上传优先，否则用样片名
    if file is not None and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            raise HTTPException(400, f"不支持的视频格式：{suffix}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        video_path = OUT_DIR / f"upload{suffix}"
        video_path.write_bytes(await file.read())
    elif sample:
        # 防路径穿越：只取文件名，限定在 samples 目录内
        video_path = SAMPLES_DIR / Path(sample).name
        if not video_path.exists():
            raise HTTPException(404, f"样片不存在：{sample}")
    else:
        raise HTTPException(400, "请选择样片或上传视频")

    try:
        payload = analyze_event_stream(
            video_path,
            OUT_DIR,
            fps=fps,
            run_llm=not dry_run,
            with_face=with_face,
            objective=objective or None,
            max_keyframes=max_keyframes,
            include_keyframe_images=True,
            session_id="eventmonitor",
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"事件理解失败：{exc}") from exc
    return payload
