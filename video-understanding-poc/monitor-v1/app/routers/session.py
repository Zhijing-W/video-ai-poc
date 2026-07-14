"""`/monitor-sessions` 会话管理路由。"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..core import (
    MONITOR_DIR,
    delete_monitor_session_cache,
    get_monitor_session_cache,
    set_monitor_session_cache,
)
from ..models import MonitorSessionRequest, SummarizeRequest
from ..services import summarize_session_events
from ..utils import save_data_uri_image

router = APIRouter(tags=["sessions"])
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


@router.post("/summarize")
def summarize(req: SummarizeRequest) -> dict:
    """实时整段分析跑完后，把累积的逐帧事件归纳成末尾总结。"""
    try:
        return summarize_session_events(req.events)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/monitor-sessions")
def save_monitor_session(sess: MonitorSessionRequest) -> dict:
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    session_dir = MONITOR_DIR / session_id
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    meta_entries = []
    for entry in sess.entries:
        frame_url = None
        if entry.image:
            frame_path = frames_dir / f"{entry.seq}.jpg"
            if save_data_uri_image(entry.image, frame_path):
                frame_url = f"/monitor-sessions/{session_id}/frames/{entry.seq}.jpg"
        meta_entries.append(
            {
                "seq": entry.seq,
                "ts": entry.ts,
                "level": entry.level,
                "msg": entry.msg,
                "is_match": entry.is_match,
                "frame_url": frame_url,
                "result": entry.result,
            }
        )

    meta = {
        "id": session_id,
        "started_at": sess.started_at,
        "ended_at": sess.ended_at,
        "target": sess.target,
        "mode": sess.mode,
        "stats": sess.stats or {},
        "summary": sess.summary,
        "entries": meta_entries,
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    set_monitor_session_cache(session_id, meta)
    return {"id": session_id, "frames": len(meta_entries)}


@router.get("/monitor-sessions")
def list_monitor_sessions() -> dict:
    sessions = []
    if MONITOR_DIR.exists():
        for session_dir in sorted(MONITOR_DIR.iterdir(), reverse=True):
            meta_path = session_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            stats = meta.get("stats") or {}
            sessions.append(
                {
                    "id": meta.get("id"),
                    "started_at": meta.get("started_at"),
                    "ended_at": meta.get("ended_at"),
                    "target": meta.get("target"),
                    "mode": meta.get("mode"),
                    "summary": (meta.get("summary") or {}).get("summary") if meta.get("summary") else None,
                    "frames": stats.get("frames") or len(meta.get("entries", [])),
                    "match": stats.get("match", 0),
                    "alert": stats.get("alert", 0),
                }
            )
    return {"sessions": sessions}


@router.get("/monitor-sessions/{session_id}")
def get_monitor_session(session_id: str) -> dict:
    if not _ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="非法 session id")
    cached = get_monitor_session_cache(session_id)
    if cached:
        return cached
    meta_path = MONITOR_DIR / session_id / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session 不存在")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    set_monitor_session_cache(session_id, meta)
    return meta


@router.delete("/monitor-sessions/{session_id}")
def delete_monitor_session(session_id: str) -> dict:
    if not _ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="非法 session id")
    session_dir = MONITOR_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="session 不存在")
    shutil.rmtree(session_dir)
    delete_monitor_session_cache(session_id)
    return {"deleted": session_id}


@router.get("/monitor-sessions/{session_id}/frames/{name}")
def get_monitor_frame(session_id: str, name: str) -> FileResponse:
    if not _ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="非法 session id")
    safe_name = Path(name).name
    frame_path = MONITOR_DIR / session_id / "frames" / safe_name
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="frame 不存在")
    return FileResponse(frame_path, media_type="image/jpeg")
