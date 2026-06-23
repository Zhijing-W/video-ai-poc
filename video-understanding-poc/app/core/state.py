"""运行时内存状态：视频任务、会话缓存与时间工具。"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

JOBS: dict[str, dict[str, Any]] = {}
JOB_LOCK = threading.Lock()
MONITOR_SESSIONS: dict[str, dict[str, Any]] = {}
SESSION_LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_job(job_id: str, **fields: Any) -> None:
    with JOB_LOCK:
        JOBS.setdefault(job_id, {})
        JOBS[job_id].update(fields)


def get_job(job_id: str) -> dict[str, Any] | None:
    with JOB_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def set_monitor_session_cache(session_id: str, payload: dict[str, Any]) -> None:
    with SESSION_LOCK:
        MONITOR_SESSIONS[session_id] = payload


def get_monitor_session_cache(session_id: str) -> dict[str, Any] | None:
    with SESSION_LOCK:
        cached = MONITOR_SESSIONS.get(session_id)
        return dict(cached) if cached else None


def delete_monitor_session_cache(session_id: str) -> None:
    with SESSION_LOCK:
        MONITOR_SESSIONS.pop(session_id, None)
