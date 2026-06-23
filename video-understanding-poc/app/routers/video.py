"""健康检查路由。"""
from __future__ import annotations

from fastapi import APIRouter

from .. import storage
from ..core import settings

router = APIRouter(tags=["video"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "deployment": settings.azure_openai_deployment,
        "storage_enabled": storage.is_enabled(),
    }
