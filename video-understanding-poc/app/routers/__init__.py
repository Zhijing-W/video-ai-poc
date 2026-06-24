"""FastAPI routers."""
from .analyze import router as analyze_router
from .compare import router as compare_router
from .detect import router as detect_router
from .eventmonitor import router as eventmonitor_router
from .fusion import router as fusion_router
from .identify import router as identify_router
from .session import router as session_router
from .track import router as track_router
from .video import router as video_router

__all__ = [
    "analyze_router",
    "compare_router",
    "detect_router",
    "eventmonitor_router",
    "fusion_router",
    "identify_router",
    "session_router",
    "track_router",
    "video_router",
]
