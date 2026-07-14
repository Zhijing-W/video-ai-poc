"""FastAPI 入口：注册页面与业务路由。"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .core import STATIC_DIR, TEMPLATES_DIR
from .routers import (
    analyze_router,
    compare_router,
    detect_router,
    fusion_router,
    identify_router,
    session_router,
    track_router,
    video_router,
)

app = FastAPI(title="视频理解 PoC", version="0.3.0")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def no_cache_dev_assets(request: Request, call_next):
    """开发期防缓存：前端 JS/CSS/页面频繁迭代，禁用浏览器缓存避免跑到旧代码。"""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static") or path in ("/", "/monitor"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

app.include_router(video_router)
app.include_router(analyze_router)
app.include_router(detect_router)
app.include_router(track_router)
app.include_router(identify_router)
app.include_router(fusion_router)
app.include_router(compare_router)
app.include_router(session_router)


@app.get("/")
def index() -> RedirectResponse:
    # 上传页已并入监控页（mode② 整段报告），统一入口到 /monitor。
    return RedirectResponse(url="/monitor")


@app.get("/monitor", response_class=HTMLResponse)
def monitor(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "monitor.html")
