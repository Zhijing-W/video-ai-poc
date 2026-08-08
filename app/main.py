"""Event Monitor FastAPI entry point."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .core import STATIC_DIR, TEMPLATES_DIR
from .event_monitor_i18n import build_page_bundle
from .routers import event_monitor_router, health_router

app = FastAPI(title="Event Monitor", version="1.0.0")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def no_cache_dev_assets(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path in {
        "/",
        "/event-monitor",
        "/event-monitor/zh",
        "/eventmonitor",
    }:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


app.include_router(health_router)
app.include_router(event_monitor_router)


@app.get("/")
def index() -> RedirectResponse:
    return RedirectResponse(url="/event-monitor")


@app.get("/event-monitor", response_class=HTMLResponse)
def event_monitor(request: Request) -> HTMLResponse:
    bundle = build_page_bundle("en")
    return templates.TemplateResponse(
        request,
        "event-monitor.html",
        {
            "ui": bundle["messages"],
            "ui_bundle": bundle,
        },
    )


@app.get("/event-monitor/zh", response_class=HTMLResponse)
def event_monitor_zh(request: Request) -> HTMLResponse:
    bundle = build_page_bundle("zh-CN")
    return templates.TemplateResponse(
        request,
        "event-monitor.html",
        {
            "ui": bundle["messages"],
            "ui_bundle": bundle,
        },
    )


@app.get("/eventmonitor", include_in_schema=False)
def legacy_event_monitor_url() -> RedirectResponse:
    return RedirectResponse(url="/event-monitor")
