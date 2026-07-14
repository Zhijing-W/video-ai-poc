# Monitor v1 代码地图

> 状态：冻结旧版。除非任务明确指定 Monitor v1，否则不要修改。

## 入口

| 区域 | 文件 |
|---|---|
| FastAPI 应用 | `app/main.py` |
| 监控页面 | `templates/monitor.html` |
| 前端入口 | `static/js/main.js` |
| 前端样式 | `static/css/` |
| API 路由 | `app/routers/` |
| 路由服务层 | `app/services/` |
| API 数据结构 | `app/models/` |
| 配置与运行状态 | `app/core/` |

## 主流程

```text
/monitor
  -> static/js/main.js
  -> /analyze-frame 或 /track
  -> tracker.py
  -> reid.py + gallery.py
  -> track_fusion.py
  -> 页面展示与历史归档
```

## 功能定位

| 修改内容 | 文件 |
|---|---|
| 逐帧分析与 LLM 门控 | `app/routers/analyze.py`、`app/services/gate_service.py`、`app/services/llm_service.py` |
| 检测 | `app/routers/detect.py`、`app/detector.py`、`app/services/yolo_service.py` |
| 跟踪 | `app/routers/track.py`、`app/tracker.py` |
| 人形 ReID 与主体库 | `app/routers/identify.py`、`app/reid.py`、`app/gallery.py` |
| 轨迹级多帧融合 | `app/routers/fusion.py`、`app/track_fusion.py` |
| 目标比对 | `app/routers/compare.py`、`app/services/cruise_service.py` |
| 会话归档 | `app/routers/session.py`、`app/core/state.py` |
| 旧版前端 | `templates/monitor.html`、`static/js/`、`static/css/` |

## 文档与评估

- Phase 3 图：`docs/phase3-*.png`
- 历史重构计划：`docs/REFACTOR_PLAN_LEGACY.md`
- Phase 3 评估：`scripts/eval_phase3.py`

本应用中不再保留 Event Monitor 文件。
