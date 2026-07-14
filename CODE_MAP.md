# Event Monitor 代码地图

> 当前默认开发范围。除非任务明确涉及旧版，否则不要读取 `../monitor-v1/`。

## 入口

| 区域 | 文件 |
|---|---|
| FastAPI 应用 | `app/main.py` |
| Event Monitor 路由 | `app/routers/event_monitor.py` |
| 页面 | `templates/event-monitor.html` |
| 前端逻辑 | `static/js/event-monitor/main.js` |
| 页面样式 | `static/css/event-monitor.css` 和基础 CSS |
| CLI 演示 | `scripts/event_analysis_demo.py` |

## 主流程

```text
/event-monitor
  -> POST /api/event-monitor/understand
  -> app/event_analysis_pipeline.py
     -> detector.py + tracker.py
     -> body_reid.py + body_gallery.py
     -> face.py
     -> gait.py
     -> ocr.py
     -> keyframe.py
     -> services/multimodal_identity_fusion.py
     -> services/identity_grounding.py
     -> services/event_understanding.py
  -> 事件窗时间线与整段报告
```

## 功能定位

| 修改内容 | 文件 |
|---|---|
| 上传、样片和本次运行设置 | `app/routers/event_monitor.py` |
| 端到端编排 | `app/event_analysis_pipeline.py` |
| 检测与跟踪 | `app/detector.py`、`app/tracker.py` |
| 人形身份 | `app/body_reid.py`、`app/body_gallery.py` |
| 人脸身份与质量 | `app/face.py` |
| 步态身份 | `app/gait.py` |
| 场景 OCR | `app/ocr.py` |
| 关键帧选择 | `app/keyframe.py` |
| 三路置信度聚合 | `app/services/multimodal_identity_fusion.py` |
| 身份信息打包 | `app/services/identity_grounding.py` |
| 事件 LLM 与整段总结 | `app/services/event_understanding.py` |
| 时间线与设置前端 | `static/js/event-monitor/main.js`、`static/css/event-monitor.css` |

## 文档与实验

- Phase 4 流程图：`docs/phase4-logic-flow.*`
- 身份和人脸质量：`docs/人脸质量与身份融合逻辑.md`
- 云部署：`docs/cloud-deploy/`、`docs/AZURE_DEPLOY.md`
- 糊脸与身份实验：`experiment/糊脸消融实验/`

## 范围规则

1. 默认搜索限制在本目录。
2. 运行数据、模型权重、`.venv` 和输出文件不属于源码。
3. 旧 `track_fusion.py` 只存在于 `../monitor-v1/`。
4. 后续摄像头实时输入作为本应用的新输入适配器，不再复制第三套应用。
