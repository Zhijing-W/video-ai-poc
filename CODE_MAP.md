# Event Monitor 代码地图

> 当前默认开发范围。旧版位于独立的 `feature/monitor-v1` 分支。

## 入口

| 区域 | 文件 |
|---|---|
| FastAPI 应用 | `app/main.py` |
| Event Monitor 路由 | `app/routers/event_monitor.py` |
| 页面 | `templates/event-monitor.html` |
| 前端逻辑 | `static/js/event-monitor/`（入口、API、设置、渲染、身份画廊、时间线、灯箱） |
| 页面样式 | `static/css/event-monitor.css` + `static/css/event-monitor/` 连续级联切片 |
| CLI 演示 | `scripts/event_analysis_demo.py` |

## 主流程

```text
/event-monitor
  -> POST /api/event-monitor/understand
  -> app/event_analysis_pipeline.py
     -> app/pipeline/session.py
     -> app/pipeline/windowing.py
     -> app/pipeline/spatial_context.py
     -> app/pipeline/object_context.py
     -> detector.py + tracker.py
     -> body_reid.py + identity/embedding_gallery.py
     -> identity/resolution.py
     -> identity/face_attachment.py
     -> face.py + identity/face/{quality,super_resolution,fiqa/cr_fiqa}.py + face_fiqa.py
     -> gait.py
     -> ocr.py
     -> keyframe.py
     -> identity/identity_confidence.py
     -> identity/identity_context.py
     -> services/event_reporter.py + openai_client.py
  -> 事件窗时间线与整段报告
```

## 功能定位

| 修改内容 | 文件 |
|---|---|
| 上传、样片和本次运行设置 | `app/routers/event_monitor.py` |
| 端到端编排兼容入口 | `app/event_analysis_pipeline.py` |
| 事件会话与分窗/grounding | `app/pipeline/session.py`、`app/pipeline/windowing.py`、`app/pipeline/spatial_context.py`、`app/pipeline/object_context.py` |
| 检测与跟踪 | `app/detector.py`、`app/tracker.py` |
| 人形身份与 gallery | `app/body_reid.py`、`app/identity/embedding_gallery.py`、`app/body_gallery.py` |
| 身份证据选帧/归并 | `app/identity/evidence_selection.py`、`app/identity/face_attachment.py`、`app/identity/resolution.py` |
| 人脸身份与质量 | `app/face.py`、`app/identity/face/quality.py`、`app/identity/face/super_resolution.py`、`app/identity/face/fiqa/cr_fiqa.py`、`app/face_fiqa.py` |
| 步态身份 | `app/gait.py` |
| 场景 OCR | `app/ocr.py` |
| 关键帧选择 | `app/keyframe.py` |
| 多路线身份置信度 | `app/identity/identity_confidence.py`；旧导入兼容在 `app/services/multimodal_identity_fusion.py` |
| 身份信息打包 | `app/identity/identity_context.py`、`app/services/identity_grounding.py` |
| OpenAI 客户端与事件报告 | `app/openai_client.py`、`app/llm_client.py`、`app/services/event_reporter.py`、`app/services/event_understanding.py` |
| 时间线与设置前端 | `static/js/event-monitor/`、`static/css/event-monitor.css`、`static/css/event-monitor/` |
| 行为保护测试 | `tests/` |

## 文档与实验

- Phase 4 流程图：`docs/phase4-logic-flow.*`
- 身份和人脸质量：`docs/人脸质量与身份融合逻辑.md`
- 云部署：`docs/cloud-deploy/`、`docs/AZURE_DEPLOY.md`
- 糊脸与身份实验：`experiment/糊脸消融实验/`
- MEVID公共实验工具：`experiment/糊脸消融实验/common/mevid_eval_common.py`
- 超分门控A/B/C：`experiment/糊脸消融实验/超分实验/scripts/run_superres_gate.py`
- actor check-in固定Gallery超分schema-v3：`experiment/糊脸消融实验/超分实验/scripts/run_checkin_superres_abc.py`

## 范围规则

1. 默认搜索限制在本目录。
2. 运行数据、模型权重、`.venv` 和输出文件不属于源码。
3. 旧 `track_fusion.py` 只存在于 `feature/monitor-v1` 分支。
4. 后续摄像头实时输入作为本应用的新输入适配器，不再复制第三套应用。

## 人脸调用顺序

```text
SCRFD检测与5点对齐
→ CR-FIQA + 尺寸/姿态/拉普拉斯质量评估
→ 输出eligibility=direct / recoverable / unusable
→ direct使用原图；recoverable仅在超分成功后可匹配
→ 低于FACE_RECOVERABLE_MIN_SIZE直接拒绝
→ 仅match_ready=true时调用人脸库
→ 查询人脸向量库
```

CR-FIQA内部512维特征仅用于质量回归；产品ArcFace/AdaFace的512维特征仅用于身份匹配，两者
不能混入同一向量库。
