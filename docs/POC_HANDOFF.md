# Video AI / Event Monitor POC 交接手册

> **唯一产品基线**：`origin/feature/event-understanding`，产品提交
> `fc2dae1e713f202228a24acb449f4ca6a4f01aba`（`Expose tracker-only appearance mode`）。
> 本文档提交仅增加交接资料，不改变产品行为。`origin/main` 仍停在
> `4b8b07cf93e0e19fbc9f08acfad2286bfd9f8d58`，明显落后；在
> `feature/event-understanding` 合并进 `main` 前，所有新工作都必须从前者分支。

## 1. POC 做什么

Event Monitor 把上传视频或本地样片转换为带身份证据的事件时间线：高频视觉链路负责检测、
跟踪和“谁在何处”，语义分窗后再把身份、空间、OCR、物体和关键帧证据交给多模态模型解释
“发生了什么”，最后生成逐窗报告和整段总结。

实际运行链路如下：

1. [`app/routers/event_monitor.py`](../app/routers/event_monitor.py) 接收一次运行的文件、样片和覆盖参数，并用进程内锁串行执行 POC 请求。
2. [`app/event_analysis_pipeline.py`](../app/event_analysis_pipeline.py) 组织高频跟踪、语义抽帧、事件会话和结束阶段。
3. [`app/pipeline/`](../app/pipeline/) 负责会话、分窗、空间和物体上下文；检测/跟踪及身份提供者产生结构化证据。
4. [`app/identity/`](../app/identity/) 完成 Gallery 匹配、证据选择、跨轨迹归并和身份置信度；关键帧、OCR、步态是可选证据源。
5. [`app/services/event_reporter.py`](../app/services/event_reporter.py) 与 [`app/services/prompt_compaction.py`](../app/services/prompt_compaction.py) 把可信任务指令和不可信证据分离后调用 Foundry/Azure OpenAI。
6. 结果写入 run 目录并由 [`templates/event-monitor.html`](../templates/event-monitor.html) 与 [`static/js/event-monitor/`](../static/js/event-monitor/) 渲染。

先看[代码地图](../CODE_MAP.md)，再看[代码证据对齐的 POC 架构图](poc-architecture.svg)。
`phase4-logic-flow.*` 仅为兼容别名，不是另一套架构。

## 2. 改哪里

| 子系统 | 主要修改点 |
|---|---|
| 页面路由与 API | [`app/main.py`](../app/main.py)、[`app/routers/event_monitor.py`](../app/routers/event_monitor.py)、[`app/routers/health.py`](../app/routers/health.py) |
| 主编排、会话与分窗 | [`app/event_analysis_pipeline.py`](../app/event_analysis_pipeline.py)、[`app/pipeline/session.py`](../app/pipeline/session.py)、[`app/pipeline/windowing.py`](../app/pipeline/windowing.py)、[`app/pipeline/spatial_context.py`](../app/pipeline/spatial_context.py)、[`app/pipeline/object_context.py`](../app/pipeline/object_context.py) |
| 检测与 MOT | [`app/detector.py`](../app/detector.py)、[`app/tracker.py`](../app/tracker.py)、[`app/core/config.py`](../app/core/config.py) |
| Tracker 专用外观关联 | [`app/tracker.py`](../app/tracker.py) 的 `_AppReIDEncoder`；只在 `botsort_reid` 中使用轻量 OSNet，配置入口是 `TRACK_BACKEND`/请求 `track_backend`。`TRACK_REID_BACKEND` 当前只接受 `osnet` |
| 最终人物身份 Body ReID / Gallery | [`app/body_reid.py`](../app/body_reid.py)、[`app/identity/body_reid_backends/`](../app/identity/body_reid_backends/)、[`app/identity/embedding_gallery.py`](../app/identity/embedding_gallery.py)、[`app/identity/gallery_seed.py`](../app/identity/gallery_seed.py)、[`app/body_gallery.py`](../app/body_gallery.py) |
| 人脸、FIQA、超分 | 兼容编排入口 [`app/face.py`](../app/face.py)；实现位于 [`app/identity/face/`](../app/identity/face/)，超分注册表为 [`super_resolution.py`](../app/identity/face/super_resolution.py)，后端在 [`superres_backends/`](../app/identity/face/superres_backends/) |
| 步态、OCR、关键帧 | [`app/gait.py`](../app/gait.py)、[`app/pose.py`](../app/pose.py)、[`app/ocr.py`](../app/ocr.py)、[`app/keyframe.py`](../app/keyframe.py) |
| 身份归并与 grounding | [`app/identity/resolution.py`](../app/identity/resolution.py)、[`app/identity/identity_confidence.py`](../app/identity/identity_confidence.py)、[`app/identity/identity_context.py`](../app/identity/identity_context.py)、[`app/services/identity_grounding.py`](../app/services/identity_grounding.py) |
| Foundry 路由与事件报告 | [`app/openai_client.py`](../app/openai_client.py)、[`app/llm_client.py`](../app/llm_client.py)、[`app/services/llm_models.py`](../app/services/llm_models.py)、[`app/services/llm_catalog.py`](../app/services/llm_catalog.py)、[`app/services/event_reporter.py`](../app/services/event_reporter.py)、[`app/services/event_chat.py`](../app/services/event_chat.py) |
| 前端设置与结果 | [`templates/event-monitor.html`](../templates/event-monitor.html)、[`static/js/event-monitor/`](../static/js/event-monitor/)、[`static/css/event-monitor.css`](../static/css/event-monitor.css)、[`static/css/event-monitor/`](../static/css/event-monitor/) |
| 行为与契约测试 | [`tests/`](../tests/) |
| 模型资产 | [`models/manifest.json`](../models/manifest.json)、[`models/README.md`](../models/README.md)、[`scripts/download_models.py`](../scripts/download_models.py)；二进制只放 `MODEL_ROOT` |
| Azure、容器与 Helm | [`infra/`](../infra/)、[`charts/video-poc/`](../charts/video-poc/)、[`Dockerfile.cpu`](../Dockerfile.cpu)、[`Dockerfile.gpu.base`](../Dockerfile.gpu.base)、[`Dockerfile.gpu`](../Dockerfile.gpu)、[Azure 部署说明](AZURE_DEPLOY.md)、[单 VM 部署说明](cloud-deploy/README.md) |

**不要混淆两套外观特征。** Tracker 专用 OSNet 只帮助 BoT-SORT 在相邻帧遮挡/交叉时维持
`track_id`；最终身份路径独立选择 DIFFER、CLIP-ReID 等后端，并与命名 Gallery 做跨轨迹/
跨镜头匹配。两者的模型定位、调用目的、指标和配置入口都不同。

## 3. 行为安全默认值与覆盖点

| 策略 | 默认与边界 | 显式覆盖点 |
|---|---|---|
| 最终 Body ReID | `differ`；GPU、低并发、精度优先 POC 的冻结默认。CLIP-ReID 是延迟敏感替代项 | [`.env.example`](../.env.example) 的 `REID_BACKEND`；API/UI 的 `reid_backend`。只有显式 `auto` 可按 OSNet → ResNet50 → coarse 回退，显式后端加载失败必须报错 |
| Tracker 外观 | `TRACK_BACKEND=botsort` 时不启用外观编码；显式 `botsort_reid` 才使用 Tracker 专用 OSNet | `.env` 的 `TRACK_BACKEND`、本次运行的 `track_backend`；不要把 `REID_BACKEND` 接到 tracker |
| 人脸超分 | `FACE_SUPERRES=off`；不能因“视觉更清楚”推断身份更准确 | `.env` 或本次运行 `face_superres`；CodeFormer 另有 `face_codeformer_fidelity`，默认 `1.0`（身份优先） |
| 可选证据 | 页面默认启用 Body，Face/Gait/OCR/Object 默认不随每次请求启用 | API/UI 的 `with_*` 开关；提供者缺失时必须返回明确警告/错误，不得伪装已运行 |
| 未命名身份 | `IDENTITY_AUTO_ENROLL_UNKNOWN=false`、`IDENTITY_MERGE_UNNAMED_TRACKS=false`，宁可碎片化也不冒险误合并 | 仅在理解误合并风险并补齐测试后改 `.env`/配置 |
| 采样与状态 | `EVENT_TRACKING_FPS=15` 与语义 `fps` 分离；请求串行，tracker/gallery 按 session 隔离 | `.env`、样片 preset 或请求参数；改动必须同时验证时间制 buffer、帧上限和语义覆盖 |
| LLM | `EVENT_ANALYSIS_MODEL=auto`、`EVENT_CHAT_MODEL=auto` 只在服务端已发现且允许的 deployment 中路由 | `.env` 的 Foundry deployment/model alias 或本次运行模型选择；密钥和资源 ID不得返回浏览器 |

请求级覆盖由 [`static/js/event-monitor/settings.js`](../static/js/event-monitor/settings.js) 组装，
由 [`app/routers/event_monitor.py`](../app/routers/event_monitor.py) 校验并在单次运行上下文中生效，
不会回写 `.env`。新增开关时不要只改页面。

## 4. 本地环境、模型和数据边界

- 环境与启动以 [README](../README.md) 为准：Python 3.11/3.12、`ffmpeg`、`requirements.txt`；测试再安装 `requirements-dev.txt`。
- 从 [`.env.example`](../.env.example) 复制本地 `.env`。不要提交 endpoint 密钥、客户数据、运行输出或模型权重。
- 模型清单看 [`models/manifest.json`](../models/manifest.json)，准备方式和许可证边界看 [`models/README.md`](../models/README.md)。运行时只从 `MODEL_ROOT` 懒加载，不应联网隐式下载。
- 数据目录约定看 [`data/README.md`](../data/README.md)。大型/受许可数据、MEVID、客户视频、embedding cache 和实验输出均留在 Git 外。
- 容器、Azure 和 Helm 只参考 [部署总览](AZURE_DEPLOY.md) 与[当前单 VM 方案](cloud-deploy/README.md)；其中是部署操作手册，不代表资源当前仍在运行。

## 5. 现有验证入口

先安装 `requirements-dev.txt`，再按改动选择最小分组；这些命令均使用仓库现有 pytest：

```powershell
# 路由、主契约、采样/跟踪门控
python -m pytest tests/test_router_and_app.py tests/test_pipeline_contract.py tests/test_tracking_rate_and_gates.py

# 最终 Body ReID、默认策略、Gallery 和冻结实验协议
python -m pytest tests/test_body_reid_backends.py tests/test_reid_product_default.py tests/test_gallery_seed.py tests/test_body_reid_experiment.py

# 人脸超分注册、模型适配和产品接线
python -m pytest tests/test_super_resolution.py tests/test_superres_model_backends.py tests/test_superres_product_integration.py

# 前端设置/结果契约
python -m pytest tests/test_event_monitor_render.py

# Foundry 路由、报告上下文和紧凑证据
python -m pytest tests/test_foundry_models_and_chat.py tests/test_llm_catalog.py tests/test_prompt_compaction.py
```

文档变更至少运行 `git diff --check` 并逐一验证相对链接；不要为纯文档变更启动完整模型测试。

## 6. 实验与报告

- 人脸超分阶段结论：[正式阶段报告](../experiment/face_blur_ablation/super_resolution_stage_report.md)。
- 超分实验设计与可运行实现：[`experiment/face_blur_ablation/super_resolution/`](../experiment/face_blur_ablation/super_resolution/)；固定 Gallery 的 schema-v3 实现在 [`checkin_superres/`](../experiment/face_blur_ablation/super_resolution/checkin_superres/)。
- Body ReID 冻结协议、命令和公平性约束：[实验计划](../experiment/body_reid/experiment_plan.md)。
- Body ReID 协议/运行/汇总代码：[`experiment/body_reid/body_reid_experiment/`](../experiment/body_reid/body_reid_experiment/) 和 [`experiment/body_reid/scripts/`](../experiment/body_reid/scripts/)。
- 正式数据 manifest、模型权重、中间 embedding 和本地结果不能伪造，也不应提交；实验代码要求相同 Query/Gallery、独立 train 阈值校准、真实设备/权重/源码 provenance 和禁止静默回退。

超分结论应保持克制：GFPGAN/CodeFormer 可以改善观感或 FIQA，却可能改变身份特征；当前证据不
支持默认开启。Body ReID 的当前产品结论来自冻结协议：DIFFER 作为精度优先默认，CLIP-ReID
作为明确的延迟敏感选项；上线前仍需在目标视频和目标 T4 上测量。

## 7. 常见修改清单

### 新增或更换 Body ReID 后端

1. 在 [`app/identity/body_reid_backends/`](../app/identity/body_reid_backends/) 新增只负责该模型的懒加载适配器，并通过 [`app/body_reid.py`](../app/body_reid.py) 的注册表接入。
2. 在 [`app/core/config.py`](../app/core/config.py)、[`.env.example`](../.env.example)、[`models/manifest.json`](../models/manifest.json)、[`models/README.md`](../models/README.md) 和 [`scripts/download_models.py`](../scripts/download_models.py) 声明本地资产与许可证；启动时不得下载。
3. 让 `/reid-backends`、UI 下拉、请求校验和结果 provenance 使用注册表名称；显式选择失败时不得回退。
4. 增加后端预处理/归一化、默认策略、API/UI 和实验可比性测试；先跑 Body ReID 最小测试组，再按冻结协议比较同一 Query/Gallery。

### 新增或更换人脸超分后端

1. 在 [`app/identity/face/superres_backends/`](../app/identity/face/superres_backends/) 增加暴露 `register(register_backend, settings)` 的适配器，重依赖只能在 loader 首次执行时导入。
2. 在 [`app/identity/face/super_resolution.py`](../app/identity/face/super_resolution.py) 注册，并补齐配置、模型清单、下载、许可证和明确的失败行为。
3. 复用 `/superres-backends` 驱动 UI；若有专属参数，贯通模板、`settings.js`、路由校验和单次上下文恢复。
4. 跑超分最小测试组，并用固定 Gallery/Query 与 resize control 做身份指标实验；不得以视觉样例替代身份结论，也不得直接改为默认开启。

### 新增前端设置

1. 同时更新 [`templates/event-monitor.html`](../templates/event-monitor.html)、[`static/js/event-monitor/settings.js`](../static/js/event-monitor/settings.js)、必要的 API catalog/路由校验和 [`app/event_monitor_i18n.py`](../app/event_monitor_i18n.py)。
2. 明确“留空使用服务端默认”还是“总是发送值”，保证本次运行覆盖不会泄漏到排队请求。
3. 在结果中记录实际生效值，而非只显示用户请求值；补 [`tests/test_event_monitor_render.py`](../tests/test_event_monitor_render.py) 和路由测试。

### 新增 pipeline stage

1. 先确定属于逐帧高频路径、语义分窗路径还是结束阶段，接入 [`app/event_analysis_pipeline.py`](../app/event_analysis_pipeline.py) 与相应 [`app/pipeline/`](../app/pipeline/) 模块，不要复制第三套编排。
2. 明确输入/输出 schema、session reset、失败等级、是否可选及资源缺失时的可见状态；避免全局状态跨 run 泄漏。
3. 若记录耗时，加入互斥的 `stage_timings`；不要把重叠的 per-call 诊断耗时相加冒充总耗时。
4. 更新 grounding/紧凑证据、结果 JSON、前端渲染和契约测试；优先跑路由/主契约最小组。

## 8. 已知限制与不能声称的能力

- 这是串行、低并发的 POC，不是生产服务；没有完成多租户隔离、任务队列、弹性扩缩和 SLA 验证。
- 当前输入是文件/样片；“未来摄像头流”只是适配方向，不能声称端到端实时。
- DIFFER 的默认定位依赖 GPU 精度优先场景；CPU 与高并发性能未验证，不能声称 CPU readiness。
- 不能把局部模型延迟、tracker 外观调用或身份调用简单相加为端到端吞吐。
- Face/Gait/OCR/超分依赖可选模型和许可证；缺资产不等于能力已运行。
- 生成式人脸超分可能损害身份特征，默认关闭；FIQA 当前仅是诊断信号，不是产品身份门控。
- LLM 报告依赖可用的 Foundry/Azure OpenAI deployment；OCR 和模型输入均为不可信证据，不能成为指令。
- Gallery、阈值和实验结果受数据域影响；没有客户域校准、隐私审查和生产安全评估。

## 9. 分支与整合说明

- Gallery 产品化、产品调试、Foundry 路由、审计后的 ReID 报告、GFPGAN 资料和代码证据架构图已整合进 `feature/event-understanding`，但整合后的提交 SHA 与历史功能分支不同。
- 不要从 `5795451`、旧产品调试分支、旧 Gallery/GFPGAN 分支或历史 Copilot worktree 继续开发；它们不是当前事实来源。
- `CODE_MAP.md`、`docs/poc-architecture.*` 和本手册是当前导航入口；`docs/phase4-logic-flow.*` 仅保留兼容。
- `feature/monitor-v1` 只保存旧逐帧监控，不应把其 `track_fusion.py` 或页面复制回当前产品。
- 开工前先确认 `git merge-base HEAD origin/feature/event-understanding` 至少包含产品基线 `fc2dae1e...`；在主分支追平前，不从 `main` 开新工作。

## 10. Azure 退役状态

最近一次**只读**审计显示：

- `videopoc-rg` 仍存在，尚未删除；其中观察到全部 25 个 ARM 资源，包括 GPU VM/磁盘/网络、ACR、4 个 Storage account、监控和 Foundry。
- 未发现资源锁，也未发现额外的 `MC_*` 或相关 POC resource group；GPU VM 已 deallocate，但磁盘、存储、ACR、监控等仍可能计费。
- 退役尚未完成。删除 resource group 前应按组织流程确认数据保留、模型/结果导出、Foundry 与监控依赖及最终审批。
- GitHub Actions variables/secrets，以及 Microsoft Entra OIDC app、service principal 和 federated credentials 不属于 resource group 删除范围，必须单独清理。
- 审计读取上述外部 GitHub/Entra 对象时收到 403，因此不能声称它们不存在或已经清理；需要相应管理员复核并留下删除证据。

## 11. 下一位维护者第一天

1. 拉取并检出 `feature/event-understanding`，核对产品基线至少包含 `fc2dae1e...`，不要从 `main` 或旧 worktree 开工。
2. 阅读本手册、[代码地图](../CODE_MAP.md)、[架构图](poc-architecture.svg)和 [README](../README.md)。
3. 复制 [`.env.example`](../.env.example)，只在本地填写凭据；按 [`models/README.md`](../models/README.md) 检查 `MODEL_ROOT`，不要把权重复制进仓库。
4. 先以 dry-run 和最小测试组理解实际输出契约，再连接 Foundry；记录目标视频、后端、设备和实际生效设置。
5. 修改前确认属于 tracker 外观关联还是最终身份 ReID；修改后按第 7 节贯通配置、API、UI、结果、测试和文档。
6. 若负责退役，先拿到 GitHub 与 Entra 管理权限，补齐 403 无法读取的外部对象审计，再执行资源删除。
