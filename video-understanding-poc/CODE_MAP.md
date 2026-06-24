# 🗺️ 代码地图 - AI 快速索引

> **当前状态：已完成前端/后端分层拆分。**
> 默认阅读顺序：**CODE_MAP.md → 对应模块文件 → 精准修改**。

---

## 📍 当前代码结构（已拆分）

| 区域 | 入口文件 | 说明 |
|---|---|---|
| 前端页面 | `templates/monitor.html` | 仅保留 HTML 结构 + 少量历史弹窗脚本，CSS/JS 已外链 |
| 前端样式 | `static/css/*.css` | 按 variables / layout / components / dashboard / technical 拆分 |
| 前端逻辑 | `static/js/main.js` | 入口文件，初始化各模块 |
| 后端入口 | `app/main.py` | 仅创建 FastAPI、挂载 `/static`、注册路由 |
| 后端路由 | `app/routers/*.py` | API 分路由 |
| 后端服务 | `app/services/*.py` | YOLO / LLM / 门控 / 巡航业务逻辑 |
| 后端模型 | `app/models/*.py` | 请求/响应模型 |
| 运行时配置 | `app/core/*.py` | 路径、环境变量、内存状态 |

---

## 🎯 功能 → 文件快速映射

### 前端功能定位

| 我要改什么 | 读取这些文件 |
|---|---|
| 修改监控页 HTML 结构 | `templates/monitor.html` |
| 修改全局主题/颜色/动画 | `static/css/variables.css` |
| 修改页面布局/顶栏/响应式 | `static/css/layout.css` |
| 修改按钮/输入框/卡片组件 | `static/css/components.css` |
| 修改监控模式 UI | `static/css/dashboard.css` |
| 修改技术模式架构图 UI | `static/css/technical.css` |
| 修改页面初始化逻辑 | `static/js/main.js` |
| 修改摄像头/上传视频逻辑 | `static/js/ui/video-controller.js` |
| 修改监控/技术模式切换 | `static/js/ui/mode-switcher.js` |
| 修改统一渲染/日志/弹窗入口 | `static/js/ui/render-engine.js` |
| 修改采样间隔/开始停止监控 | `static/js/monitoring/ticker.js` |
| 修改分析流程/巡航/审计回填 | `static/js/monitoring/analyzer.js` |
| 修改比对开始/停止/目标编译 | `static/js/monitoring/gate-handler.js` |
| 修改 YOLO 检测框绘制 | `static/js/visualization/yolo-boxes.js` |
| 修改流程图高亮/时钟/状态节点 | `static/js/visualization/flow-diagram.js` |
| 修改统计卡片/比对结果卡 | `static/js/visualization/stats-display.js` |
| 修改全局状态字段 | `static/js/core/state.js` |
| 修改 API 请求字段 | `static/js/core/api.js` |
| 修改工具函数（抓帧、时间、escape） | `static/js/core/utils.js` |
| 修改事件监控页 HTML 结构（Phase 4 · 事件窗时间线） | `templates/eventmonitor.html` |
| 修改事件时间线/事件卡/告警徽章样式 | `static/css/eventmonitor.css` |
| 修改事件页逻辑（选样片/上传/渲染事件窗时间线） | `static/js/eventmonitor/main.js` |

### 后端功能定位

| 我要改什么 | 读取这些文件 |
|---|---|
| 修改 FastAPI 入口/静态资源挂载 | `app/main.py` |
| 修改视频上传、任务状态、结果查询 | `app/routers/video.py` |
| 修改 `/analyze-frame` | `app/routers/analyze.py` + `app/services/gate_service.py` + `app/services/llm_service.py` |
| 修改实时流程里的「认人」叠加（Phase 3 · 连：track→认人→融合） | `app/services/identity_integration.py`（被 `analyze.py` 的 track 路径调用，给 person 检测补 `subject_id` 并产出身份摘要） |
| 修改 Track 门控（逐轨迹复用 · 新主体才调 LLM，Phase 3 · Step 12） | `app/routers/analyze.py`（`_analyze_frame_tracked`）+ `app/services/track_gate.py` |
| 修改 `/detect` | `app/routers/detect.py` + `app/services/yolo_service.py` |
| 修改逐帧 YOLO 推理 / 跟踪共享的检测函数 | `app/detector.py`（`_predict` / `detect_objects`） |
| 修改 ByteTrack 跟踪逻辑 / 会话隔离 / track_id（Phase 3 · Step 11） | `app/tracker.py` + `app/services/tracker_service.py` |
| 修改 `/compile-target` / `/cruise-frame` | `app/routers/compare.py` + `app/services/cruise_service.py` |
| 修改 `/track` 多目标跟踪（Phase 3 · Step 11） | `app/routers/track.py` + `app/tracker.py` + `app/services/tracker_service.py` |
| 修改 `/identify` 主体记忆/ReID 认人（Phase 3 · Step 14） | `app/routers/identify.py` + `app/services/gallery_service.py` + `app/gallery.py`（FAISS 向量库）+ `app/reid.py`（ReID 指纹，可插拔 backend） |
| 修改 YOLO 检测框颜色逻辑 / ReID 指纹后端（osnet[boxmot]/resnet50/coarse） | `app/reid.py`（默认经 boxmot 的 OSNet-AIN 域泛化 ReID；`REID_OSNET_WEIGHTS` 可调） |
| 修改人脸识别分支（Phase 4 · Step 20；已接入 Step 24 事件管线，可选 `--face`） | `app/face.py`（InsightFace 检测+对齐+512维 embedding+质量评估+最佳脸/多帧融合+对到 track_id）；配置 `app/core/config.py` 的 `FACE_*` |
| 修改步态识别分支（Phase 4 · Step 27；第三路身份信号，已接入事件管线 `--gait`） | `app/gait.py`（**SkeletonGait++**/OpenGait，GREW 权重，CPU 建网+载权重+复用其热图预处理+自构张量喂 forward → 4096维步态向量；剪影用 yolov8-seg，姿态用 yolov8-pose；`extract_persons`/`embed_track`）；接线在 `app/event_pipeline.py`（逐帧采集 pose/sil 序列→每 track `embed_track`→步态 gallery→写 `ident['gait']`）；配置 `GAIT_*`（OpenGait 路径/权重在 git 仓库外）。无脸/背身/远景兜底，上云改 `GAIT_DEVICE=cuda` |
| 修改结构化身份打包（Phase 4 · Step 22，喂 LLM 的身份上下文） | `app/services/identity_context.py`（`PersonIdentity` + `format_identity_context`：多源识别结果→LLM grounding 文本；身份外部给定、勿重新认人） |
| 修改身份感知·多帧事件理解 LLM 段（Phase 4 · Step 23 / 3.4，本阶段灵魂） | `app/services/event_understanding.py`（`understand_event`：多帧关键帧+身份上下文→跨帧事件 JSON；WHO 外部给定/勿认人，WHAT 必须看图理解；`summarize_event_windows`：跨窗整段事件总结-纯文本把多窗串成连贯故事，靠 ReID 身份跨窗关联，`EVENT_OVERALL_SUMMARY` 开关）；配置 `EVENT_LLM_*` |
| 修改选帧②/关键帧选择（Phase 4 · Step 25 / 3.3，事件驱动） | `app/keyframe.py`（`select_keyframes`：接收每帧语义事件标注→事件帧必留+每track最佳帧+去重+保时序；**事件由语义信号定义，非 ffmpeg/像素**）；配置 `KEYFRAME_*` |
| 跑/改 身份感知·多帧事件理解 **端到端**（Phase 4 · Step 24，把上面叶子串成一条流） | `app/event_pipeline.py`（`analyze_event_stream`：抽帧①→YOLO+ByteTrack→语义事件标注→ReID认人+灰区轨迹缝合(同视频内并同一人)→流式分窗(活动段+时长上限)→选帧②→身份打包→`understand_event`）+ CLI `scripts/event_understand_demo.py`（`--dry-run` 不调LLM验链路；`--max-window-seconds` 调窗长；`--stitch-thresh` 调缝合阈值(默认 EVENT_STITCH_THRESH=0.45)；默认视频 `data/samples/mixkit_31372.mp4`） |
| 修改事件监控页/接口（Phase 4 · Step 26，身份感知事件时间线 Web 入口） | `app/routers/eventmonitor.py`（`/eventmonitor/samples` 列样片 + `POST /eventmonitor/understand` 同步跑端到端）→ `app/event_pipeline.py`；页面路由 `/eventmonitor` 在 `app/main.py` |
| 修改 `/fusion` 多线索融合/最佳帧投票（Phase 3 · Step 15 / 3.5） | `app/routers/fusion.py` + `app/services/fusion_service.py` + `app/track_fusion.py`（按 track 攒多帧证据：最佳帧+投票+多线索融合） |
| 跑 Phase 3 识别+省钱评估（Phase 3 · Step 19，证明省钱没掉精度） | `scripts/eval_phase3.py`（合成自检 / `--manifest` 真实数据；精度·召回·ID切换 + 单位视频 LLM 调用省比 + 逐帧vs融合对比） |
| 修改历史监控会话接口 | `app/routers/session.py` |
| 修改环境变量/目录路径 | `app/core/config.py` |
| 修改内存任务表/会话缓存 | `app/core/state.py` |
| 修改请求模型 | `app/models/request_models.py` |
| 修改响应模型 | `app/models/response_models.py` |
| 修改图片编解码/裁图工具 | `app/utils/image_utils.py` |
| 修改颜色识别辅助逻辑 | `app/utils/color_utils.py` |
| 修改细粒度取色 / Pose 躯干区（修 Phase 2 颜色误判，Phase 3 · Step 13） | `app/pose.py`（YOLO-Pose 关键点→躯干区）+ `app/services/perception_service.py`（匹配检测框+取色，被 `yolo_service`/`cruise_service` 调用，`POSE_COLOR` 开关） |
| 修改 LLM 原始客户端封装 | `app/llm_client.py` |
| 修改智能抽帧（Step 7：scene+定时兜底） | `app/video_processor.py`（`extract_frames_smart`）+ `app/pipeline.py` |
| 修改末尾总结（整段分析跑完归纳总结） | `app/llm_client.py`（`summarize_events`）+ `app/routers/session.py`（`/summarize`） |
| 修改实时流智能抽帧（mode①：画面没变跳过 + 兜底间隔） | `static/js/monitoring/ticker.js` + `static/js/core/utils.js`（`frameSignature`/`signatureDiff`） |
| 修改整段视频分析（mode②：流式逐帧+末尾总结+归档） | `static/js/monitoring/ticker.js`（`startFullVideoRun`/`finishFullVideoRun`）+ `static/js/monitoring/batch-report.js`（`fetchSummary`/`showSummary`） |
| 修改视频抽帧处理 | `app/video_processor.py`（`extract_frames` 支持 `fps=` 亚秒级密采样，事件理解"选帧①"用） |

---

## 🔍 常见修改场景速查

### 1. 改视频显示比例或实时画面布局
读取：
- `static/css/dashboard.css`
- 如涉及技术模式同时修改：`static/css/technical.css`

### 2. 改采样间隔、开始就抽第一帧、运行时改间隔生效
读取：
- `static/js/monitoring/ticker.js`
- `static/js/core/utils.js`

### 3. 改 YOLO 框颜色、粗细、显示条件
读取：
- `static/js/visualization/yolo-boxes.js`
- 如涉及“显示检测框”开关镜像：`static/js/ui/mode-switcher.js`

### 4. 改“当前画面理解”或事件日志渲染
读取：
- `static/js/ui/render-engine.js`
- `static/js/visualization/stats-display.js`

### 5. 改开始/停止比对、目标编译、巡航模式说明
读取：
- `static/js/monitoring/gate-handler.js`
- `static/js/monitoring/analyzer.js`

### 6. 改 `/analyze-frame` 的门控或 LLM 调用策略
读取：
- `app/routers/analyze.py`
- `app/services/gate_service.py`
- `app/services/llm_service.py`

### 7. 改视频上传、异步任务、结果持久化
读取：
- `app/routers/video.py`
- `app/core/state.py`
- 如涉及 Blob：`app/storage.py`

### 8. 改多目标跟踪 / track_id（Phase 3 · Step 11，有状态 MOT）
读取：
- `app/routers/track.py`（`/track`、`/track/reset` 入口）
- `app/tracker.py`（ByteTrack 核心：per-session 跟踪器 + `track_objects`/`reset_tracker`）
- `app/detector.py`（`_predict` 共享推理；检测与跟踪同源不重复跑 YOLO）
- 阈值/缓冲调参：`app/core/config.py`（`TRACK_*` 环境变量）

### 9. 改 Track 门控 / 逐轨迹复用结论（Phase 3 · Step 12，三时钟解耦）
读取：
- `app/services/track_gate.py`（按活跃轨迹集合决定调 LLM / 复用 / 跳过 + 按 session 缓存结论与省钱统计）
- `app/routers/analyze.py`（`_analyze_frame_tracked`：`track_enabled` 时走 track 门控路径）
- 前端：`static/js/monitoring/gate-handler.js`（payload 注入 `track_enabled`/`session_id`）、
  `static/js/monitoring/analyzer.js`（`track_gate` 分支渲染）、`static/js/monitoring/ticker.js`（开始监控生成 session + `/track/reset`）
- 开关 UI：`templates/monitor.html`（`trackToggle` / `trackToggle-dash`）+ `static/js/ui/mode-switcher.js`（镜像）

### 10. 改细粒度取色 / Pose 躯干区（修 Phase 2 颜色误判，Phase 3 · Step 13）
读取：
- `app/pose.py`（YOLO-Pose 关键点 → 躯干区 `torso_region`，含几何理智门控 + 缺关键点降级）
- `app/services/perception_service.py`（IoU 匹配检测框↔Pose，取躯干色；`person_torso_color` / `person_color_matches`）
- 接入点：`app/services/yolo_service.py`（`enrich_detection_colors` 覆盖 person 颜色）、`app/services/cruise_service.py`（`apply_plan` 比对用 Pose 色）
- 开关/调参：`app/core/config.py`（`POSE_COLOR` / `POSE_MODEL` / `POSE_KPT_CONF`）。返回里 `color_source` 标明取色来源

---

## 📦 关键依赖关系

### 前端依赖树（核心）

```text
main.js
├── core/state.js
├── core/utils.js
├── ui/video-controller.js
├── ui/mode-switcher.js
├── monitoring/ticker.js
│   ├── monitoring/analyzer.js
│   └── monitoring/gate-handler.js
├── ui/render-engine.js
└── visualization/
    ├── flow-diagram.js
    ├── yolo-boxes.js
    └── stats-display.js
```

### 后端依赖树（核心）

```text
main.py
├── routers/video.py
├── routers/analyze.py        # Phase 3 · Step 12：track_enabled 时走 track 门控
│   └── services/track_gate.py（按轨迹集合复用结论）→ app/tracker.py（ByteTrack）
├── routers/detect.py
├── routers/track.py          # Phase 3 · Step 11：有状态 MOT
│   └── services/tracker_service.py → app/tracker.py（ByteTrack）→ app/detector.py（_predict）
├── routers/identify.py        # Phase 3 · Step 14：主体记忆 / ReID 认人
│   └── services/gallery_service.py → app/reid.py（ReID 指纹）+ app/gallery.py（FAISS 向量库）
├── routers/fusion.py          # Phase 3 · Step 15/3.5：多线索融合 + 最佳帧投票
│   └── services/fusion_service.py → app/track_fusion.py（按 track 攒多帧证据再裁决）
├── routers/compare.py
└── routers/session.py
    ├── services/yolo_service.py
    ├── services/llm_service.py
    ├── services/gate_service.py
    ├── services/cruise_service.py
    ├── models/request_models.py
    ├── models/response_models.py
    └── core/state.py
# Phase 4 事件监控页（独立范式）
main.py  /eventmonitor (页面)  +  routers/eventmonitor.py (/eventmonitor/understand, /samples)
    └── app/event_pipeline.py（见下方 Phase 4 端到端）
```

### Phase 4 事件理解端到端（脚本流，非 FastAPI；Step 24）

```text
scripts/event_understand_demo.py  (CLI, --dry-run/--face/--fps)
└── app/event_pipeline.py  analyze_event_stream（编排"流式开关窗"）
    ├── video_processor.extract_frames(fps=)        # 选帧① 定时密采样
    ├── tracker.track_objects → detector._predict   # YOLO + ByteTrack（稳定 track_id）
    ├── reid.embed + gallery.identify_or_enroll      # 人形指纹 + 主体记忆（认人=身份，按 subject 合并）
    ├── face.detect（可选 --face）                    # 人脸指纹（清晰正脸才有用）
    ├── keyframe.select_keyframes                     # 选帧② 事件驱动砍帧
    ├── services/identity_context.format_identity_context  # 身份打包成 grounding 文本
    └── services/event_understanding.understand_event      # 多帧+身份 → 跨帧事件叙述（gpt-4o）
```

---

## ✅ AI 修改规则

1. **先读这个 CODE_MAP，再读目标模块**，不要回退到整页通读。
2. **前端优先只读 1-3 个模块**，例如改按钮样式不需要读 JS。
3. **后端优先从 router 找入口，再进入 service 找业务逻辑。**
4. 如果改 HTML 结构导致 ID/class 变化，要同步检查：
   - `static/js/ui/*.js`
   - `static/css/*.css`
5. 如果新增 API，至少同步修改：
   - `app/routers/*.py`
   - `app/models/*.py`
   - 前端 `static/js/core/api.js`

---

## 📌 当前重构结果

- `templates/monitor.html`：**576 行**（已去掉大段内联 CSS/JS）
- `static/css/`：5 个样式模块
- `static/js/`：13 个前端模块 + 1 个入口
- `app/main.py`：**36 行**
- `app/routers/`：5 个路由模块
- `app/services/`：4 个服务模块
- `app/models/`：2 个模型模块
- `app/core/`：配置与状态拆分完成

> 现在 AI 修改单个功能通常只需要读 **1-3 个文件**，不再需要整页扫描 `monitor.html` 或 `app/main.py`。
