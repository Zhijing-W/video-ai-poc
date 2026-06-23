# 代码重构计划 - AI友好架构

## 🎯 重构目标
1. **模块化**：每个文件 < 400 行
2. **可读性**：AI 看文件名就知道改哪里
3. **可维护**：修改一个功能只需读 1-3 个文件

---

## 📁 新目录结构

```
video-understanding-poc/
├── app/
│   ├── main.py                      # FastAPI 主入口（仅路由注册）~50 行
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── analyze.py               # /analyze-frame 分析路由 ~100 行
│   │   ├── detect.py                # /detect YOLO 检测路由 ~80 行
│   │   ├── compare.py               # /compile-target + /cruise-frame 比对路由 ~120 行
│   │   └── session.py               # /monitor-sessions 会话管理 ~60 行
│   ├── services/
│   │   ├── __init__.py
│   │   ├── yolo_service.py          # YOLO 检测服务（整合现有 detect 相关）~150 行
│   │   ├── llm_service.py           # LLM 调用服务（整合 llm_client）~200 行
│   │   ├── gate_service.py          # 门控逻辑服务 ~80 行
│   │   └── cruise_service.py        # 巡航裁决服务 ~100 行
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request_models.py        # Pydantic 请求模型 ~100 行
│   │   └── response_models.py       # Pydantic 响应模型 ~100 行
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # 配置管理 ~50 行
│   │   └── state.py                 # 全局状态管理（JOBS, sessions）~40 行
│   ├── video_processor.py           # 保持原样 ~105 行
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py           # 图片处理工具 ~60 行
│       └── color_utils.py           # 颜色识别工具 ~40 行
│
├── static/
│   ├── css/
│   │   ├── variables.css            # CSS 变量和主题 ~60 行
│   │   ├── layout.css               # 布局样式（grid、flex）~150 行
│   │   ├── components.css           # 组件样式（按钮、输入框等）~120 行
│   │   ├── dashboard.css            # 监控模式样式 ~200 行
│   │   └── technical.css            # 技术模式样式（架构图）~250 行
│   ├── js/
│   │   ├── main.js                  # 主入口和初始化 ~100 行
│   │   ├── state.js                 # 前端状态管理 ~80 行
│   │   ├── api.js                   # API 调用封装 ~150 行
│   │   ├── ui/
│   │   │   ├── video-controller.js  # 视频控制（摄像头、上传）~150 行
│   │   │   ├── monitor-mode.js      # 监控模式 UI 逻辑 ~200 行
│   │   │   ├── technical-mode.js    # 技术模式 UI 逻辑 ~180 行
│   │   │   ├── mode-switcher.js     # 模式切换逻辑 ~80 行
│   │   │   └── render-engine.js     # 渲染引擎（统一更新 DOM）~220 行
│   │   ├── monitoring/
│   │   │   ├── ticker.js            # 定时抓帧逻辑 ~120 行
│   │   │   ├── analyzer.js          # 分析流程编排 ~150 行
│   │   │   ├── cruise-handler.js    # 巡航处理逻辑 ~100 行
│   │   │   └── gate-handler.js      # 门控处理逻辑 ~60 行
│   │   ├── visualization/
│   │   │   ├── yolo-boxes.js        # YOLO 框绘制 ~100 行
│   │   │   ├── flow-diagram.js      # 流程图动画 ~120 行
│   │   │   └── stats-display.js     # 统计数据展示 ~80 行
│   │   └── utils/
│   │       ├── dom-helpers.js       # DOM 操作工具 ~60 行
│   │       ├── canvas-utils.js      # Canvas 工具 ~50 行
│   │       └── format-utils.js      # 格式化工具 ~40 行
│
├── templates/
│   ├── monitor.html                 # 主页面（仅 HTML 骨架）~400 行
│   ├── components/
│   │   ├── topbar.html              # 顶栏组件 ~30 行
│   │   ├── dashboard-view.html      # 监控视图 HTML ~150 行
│   │   └── technical-view.html      # 技术视图 HTML ~200 行
│
├── docs/
│   ├── CODE_MAP.md                  # 🗺️ 代码地图（AI 必读）
│   ├── API.md                       # API 接口文档
│   └── COMPONENTS.md                # 前端组件文档
│
└── scripts/
    └── make_arch_diagram_phase4.py  # 保持原样
```

---

## 🗺️ CODE_MAP.md 设计（AI 索引文件）

这是关键！每次修改前，AI 先读这个文件，就知道该读哪些文件。

```markdown
# 代码地图 - 快速定位修改位置

## 🎯 我要修改什么？→ 应该改哪些文件？

### 前端功能修改

| 功能需求 | 需要修改的文件 | 说明 |
|---------|---------------|------|
| 修改视频显示样式 | `static/css/dashboard.css` (L50-80) | 视频容器样式 |
| 修改视频显示样式 | `static/js/ui/video-controller.js` (L20-60) | 视频控制逻辑 |
| 添加新的监控按钮 | `templates/components/dashboard-view.html` (L30-50) | HTML 结构 |
| 添加新的监控按钮 | `static/css/components.css` (L80-100) | 按钮样式 |
| 添加新的监控按钮 | `static/js/ui/monitor-mode.js` (L150-180) | 按钮事件 |
| 修改 YOLO 框颜色 | `static/js/visualization/yolo-boxes.js` (L60-90) | 绘制逻辑 |
| 修改采样间隔逻辑 | `static/js/monitoring/ticker.js` (L40-80) | 定时器控制 |
| 修改画面理解显示 | `static/js/ui/render-engine.js` (L120-160) | DOM 更新 |
| 修改统计数据显示 | `static/js/visualization/stats-display.js` (全文) | 统计逻辑 |
| 修改模式切换逻辑 | `static/js/ui/mode-switcher.js` (全文) | 模式切换 |
| 修改架构图样式 | `static/css/technical.css` (L100-250) | 流程图样式 |
| 修改架构图动画 | `static/js/visualization/flow-diagram.js` (全文) | 流程图动画 |

### 后端功能修改

| 功能需求 | 需要修改的文件 | 说明 |
|---------|---------------|------|
| 修改分析接口逻辑 | `app/routers/analyze.py` (全文) | 分析路由 |
| 修改 YOLO 检测 | `app/services/yolo_service.py` (全文) | YOLO 服务 |
| 修改 LLM 提示词 | `app/services/llm_service.py` (L80-150) | LLM 服务 |
| 修改门控策略 | `app/services/gate_service.py` (全文) | 门控服务 |
| 修改巡航策略 | `app/services/cruise_service.py` (全文) | 巡航服务 |
| 添加新接口 | `app/routers/` (新建文件) + `app/main.py` (注册路由) | 路由和服务 |
| 修改请求模型 | `app/models/request_models.py` | 请求验证 |
| 修改响应模型 | `app/models/response_models.py` | 响应格式 |

---

## 📦 核心模块说明

### 前端核心流程

```
用户操作 → main.js (初始化)
    ↓
video-controller.js (开启摄像头/上传视频)
    ↓
ticker.js (定时抓帧) → canvas-utils.js (帧处理)
    ↓
api.js (调用后端) → analyzer.js (流程编排)
    ↓
render-engine.js (统一更新 DOM)
    ├→ monitor-mode.js (更新监控模式 UI)
    ├→ technical-mode.js (更新技术模式 UI)
    ├→ yolo-boxes.js (绘制检测框)
    └→ stats-display.js (更新统计数据)
```

### 后端核心流程

```
请求 → main.py (路由分发)
    ↓
analyze.py (路由处理)
    ↓
gate_service.py (门控判断)
    ├→ 跳过 → 返回 YOLO 结果
    └→ 通过 → llm_service.py (调用 LLM)
        ↓
cruise_service.py (巡航模式) or 直接返回
    ↓
response_models.py (格式化响应) → 返回前端
```

---

## 🔧 常见修改场景速查

### 场景1：修改监控模式的视频显示比例（4:3 → 16:9）
**需要读的文件（按顺序）：**
1. `static/css/dashboard.css` (L50-80) - 修改 `.dash-video-container` 的 `aspect-ratio`
2. **不需要读其他文件**

### 场景2：修改采样间隔的最小值（从 200ms → 500ms）
**需要读的文件：**
1. `static/js/monitoring/ticker.js` (L10-20) - 修改 `MIN_INTERVAL` 常量
2. `templates/components/dashboard-view.html` (L60) - 修改 input 的 `min` 属性

### 场景3：添加新的 LLM 模型选项
**需要读的文件：**
1. `templates/components/dashboard-view.html` (L70-80) - 添加 `<option>`
2. `app/services/llm_service.py` (L20-40) - 添加模型配置
3. **不需要读其他文件**

### 场景4：修改 YOLO 框绘制逻辑（改变颜色/粗细）
**需要读的文件：**
1. `static/js/visualization/yolo-boxes.js` (全文 ~100 行)
2. **不需要读其他文件**

---

## 💡 AI 使用指南

### 给 AI 的提示词模板

```
我想修改【功能描述】。

根据 CODE_MAP.md，我需要修改的文件是：
- 文件A：【具体修改内容】
- 文件B：【具体修改内容】

请帮我：
1. 先读取这些文件的相关部分
2. 确认修改方案
3. 进行修改
```

### 最佳实践

1. **总是先读 CODE_MAP.md**
2. **只读需要的文件**，不要一次加载整个项目
3. **一次只改一个功能**
4. **修改后更新 CODE_MAP.md 的行号**（如果结构变化）

---

## 📝 文件依赖关系

### 前端依赖树（核心）
```
main.js (入口)
├── state.js (全局状态)
├── api.js (后端通信)
└── ui/
    ├── mode-switcher.js
    │   ├── monitor-mode.js
    │   └── technical-mode.js
    └── video-controller.js
        └── monitoring/
            ├── ticker.js
            ├── analyzer.js
            └── gate-handler.js
                └── render-engine.js
                    ├── yolo-boxes.js
                    ├── stats-display.js
                    └── flow-diagram.js
```

### 后端依赖树（核心）
```
main.py (入口)
└── routers/
    ├── analyze.py
    ├── detect.py
    ├── compare.py
    └── session.py
        └── services/
            ├── yolo_service.py
            ├── llm_service.py
            ├── gate_service.py
            └── cruise_service.py
                └── models/
                    ├── request_models.py
                    └── response_models.py
```

---

## 🚀 重构优先级

### Phase 1（立即进行）- 前端拆分 CSS
- [x] 拆分 CSS（最容易，收益最大）
- [x] 创建 CODE_MAP.md

### Phase 2（本周）- 前端拆分 JS
- [x] 拆分 JS 核心模块（state, api, video-controller）
- [x] 拆分 UI 模块（monitor-mode, technical-mode）
- [x] 拆分可视化模块（yolo-boxes, flow-diagram）

### Phase 3（下周）- 后端拆分
- [x] 拆分路由（routers/）
- [x] 拆分服务（services/）
- [x] 拆分模型（models/）

---

## 📊 重构前后对比

| 指标 | 重构前 | 重构后 | 改善 |
|-----|--------|--------|------|
| 最大文件行数 | 1893 | 400 | ↓ 79% |
| 修改时需读行数 | ~1893 | ~200 | ↓ 89% |
| AI 响应时间 | 慢（大量上下文）| 快（精准定位）| ↑ 5x |
| 并行开发能力 | 低（冲突多）| 高（模块独立）| ↑ 3x |
| 代码可读性 | 😰 | 😊 | ++ |

---

## ⚠️ 注意事项

1. **保持向后兼容**：重构期间确保旧代码仍可运行
2. **逐步迁移**：不要一次性重构所有内容
3. **测试驱动**：每拆分一个模块就测试一次
4. **文档同步**：修改代码的同时更新 CODE_MAP.md
```
