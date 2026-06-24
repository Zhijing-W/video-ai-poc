# 视频 AI PoC — 身份感知 · 多帧事件理解

[English](README.md) ｜ **中文**

一个视频理解概念验证：把监控视频流变成一条**身份感知的事件时间线**。
传统 CV（人脸 + 人形 ReID + 步态）判定画面里**是谁**；多模态大模型再**看图**判定**发生了什么** ——
产出按事件的叙述（谁、何时、做了什么、是否异常），而不只是逐帧描述。

设计原则：**便宜的 CV 在每一帧都跑；昂贵的 LLM 每个事件只看少量精选关键帧。**

> 🔗 仓库：https://github.com/Zhijing-W/video-ai-poc

---

## 它做什么（当前主线：Phase 4）

```
视频流
  → ① 定时密采样（每一帧都跑本地 CV — 便宜）
  → YOLO 检测 + ByteTrack 跟踪（稳定 track_id）
  → 身份提供器（可插拔）：人脸 + 人形 ReID + 步态 → 谁
       · 人脸  — InsightFace ArcFace（脸清晰时最强）
       · 人形  — OSNet-AIN ReID + 主体记忆库（认出回头客）
       · 步态  — SkeletonGait++（OpenGait，GREW 权重）— 无脸/背身时兜底
       · 灰区轨迹缝合 — 把 ByteTrack 断成多段的同一个人并回来
  → 语义事件标注（new_track / track_left / count_change / identity_hit）
  → 流式分窗（一个事件窗 = 一次 LLM 调用；长事件按时长上限切多窗）
  → ② 事件驱动关键帧选择（把几百帧砍到几张）
  → 身份打包（按 subject 合并）→ 喂 LLM 的 grounding 文本
  → ③ 多模态 gpt-4o 跨帧事件理解（身份是外部给定的，动作必须看图）
  → 事件窗时间线 + 可选的跨窗整段总结
```

核心思想就是 **"谁"由外部给定、别重新认人；"做了什么"必须从图像里看出来**：
身份来自传统 CV，并明确告诉 LLM 不要重新猜身份，而是用视觉去理解每个人跨帧到底在做什么。

## 逻辑流

完整的**按计划决策树**（实线 = 已实现，虚线灰 = 预留，带 P1/P2 徽章）：

![Phase 4 逻辑流](video-understanding-poc/docs/phase4-logic-flow.svg)

> 早期 Phase 1–3 运行时树：[`docs/phase3-logic-flow.png`](video-understanding-poc/docs/phase3-logic-flow.png)。

## 怎么跑

```powershell
cd video-understanding-poc

# 1) 建虚拟环境并装依赖
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) 配置 Azure OpenAI
copy .env.example .env
#   编辑 .env：填 AZURE_OPENAI_ENDPOINT / API_KEY / DEPLOYMENT（vision 模型）

# 3a) 事件理解 — 命令行（单段视频端到端）
.\.venv\Scripts\python.exe scripts\event_understand_demo.py --dry-run        # 不调 LLM，验证链路
.\.venv\Scripts\python.exe scripts\event_understand_demo.py                  # 真调 gpt-4o 出事件叙述

# 3b) 或启动 Web 应用
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

- **事件监控页**（Phase 4）：`http://127.0.0.1:8000/eventmonitor` — 选样片 → 事件窗时间线
- **实时监控页**（Phase 1–3）：`http://127.0.0.1:8000/monitor` — 逐帧分析 + 主体认人
- Swagger 在 `/docs`，健康检查 `/health`。

> 步态（SkeletonGait++）需要 OpenGait 仓库 + GREW 权重（约 726MB），二者放在 git 仓库外，
> 路径通过 `app/core/config.py` 的 `GAIT_*` 配置。本机纯 CPU 可跑（慢，但精度与 GPU 相同；
> 上云改 `GAIT_DEVICE=cuda`）。

## 各 Phase 完成情况

| Phase | 内容 | 状态 |
|---|---|---|
| **Phase 1** | LLM-first MVP：视频 → ffmpeg 抽帧 → gpt-4o → 结构化 JSON | ✅ 已落地 |
| **Phase 2** | 成本可控混合：YOLO 检测 + 事件门控 + 智能抽帧，LLM 只在"每事件"调 | ✅ 已落地 |
| **Phase 3** | 逐轨迹识别 + 主体记忆：ByteTrack、ReID FAISS 向量库、多帧融合、评估 | ✅ 已落地 |
| **Phase 4** | **身份感知 · 多帧事件理解 — 当前主线** | 🚧 进行中（核心已落地） |
| **Phase 5** | 基于 Azure 的全链路上云（推流 → GPU 推理 → 拉流） | 📝 设计文档 |

## Phase 4 — 已落地

| 能力 | 模块 | 状态 |
|---|---|---|
| 两段选帧（① 定时密采样 → ② 事件驱动关键帧） | `app/video_processor.py`（`fps=`）、`app/keyframe.py` | ✅ |
| 人脸身份（InsightFace ArcFace 512 维、质量门控、多帧融合） | `app/face.py` | ✅ |
| 人形 ReID 升级为 OSNet-AIN（域泛化，经 boxmot） | `app/reid.py` | ✅ |
| 步态身份 — **SkeletonGait++**（OpenGait，GREW 权重），CPU 加载 + 4096 维向量 | `app/gait.py` | ✅ 核心 |
| 同视频内灰区轨迹缝合（把同一人的断片并回来） | `app/event_pipeline.py`（`_stitch_orphans`） | ✅ |
| 结构化身份打包（按 subject 合并 → LLM grounding） | `app/services/identity_context.py` | ✅ |
| 身份感知跨帧事件理解（身份给定 / 动作看图；429 退避） | `app/services/event_understanding.py` | ✅ |
| 流式事件分窗 + 时长上限；端到端编排 | `app/event_pipeline.py`、`scripts/event_understand_demo.py` | ✅ |
| 跨窗整段事件总结 | `app/services/event_understanding.py`（`summarize_event_windows`） | ✅ |
| 事件监控 Web 页（事件窗时间线 + JSON 导出） | `app/routers/eventmonitor.py`、`/eventmonitor` | ✅ |
| 步态融进身份 · bad-case 评估 · 宠物/车辆/包裹/OCR 提供器 | — | 📝 预留（P1/P2） |

## Phase 1–3 基础（被 Phase 4 复用）

| 能力 | 模块 |
|---|---|
| 多目标跟踪（ByteTrack，稳定 track_id，按 session 隔离） | `app/tracker.py`、`/track` |
| track 门控 / 三时钟解耦（轨迹未变就复用，新主体才调 LLM） | `app/services/track_gate.py` |
| 主体记忆 ReID 向量库（FAISS 余弦 + 开放集登记 + 质量门控 + 负缓存） | `app/gallery.py`、`app/reid.py`、`/identify` |
| 多线索融合 + 最佳帧投票 | `app/track_fusion.py`、`/fusion` |
| 评估体系（准召 + 单位视频 LLM 调用数） | `scripts/eval_phase3.py` |

## 分支结构

| 分支 | 用途 |
|---|---|
| `feature/event-understanding` | 🚩 **主线** —— 身份感知事件理解（活跃开发） |
| `snapshot/baseline-phase1-3` | 🧊 冻结快照 —— Phase 1–3 还原点（保留备份） |
| `main` | 集成主干 |

## 文档

- `video-understanding-poc/CODE_MAP.md` —— 代码地图（功能 → 文件索引）
- `docs/phase/` —— Phase 1–5 设计文档
- `video-understanding-poc/docs/phase4-logic-flow.svg` —— Phase 4 决策树

---

> 🧭 Phase 4 口诀：**身份由传统 CV 给定；大模型看图，把故事讲出来。**
> 🧭 成本口诀：便宜模型扛量，跟踪 + 记忆扛重复，LLM 每个事件只看少量关键帧。
