# 视频 AI PoC

[English](README.md) ｜ **中文**

一个视频理解概念验证：**便宜的 CV 扛量、跟踪复用扛重复、向量库记住主体、LLM 只在必要时调** —— 既识得准、又花得省。

> 🔗 仓库：https://github.com/Zhijing-W/video-ai-poc

---

## 运行时逻辑流（当前已实现）

![运行时逻辑树](video-understanding-poc/docs/phase3-logic-flow.png)

**功能开关驱动的决策树（多分支 + 路径合并）：**

![决策树](video-understanding-poc/docs/phase3-decision-tree.png)

## 各 Phase 完成情况

| Phase | 内容 | 状态 |
|---|---|---|
| **Phase 1** | LLM-first MVP：视频 → ffmpeg 抽帧 → gpt-4o → 结构化 JSON | ✅ 已落地 |
| **Phase 2** | 成本可控混合：YOLO 检测 + 事件门控 + 智能抽帧，LLM 只在"每事件"调 | ✅ 已落地 |
| **Phase 3** | 逐轨迹识别 + 主体记忆：ByteTrack 跟踪、ReID FAISS 向量库、多帧融合、实时认人集成、评估脚本 | ✅ 已落地 |
| **Phase 4** | 客户对齐 · 身份感知多帧事件理解（人脸 + 人形 + 步态 → 身份 → 事件理解） | 📝 设计文档 |
| **Phase 5** | 基于 Azure 的全链路上云（推流 → AML 推理 → 拉流） | 📝 设计文档 |

## Phase 3 已落地的关键能力

| 能力 | 模块 |
|---|---|
| 多目标跟踪（ByteTrack，稳定 track_id，按 session 隔离） | `app/tracker.py`、`/track` |
| 三时钟解耦 / track 门控（轨迹未变就复用结论，新主体才调 LLM） | `app/services/track_gate.py` |
| 细粒度感知 v1（YOLO-Pose 躯干区取色，修颜色误判） | `app/pose.py` |
| 主体记忆 ReID 向量库（FAISS 余弦 + 开放集登记 + 质量门控 + 负缓存） | `app/gallery.py`、`app/reid.py`、`/identify` |
| 多线索融合 + 最佳帧投票（时序/ReID/颜色/位置，人脸留槽） | `app/track_fusion.py`、`/fusion` |
| 实时认人集成（认出回头客、跨 track 复用、前端展示） | `app/services/identity_integration.py` |
| 评估体系（识别准召 + 单位视频 LLM 调用数，证明省钱没掉精度） | `scripts/eval_phase3.py` |

## 快速开始

```powershell
cd video-understanding-poc

# 1) 建虚拟环境并装依赖
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) 配置 Azure OpenAI
copy .env.example .env
#   编辑 .env：填 AZURE_OPENAI_ENDPOINT / API_KEY / DEPLOYMENT（vision 模型）

# 3) 启动
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

打开 `http://127.0.0.1:8000/`（实时监控页）—— Swagger 在 `/docs`，健康检查 `/health`。

## 分支结构

| 分支 | 用途 |
|---|---|
| `main` | 集成主干 |
| `snapshot/baseline-phase1-3` | 🧊 冻结快照 —— 修改 task 前的还原点 |
| `feature/event-understanding` | 🚧 正在开发 —— 事件理解新方向 |

## 文档

- `docs/phase/` —— Phase 1–5 设计文档
- `assets/` —— 各 Phase 架构图

---

> 🧭 口诀：便宜模型扛量，跟踪复用扛重复，向量库扛记忆，LLM 只裁灰区。
