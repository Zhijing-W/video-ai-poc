# 视频理解 Demo · Phase 3 — 逐轨迹识别与主体记忆（track-and-identify 设计草案）

> 🔖 **用途**：Phase 2（成本可控混合架构）用"YOLO 巡逻 + LLM 监工"把成本压到了"每事件"。Phase 3 要回答更难的问题——
> **"摄像头里这个人/物，到底是谁、是不是之前出现过的同一个？"** —— 即从"逐帧看图"升级成 **"逐轨迹认人 + 主体记忆"**。
> ⚠️ **状态**：**本阶段全部为设计草案（规划中），尚未写代码**。本文是把前期多轮设计讨论沉淀成可落地蓝图，供评审与排期。
> 🛠️ **维护约定**：开工实现后，每落地一个机制回来勾选 + 填实际值；架构定稿后新增 `video-understanding-poc/scripts/make_arch_diagram_phase3.py` 生成图并同步本文。
> ⏱️ **最后更新**：2026-06-16
> 🔙 **上一阶段**：Phase 2（成本可控混合架构）见 [`Phase2-成本可控混合架构.md`](Phase2-成本可控混合架构.md)
> 🚀 **下一阶段**：Phase 4（客户对齐 · 身份感知多帧事件理解）见 [`Phase4-客户对齐-身份感知多帧事件理解.md`](Phase4-客户对齐-身份感知多帧事件理解.md)

---

## 架构总览（Phase 3）

> 🚧 **架构图待生成**：定稿后由 `video-understanding-poc/scripts/make_arch_diagram_phase3.py` 输出到 `../../assets/architecture-phase3.{svg,png}`（**本图只画 Phase 3 本阶段新增**：MOT 跟踪 + 主体记忆向量库 + 细粒度感知，不重复 Phase 1/2 上传与门控管线）。

文字版数据流（每帧 → 每轨迹 → 仅事件）：

```
实时帧
  └─① YOLO 检测（每帧，沿用 Phase 2 detector.py）
        └─② MOT 多目标跟踪 ByteTrack/BoT-SORT —— 给每个目标分配稳定 track_id（补"上下帧关系"）
              ├─ 已识别的 track：顺着轨迹复用结论（不再花算力）
              └─ 新出生/低置信 track：
                    └─③ 细粒度感知（task-conditioned，按目标条件只裁需要的部件区）
                          └─④ 提特征 embedding → 查主体记忆向量库（gallery / FAISS）
                                ├─ 命中已知主体（高分）→ 直接认出，复用档案
                                ├─ 灰区（不确定）→ ⑤ open-set 早退裁决；必要时升级 LLM
                                └─ 未命中 → ⑥ 事件驱动调 gpt-4o（送 crop 不送整帧）→ 标注一次 → 登记进向量库
```

---

## 一、为什么要做 Phase 3（一句话）

Phase 2 仍然是**逐帧（per-frame）思维**：每个关键帧独立判断"画面里有什么"。
但真实监控的诉求是**逐主体（per-subject）**：

> **"这是不是刚才那个人？""数据库里登记过的那个人来了没？""他从进门到现在干了什么？"**

逐帧思维答不了这些，因为它**没有"同一个对象跨帧的身份"概念**。Phase 3 的核心转变：

> **从"逐帧看图"（recognize each frame）升级到"逐轨迹认人"（track once, identify per-track, reuse along trajectory）。**

一旦给目标建立稳定身份（track + identity），就能：① 顺着轨迹复用结论（省钱）；② 跨摄像头/跨时间做主体比对（认人）；③ 把"上下帧关系"变成最强先验（更准）。

---

## 二、心智模型：粗筛 → 细筛 → 认人

把它想成**安保流程**，而不是"每张照片都重新看一遍"：

| 角色 | 干什么 | 对应技术 | 频率 |
|---|---|---|---|
| **巡逻兵** | 每帧扫一遍，框出所有人/物 | YOLO 检测（Phase 2 已有） | 每帧 |
| **点名册** | 给每个目标一个稳定编号，跟住它的移动 | MOT 跟踪（ByteTrack/BoT-SORT） | 每帧（轻量） |
| **验明正身** | 只对"新来的/认不准的"做仔细检查 | 细粒度感知 + ReID/人脸 embedding | 每 track 出生时 |
| **档案库** | 把见过的人存指纹，下次秒认 | 主体记忆向量库（gallery） | 命中即复用 |
| **专家会诊** | 实在拿不准才请最贵的专家 | gpt-4o（事件驱动、送 crop） | 仅灰区/新主体 |

> **一句话**：先用便宜的把"量"扛住（检测+跟踪），只在"身份不确定"时才花贵的算力（细粒度+LLM），认过一次就记住、顺着轨迹复用——这就是 **coarse-to-fine（粗筛→细筛）+ track-and-identify（逐轨迹识别）**。

---

## 三、核心机制（逐个讲透）

### 3.1 MOT 多目标跟踪 —— 把"上下帧关系"补上

YOLO 每帧独立检测，**两帧之间不知道"这个框和上一帧哪个框是同一个人"**。MOT（Multi-Object Tracking）就是补这条线：

- **ByteTrack / BoT-SORT**（都有成熟预训练实现，ultralytics 自带）：用 IoU + 运动预测（卡尔曼滤波）+ 外观特征，把跨帧的检测框关联成**轨迹（track）**，给每个目标一个**稳定 `track_id`**。
- **意义**：
  - "上下帧关系"从此显式存在 → 可以问"这个 track 从出现到现在经过了哪些位置/做了什么"。
  - **轨迹连续性是最强先验**：同一个 track 短时间内不会突然变成另一个人，识别只需做一次、其余帧复用。
  - 漏检一两帧也能靠运动预测续上（比逐帧检测稳）。

### 3.2 三时钟解耦 —— 不同任务用不同频率

省钱的关键洞察：**不是所有事都要每帧做**。把流水线拆成三个独立"时钟"：

| 时钟 | 任务 | 频率 | 成本 |
|---|---|---|---|
| **检测时钟** | YOLO 检测 + MOT 关联 | **每帧** | 低（CPU/小 GPU 可扛） |
| **识别时钟** | 细粒度感知 + ReID/人脸提特征 → 查库 | **每个 track 出生时（+ 灰区时重试）** | 中（只对新目标） |
| **理解时钟** | gpt-4o 语义理解 / 比对裁决 | **仅事件（新主体 / 灰区 / 报警条件）** | 高（最省着用） |

> 三个时钟独立运转，互不绑死。绝大多数帧只走最便宜的检测时钟，**贵的识别/理解被"按需"触发**。

### 3.3 细粒度感知（task-conditioned perception）—— 别拿粗框硬判属性

**为什么 Phase 2 颜色/属性老错**：拿整个 `person` 粗框去判"什么颜色衣服"必败——框里混了背景、皮肤、裤子、头发，HSV 主色被污染（黑人肤色被当成"衣服颜色"、橙色提示却判成蓝衣）。

**解法**：先把 person 框**拆成部件区**，只在该看的部位上判该判的属性。全部用**预训练模型、不自训**：

| 技术 | 作用 | 适用 |
|---|---|---|
| **Pose 关键点**（YOLO-Pose/RTMPose） | 用骨架点派生"上半身/下半身/头部"区域 | 快、给衣服/裤子定位 |
| **Human Parsing**（人体解析 mask） | 像素级分出 衣/裤/发/肤/鞋 | 取色最准（在 mask 内算主色） |
| **人脸 landmark** | 定位脸/眼/五官 | 人脸比对、眼镜口罩判断 |
| **PAR**（行人属性识别） | 直接预测 性别/年龄/上下身颜色/背包… | 一站式属性，省事 |
| **开放词表检测**（GroundingDINO / YOLO-World） | 用文字框任意部件（"项链""帽子""红色外套"） | 不受 COCO 80 类限制 |
| **CLIP 零样本** | 对裁出的 crop 判"是不是红色/是不是戴口罩" | 零样本、灵活 |
| **ReID embedding** | 提取"这个人"的外观指纹向量 | 跨帧/跨摄像头认同一人 |

**task-conditioned（按任务触发）**：把 Phase 2 的 `compile-target`（自然语言 → YOLO plan）升级成 **"感知计划 plan"**——LLM 把"找戴口罩的红衣男人"编译成：
`需要 [人脸 landmark→口罩判定] + [上半身 parsing→红色判定] + [PAR→性别]`。
**只触发计划里需要的部件感知，其余不跑** → 省算力。比对人脸就重点裁脸，比对环境就重点裁环境变量。

### 3.4 主体记忆 / ReID 向量库 —— 认过一次就记住

让系统"记住见过的人/物"，下次秒认、跨时间跨摄像头比对。核心是**三档指纹 + 向量库 gallery**：

- **三档指纹（由粗到细，成本递增）**：
  - **L1 颜色直方图**：最便宜，粗筛"明显不是同一个人"。
  - **L2 ReID embedding**：行人重识别向量，扛得住换角度/光照，主力。
  - **L3 人脸 embedding**：最准但需正脸、近景，灰区才上。
- **向量库 gallery**（FAISS / pgvector）：
  - **multi-shot**：每个主体存多张（不同角度/帧）特征，投票更稳。
  - **开放集登记（open-set enrollment）**：见到没登记过的新主体，自动建档登记。
  - **质量门控**：模糊/遮挡/太小的 crop 不入库（避免污染）。
  - **负缓存（negative cache）**：记住"查过、确认不是库里任何人"的，避免反复白查。
- **省钱收益**：同一个人第二次出现，**查库命中即复用档案，完全不调 LLM**；跨摄像头/跨天也能认出。

### 3.5 多线索融合 + 最佳帧投票 —— 别赌单帧

单帧可能糊、可能背身。Phase 3 用**多证据**做裁决：

- **最佳帧选择**：一个 track 的若干帧里，挑最清晰/最正脸的那帧去识别。
- **多帧投票**：对同一 track 的多帧识别结果做投票/加权，压随机错误。
- **多线索融合**：把 时序连续性（最强）+ ReID + 人脸 + 颜色 + 位置 综合打分，而非只信一个。

### 3.6 open-set 置信度早退 —— 只在"灰区"花钱

不是每次识别都要请 LLM。按置信度分三带：

- **高分（确定是 A）** → 直接认出，早退，不调 LLM。
- **极低分（确定都不是库里的）** → 判为新主体，登记，早退。
- **灰区（拿不准）** → **只对这一小撮**升级到细粒度感知 / 多帧投票 / gpt-4o 裁决。

> open-set = 承认"可能是个从没见过的人"，不强行塞进已知类别。**只裁灰区**是省 token 的关键。

---

## 四、LLM token 战术（Phase 3 怎么继续省）

Phase 2 把 LLM 降到"每事件"；Phase 3 在"认人"场景里进一步榨：

1. **事件驱动**：只有 新主体 / 灰区 / 命中报警条件 才触发 gpt-4o。
2. **送 crop 不送整帧**：只把目标的小裁图喂 LLM，token 大降、还更聚焦。
3. **标注一次，之后吃 embedding**：LLM 给某 track 下过结论后，存进向量库，**同一 track / 同一主体再出现直接复用**，不再调 LLM。
4. **track 级缓存**：结论挂在 track_id 上，整条轨迹共享。
5. **批处理**：多个待裁决 crop 攒一批一次请求。

> 口诀：**便宜模型扛量，跟踪复用扛重复，向量库扛记忆，LLM 只裁灰区。**

---

## 五、控制状态机 —— 按"不确定性 × 重要性"分配预算

给整个流水线加一个**调度大脑**：不是平均用力，而是把算力/token 预算**动态分配**给"最不确定且最重要"的目标。

```
对每个活跃 track，算一个优先级分 = 不确定性 × 重要性：
  不确定性高（识别置信度低、刚出生、灰区）→ 多花算力（细粒度 + 多帧 + LLM）
  重要性高（命中报警目标、靠近敏感区）   → 优先排队
  已确定且不重要的 track                 → 降频，纯跟踪复用
预算耗尽时，低优先级 track 排队等下一轮。
```

> 这让系统在**固定成本预算**下，把钱花在刀刃上——既不漏关键目标，也不为无关路人烧 token。

---

## 六、组件与技术栈（逐个讲透）

| # | 组件 | 技术 / 服务 | 作用 | 状态 |
|---|---|---|---|---|
| ① | **MOT 跟踪** | ByteTrack / BoT-SORT（ultralytics 内置） | 给目标稳定 track_id，补上下帧关系 | ✅ 已落地（ByteTrack，`app/tracker.py` + `/track`） |
| ② | **细粒度感知** | YOLO-Pose / Human Parsing / GroundingDINO / YOLO-World / CLIP / PAR | 拆部件区、按计划只裁需要的部位判属性 | 🟢 v1 已落地（YOLO-Pose 躯干区取色修颜色误判：`app/pose.py` + `perception_service.py`）；Parsing/GroundingDINO/CLIP/PAR 待补 |
| ③ | **ReID 提特征** | OSNet / TransReID 等预训练 ReID | 行人外观指纹向量（跨帧/跨摄像头） | ✅ 已落地（`app/reid.py` 可插拔后端：OSNet[torchreid,可选] / ResNet50[默认,2048维] / 颜色直方图[兜底]） |
| ④ | **人脸比对** | 预训练人脸 embedding（如 ArcFace 系） | 灰区精确认人 | 🔵 Phase 3 规划 |
| ⑤ | **主体记忆向量库** | FAISS / pgvector + 质量门控 + 负缓存 | 存指纹、查库复用、开放集登记 | ✅ 已落地（`app/gallery.py` FAISS 余弦库 + multi-shot 投票 + 开放集登记 + 质量门控 + 负缓存，按 session 隔离；`/identify`） |
| ⑥ | **感知计划编译** | LLM（升级 Phase 2 的 compile-target） | 自然语言 → "需要哪些部件感知"的 plan | 🟠 Phase 2 已有雏形，待升级 |
| ⑦ | **控制状态机** | 自研调度逻辑 | 按不确定性×重要性分配预算 | 🔵 Phase 3 规划 |
| ⑧ | **LLM 裁决** | Azure OpenAI gpt-4o（送 crop） | 仅新主体/灰区/报警，标注一次后复用 | ✅ 已改为按 track 触发（Step 12：`track_gate.py` 逐轨迹复用） |

---

## 七、演进路线（Phase 3 实施步骤）

> 沿用前两期"本地先跑通、单点验证再集成"的节奏。建议先把"跟踪+复用"骨架立起来（最大省钱杠杆），再逐步加细粒度与向量库。

```
Step 11  集成 MOT 跟踪（ByteTrack/BoT-SORT）→ 每目标 track_id + 轨迹复用结论     ✅ 已落地（ByteTrack：`app/tracker.py` + `/track`、`/track/reset`；track_id 跨帧稳定、按 session 隔离）
Step 12  三时钟解耦：检测每帧 / 识别每 track 出生 / LLM 仅事件                    ✅ 已落地（track 门控：`app/services/track_gate.py` + `/analyze-frame?track_enabled`；同一组轨迹复用结论，新主体才调 gpt-4o；前端「Track 门控」开关可对比省调用数）
Step 13  细粒度感知 v1：Pose 派生部件区 + parsing mask 取色（修 Phase 2 颜色误判） ✅ 已落地（YOLO-Pose 躯干区取色：`app/pose.py` + `app/services/perception_service.py`，接入 `/detect`·`/analyze-frame`·巡航比对；几何门控保证坏 pose 自动回落，`POSE_COLOR` 可开关。注：pixel 级 human-parsing mask 仍留 v2）
Step 14  ReID embedding + 主体记忆向量库（FAISS）+ 开放集登记 + 质量门控/负缓存    ✅ 已落地（`video-understanding-poc/app/reid.py` 可插拔指纹[osnet/resnet50/coarse] + `app/gallery.py` FAISS 余弦库[multi-shot/开放集登记/质量门控/负缓存]；`/identify`、`/gallery/stats`、`/gallery/reset`，按 session 隔离）
Step 15  最佳帧选择 + 多帧投票 + open-set 置信度早退                              🟡 部分落地（3.5 最佳帧选择+多帧加权投票+多线索融合[时序/ReID/颜色/位置，人脸留槽] ✅ `video-understanding-poc/app/track_fusion.py`、`/fusion/observe|resolve|reset`，按 session/track 隔离；3.6 open-set 早退已在 `gallery.py` 的 hit/grey/new 阈值内）
Step 16  感知计划升级（compile-target → 部件级感知 plan，task-conditioned 触发）  ⬜ 规划
Step 17  人脸比对接入（灰区才上）+ 多线索融合打分                                  ⬜ 规划
Step 18  控制状态机：按不确定性×重要性做预算分配                                  ⬜ 规划
Step 19  评估体系：识别准确率/召回 + 单位视频 LLM 调用数（证明省钱没掉精度）       ✅ 已落地（`video-understanding-poc/scripts/eval_phase3.py`：合成自检 + `--manifest` 真实数据；purity/inverse-purity 精度·召回·F1、ID 切换、单位视频 LLM 调用 vs 每帧 baseline 省比、记忆命中率，并对比"逐帧 vs 融合"证明纠错有效）
```

**建议优先级**：Step 11 ✅ → 12 ✅ → 13 ✅ → 14 ✅（"跟踪复用 + 修颜色 + 向量库认人"已成型），后面再补投票/状态机/评估。

> 🔗 **集成（"连"）已落地**：Step 14(认人) + Step 15(融合) 已接进 Step 12 的实时 `/analyze-frame?track_enabled` 流程——`app/services/identity_integration.py` 每帧对 person 检测查主体记忆 + 融合，给框补 `subject_id`（前端画框显示 `#主体号`、回头客 ♻），并回传身份摘要（记忆库主体数 / 跨 track 回头客命中），前端「主体记忆 · ReID」卡片实时展示。换视频/重开监控时 `/track/reset` 一并清空跟踪/门控/记忆。认人为叠加维度，失败不影响门控主流程。

---

## 八、风险 / 限制 / 待办

1. **算力上台阶**：ReID/parsing/人脸/开放词表检测比 YOLO 重，本地 CPU 跑 PoC 勉强，上量要 GPU 或换 Azure 托管视觉服务（AI Vision / Face / Video Indexer）。CSA 增值点同 Phase 2：**能托管就别自建**。
2. **ReID 跨摄像头难**：换光照/角度/服装会掉点；需 multi-shot + 多帧投票 + 质量门控兜底，别指望单帧单模型。
3. **人脸依赖正脸近景**：监控多是背身/远景/低分辨率，人脸常不可用，所以 **ReID（人形）才是主力、人脸是灰区补充**。
4. **向量库污染风险**：低质 crop 入库会越查越错；质量门控 + 负缓存 + 定期清理必须有。
5. **隐私合规**：人脸/ReID 涉及生物特征与个人数据，落地需明确合规边界（留存期限、授权范围、可关闭）。Demo 阶段用合成/授权数据。
6. **评估缺位**：和 Phase 2 一样，必须建量化指标（识别准确率/召回 + 单位视频 token），否则无法证明"既省钱又没掉精度"。
7. **延续历史风险**：订阅 VM 配额=0（上云受阻）、历史泄露的 OpenAI key / Storage 密钥需轮换。

---

> 维护：本阶段开工后，架构定稿即新增 `video-understanding-poc/scripts/make_arch_diagram_phase3.py` 生成 SVG+PNG 并替换上方"架构总览"占位；每落地一个 Step 回来勾选 + 填实际实现文件。
