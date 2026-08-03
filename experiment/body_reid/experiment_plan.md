# 人形 ReID 对比实验计划

## 1. 这次实验要回答什么

这次实验只比较“看整个人的外观能不能认出身份”，不再研究人脸超分。

系统会把一张人体裁图转换成一串数字特征。登记库中的参考特征称为 **Gallery**，
待识别监控片段的特征称为 **Query**。模型通过比较两串数字的余弦相似度，判断 Query
最像 Gallery 中的哪一个人；如果最高相似度仍不够高，则应拒绝并输出“陌生人”。

首轮比较四个后端：

| 后端 | 本实验中的定位 |
|---|---|
| OSNet-AIN x1.0 / MSMT17 | 当前产品基线 |
| 官方 ViT-CLIP-ReID / MSMT17 | 成熟的 CLIP 人形 ReID |
| `MarketaJu/siglip2-person-description-reid` | 经过人物图像匹配训练的 SigLIP 2 |
| 官方 DIFFER | 专门观察换衣场景是否更稳 |

首轮不使用 MEVID 训练模型参数，而是直接评估外部预训练权重。这属于“跨数据集零样本
特征评测”。可以使用 MEVID 训练人员校准相似度阈值，但不能用正式测试 Query 调模型、
挑图片或调阈值。

---

## 2. 能否复用以前的 Gallery 和 Query

### 2.1 可以复用的内容

可以复用此前 MEVID 多模态实验的**原始数据和官方划分**：

- 数据根目录仍为 MEVID；
- 官方 test 共 54 个身份、1754 条人物轨迹；
- `query_IDX.txt` 指定 Query，其余轨迹作为 Gallery；
- 同时具有 Gallery 和 Query 的可用身份共 52 个；
- 固定随机种子为 `0`；
- 其中 27 个身份进入登记库；
- 剩余 25 个身份不登记，只用于测试陌生人误识。

还应复用此前协议中的固定数量：

- 每个身份最多使用 3 条 Gallery 轨迹；
- 每个身份最多使用 4 条 Query 轨迹；
- 每条轨迹均匀抽取 8 帧；
- Gallery 人体图片继续使用产品的面积、模糊度和长宽比质量门控。

这样 OSNet 的新结果可以与原来的 53.8% Rank-1 基线对应。

### 2.2 不能直接复用的内容

以下内容不能从超分实验直接拿来：

- 超分实验冻结的 316 条人脸 Query；
- 依据人脸清晰度选出的帧；
- 145 个人脸模板；
- ArcFace、GFPGAN 等人脸向量或中间结果；
- 任何模型已经计算好的人体 embedding。

原因是人脸最佳帧不一定是人体最佳帧，而且不同 ReID 模型的向量空间互不兼容。

因此，本实验复用的是**MEVID 身份、轨迹和官方 Gallery/Query 关系**，不是超分实验最终
生成的人脸裁图和特征。

### 2.3 必须新建并冻结人体样本清单

在运行任何模型之前，先生成一份 `frozen_body_samples.csv`。每行至少记录：

- `sample_id`；
- 数据用途：`test` 或 `train_calibration`；
- 人物真值 `person_id`；
- 身份角色：正式 test 使用 `enroll_gallery`、`genuine_query`、
  `imposter_query`；阈值校准另用 `calibration_gallery`、
  `calibration_genuine_query`、`calibration_imposter_query`；
- 官方轨迹编号；
- 摄像头编号；
- 服装编号；
- 原始图片路径及文件哈希；
- 在轨迹中的抽帧序号；
- 面积、模糊度和长宽比；
- 是否允许进入 Gallery，以及被拒绝的原因。

冻结后，四个后端必须读取完全相同的清单。不能看到某个模型的结果后再换图、删图或增加
样本。原始图片、模型权重和中间可视化图片不放入仓库，也不保存到 session artifacts。

---

## 3. 主实验如何设计

### 3.1 协议 P1：官方划分的整体结果

这是论文中的主结果，尽量保持与此前 OSNet 协议一致。

1. 27 个已登记身份只用官方 Gallery 轨迹建库。
2. 这 27 个身份的官方 Query 是“库内人员 Query”。
3. 另外 25 个身份完全不进入 Gallery，只作为“陌生人 Query”。
4. Gallery 只聚合通过质量门控的帧。
5. Query 不因质量差而删除；低质量也是实际产品必须处理的情况。
6. 每帧先提取并归一化向量。
7. 同一轨迹的多帧向量求均值后再次归一化，形成轨迹向量。
8. 同一身份的多条 Gallery 轨迹向量求均值后再次归一化，形成身份模板。
9. Query 轨迹与全部身份模板计算余弦相似度并排序。

主结果固定使用“身份均值模板”，因为当前 OSNet 基线也是这一逻辑。以后可以单独增加
“保留多个 Gallery shot，取最高分或 Top-k 平均”的产品消融，但不能只给某一个模型使用。

### 3.2 协议 P2：严格换衣结果

MEVID 轨迹含有服装编号，因此额外建立一个换衣子集：

- Gallery 与 Query 必须属于同一个人；
- Query 服装编号不能出现在该身份任何一条已通过质量门控的 Gallery 轨迹中；
- 样本是否合格只根据官方元数据确定，不能读取模型相似度；
- 若某身份没有合格的跨衣组合，则在运行模型前从该子集排除；
- 四个后端必须使用相同的身份和轨迹。

这个结果主要回答 DIFFER 是否真的比依赖衣服外观的模型更稳。

### 3.3 协议 P3：同衣诊断结果

为了理解模型为什么成功或失败，再报告同衣子集：

- Query 服装编号至少出现在该身份一条已通过质量门控的 Gallery 轨迹中；
- Query 摄像头不能出现在这些同衣 Gallery 轨迹中；
- 与 P2 使用相同的抽帧、聚合和评分方式。

P2 和 P3 是诊断结果，P1 仍是总体主结果。

---

## 4. 公平比较的规则

四个后端必须保持以下内容一致：

- 相同的身份划分；
- 相同的 Gallery 和 Query 轨迹；
- 相同的每轨迹抽帧位置；
- 相同的 Gallery 质量门控；
- 相同的轨迹与身份聚合方法；
- 相同的余弦相似度和评价代码；
- 相同的 P1、P2、P3 子集定义。

每个模型可以使用自己的官方图片预处理。例如：

- OSNet 使用 BoxMOT 官方预处理；
- CLIP-ReID 使用官方 `256×128` 输入及均值、标准差均为 `0.5` 的归一化；
- SigLIP2-Person-ReID 使用其 `224×224` processor；
- DIFFER 使用官方配置中的输入尺寸和归一化。

这不算不公平，因为预处理是模型定义的一部分。但是不得为某个模型单独挑选更清楚的图片。

正式运行时禁止 `auto` 后端回退。指定 CLIP-ReID 时如果权重加载失败，脚本必须报错退出，
不能静默退回 OSNet、ResNet50 或颜色直方图。

---

## 5. 模型和权重如何固定

每次结果都必须记录：

- 后端名称；
- 模型来源 URL；
- 权重文件名、版本或 Hugging Face revision；
- 权重 SHA-256；
- 输入尺寸和归一化方式；
- embedding 维度；
- PyTorch、Transformers、ONNX Runtime 等关键版本；
- GPU 型号和推理精度；
- 当前 Git commit。

首轮建议冻结为：

| 后端 | 权重原则 |
|---|---|
| OSNet | 产品现用 `osnet_ain_x1_0_msmt17.pt` |
| CLIP-ReID | 论文作者仓库发布的 MSMT17 ViT-CLIP-ReID 权重，不使用第三方 ONNX 作为正式结果 |
| SigLIP 2 | `MarketaJu/siglip2-person-description-reid`，固定具体 revision |
| DIFFER | 作者发布的官方换衣 ReID 权重；优先选择与 LTCC 对应的正式 checkpoint，并记录哈希 |

如果官方权重来源、许可证或预处理无法确认，该 arm 应标记为“未完成”，不能换成名称相似的
第三方模型后仍沿用原名称。

---

## 6. 实验脚本如何调用产品后端

### 6.1 后端统一接口

实验脚本不应分别复制四套推理代码。产品侧扩展 `app/body_reid.py`，让所有模型实现相同接口：

```python
active_backend() -> str
embed_dim() -> int
embed(crop) -> np.ndarray
```

其中 `embed(crop)` 接收一张 PIL 人体裁图，返回一维 `float32` 向量。每个后端内部负责自己的
缩放、归一化和模型调用，返回前统一执行 L2 归一化。

建议增加以下固定后端名：

```text
osnet
clipreid
siglip2_person_reid
differ
```

实验入口只调用产品的 `body_reid.embed()`，不能绕过产品模块直接在实验脚本中加载模型。
这样实验验证的就是未来能够接入产品的真实后端逻辑。

### 6.2 需要增加的配置

后续实现时为各后端增加独立配置，例如：

```text
REID_BACKEND
REID_CLIPREID_WEIGHTS
REID_SIGLIP2_MODEL
REID_SIGLIP2_REVISION
REID_DIFFER_WEIGHTS
REID_DEVICE
```

每次切换后端后必须执行 `reset_backend()`，再检查 `active_backend()` 是否与请求一致。

### 6.3 已实现的脚本结构

```text
experiment\body_reid\
├── experiment_plan.md
├── body_reid_experiment\
│   ├── common.py
│   ├── protocol.py
│   ├── runner.py
│   └── summary.py
├── manifests\
│   ├── frozen_body_samples.csv
│   └── frozen_protocol.json
├── scripts\
│   ├── freeze_mevid_body_protocol.py
│   ├── run_body_reid_matrix.py
│   └── summarize_body_reid.py
└── results\
    └── final\
```

三个入口脚本及共享实现已经完成。`manifests` 中的冻结 CSV/JSON 必须在真实 MEVID
数据上运行 `freeze_mevid_body_protocol.py` 后才生成；当前不会伪造或提交数据清单。
冻结命令默认拒绝覆盖已有协议，只有确认重建时才允许显式传 `--force`。

通用的 MEVID 加载、固定随机划分、均匀抽帧和质量门控应复用现有实现，必要时提取到共享模块，
不要复制一套稍有差异的新逻辑。

### 6.4 实际命令

先冻结 test 正式协议和完全独立的 train 阈值校准协议：

```powershell
python .\experiment\body_reid\scripts\freeze_mevid_body_protocol.py `
  --data <MEVID_ROOT> `
  --seed 0 `
  --enroll-subjects 27 `
  --imposter-subjects 25 `
  --frames-per-track 8 `
  --max-gallery-tracks 3 `
  --max-query-tracks 4 `
  --calibration-enroll-subjects 50 `
  --calibration-imposter-subjects -1
```

四后端先对同一小批 Query 做冒烟测试：

```powershell
python .\experiment\body_reid\scripts\run_body_reid_matrix.py `
  --data <MEVID_ROOT> `
  --manifest .\experiment\body_reid\manifests\frozen_body_samples.csv `
  --protocol .\experiment\body_reid\manifests\frozen_protocol.json `
  --backends osnet,clipreid,siglip2_person_reid,differ `
  --device cuda `
  --smoke-samples 2
```

冒烟通过后运行正式矩阵；脚本逐个加载模型，不会让四个大模型同时占用显存：

```powershell
python .\experiment\body_reid\scripts\run_body_reid_matrix.py `
  --data <MEVID_ROOT> `
  --manifest .\experiment\body_reid\manifests\frozen_body_samples.csv `
  --protocol .\experiment\body_reid\manifests\frozen_protocol.json `
  --backends osnet,clipreid,siglip2_person_reid,differ `
  --device cuda
```

最后校验四个后端确实使用相同 Query/P2/P3/Gallery，并生成最终对比表：

```powershell
python .\experiment\body_reid\scripts\summarize_body_reid.py `
  --protocol .\experiment\body_reid\manifests\frozen_protocol.json `
  --results .\experiment\body_reid\results\runs
```

逐帧 embedding 只写入被 Git 忽略的 content-addressed NPZ cache；正式后端 JSON
只保留指标、每条 Query 的预测、阈值、耗时、显存和模型 provenance，不保存中间图片。
缓存键同时绑定冻结协议、样本 CSV 哈希、设备请求、实际设备、权重/源码 provenance 和
样本顺序；FMR 目标改变时不得复用旧评测结果。

---

## 7. 阈值如何确定

不同模型的相似度分布不同，不能把 OSNet 的 `0.6` 直接给其他模型使用。

建议分成两类指标：

### 不需要阈值的指标

- Rank-1：第一名是否为正确身份；
- Rank-5：正确身份是否出现在前五名；
- mAP：正确匹配是否整体排在前面。

这些指标先回答“模型会不会排序”。

由于主协议先把每个身份聚合成一个身份均值模板，这里的
`mAP_identity_template` 数值等于各 genuine Query 倒数排名的均值；它不是把每张
Gallery 图片分别作为检索项的传统 image-level mAP，报告中必须保留这个名称。

### 需要阈值的开放集指标

- 陌生人误识率 FMR；
- 在 FMR 不超过 1%、5%、10% 时，库内人员正确识别并接受率 TPIR；
- 拒绝率。

阈值只使用 MEVID 官方 train 身份建立的独立校准集确定。train 没有官方
Gallery/Query 划分，因此脚本在冻结阶段按固定身份、轨迹排序和 seed 建立互不重叠的
校准 Gallery、库内 Query 与陌生 Query。校准集中的身份不能出现在正式 test；
校准完成后冻结每个后端的阈值，再运行一次 test。这个过程不更新模型参数，因此应表述为
“零样本特征 + 目标域阈值校准”，而不是完全不接触目标域。

---

## 8. 除准确率外还要记录什么

模型最终要进入产品，因此还要报告：

- 单张人体 crop 的平均和 P95 推理时间；
- 峰值显存；
- 模型文件大小；
- embedding 维度；
- 每秒可处理图片数；
- Gallery 建档覆盖率；
- 产生 NaN、零向量或加载失败的数量。

模型若只提高少量准确率，却让延迟和显存增长数倍，不一定适合作为默认后端。

---

## 9. 实验执行顺序

### 阶段 1：冻结数据

- 读取 MEVID 官方 test 划分；
- 固定 27 个建档身份和 25 个陌生身份；
- 固定轨迹、帧路径、摄像头和服装编号；
- 检查 Gallery 与 Query 没有路径重复；
- 保存清单和哈希。

此阶段完成后，不得再因模型结果修改样本。

### 阶段 2：后端冒烟测试

每个模型先运行少量样本，检查：

- 加载到的后端名称正确；
- 输出维度与文档一致；
- 向量没有 NaN 或无穷大；
- 向量范数接近 1；
- 同一图片重复计算结果稳定；
- 实际激活后端与请求名称一致，禁止任何静默回退。

### 阶段 3：完整提取

- 按后端逐个运行，避免四个大模型同时占用显存；
- 每个后端读取相同 frozen manifest；
- 保存每个轨迹的最终向量、耗时和模型来源；
- 运行失败时明确报错，不跳过样本伪装成成功。

### 阶段 4：校准与正式测试

- 只用 train 校准开放集阈值；
- 冻结阈值；
- 在 test 上一次性生成 P1、P2、P3 结果；
- 同时报告 Rank-1、Rank-5、mAP、TPIR@FMR、延迟和显存。

### 阶段 5：结论

最终结论至少分别回答：

1. 哪个模型总体 Rank-1 最高；
2. 哪个模型在 FMR 不超过 5% 时认出的库内人员最多；
3. 哪个模型在严格换衣子集最好；
4. 哪个模型速度和显存最适合产品；
5. 新模型相对 OSNet 救回了哪些 Query，又损害了哪些 Query。

---

## 10. 运行前必须通过的检查

- 四个后端的 Gallery/Query `sample_id` 集合完全一致；
- Gallery 中没有 25 个陌生身份；
- Query 图片没有进入 Gallery；
- test 没有参与阈值选择；
- 每个后端都记录了真实名称和权重哈希；
- 没有后端发生静默回退；
- 所有比较使用同一评分与聚合代码；
- 冻结协议内容可重新计算出同一 `protocol_id`，且结果绑定同一 CSV 哈希；
- 四个结果的实验代码、Python/依赖、Git 状态、设备和 GPU 环境指纹一致；
- P2 的 Gallery 与 Query 服装编号确实不同；
- 原始数据、权重、embedding 和中间图片不提交到仓库；
- 只保存冻结清单、最终统计结果和必要的最终图表。

满足以上条件后，四个模型的差异才能解释为模型能力差异，而不是数据选择、预处理或阈值造成的
假提升。
