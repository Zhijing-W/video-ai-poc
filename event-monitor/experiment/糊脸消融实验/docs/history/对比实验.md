# 🧪 糊脸对比实验 · 指导手册

> 目标:在**真实监控场景**下,量化不同**方案组合**对**糊脸 bad-case** 识别率的提升。

---

## 1. 数据

> 关键:核心策略是**脸糊 → 退人形/步态兜底**,所以数据**必须是整帧视频 + 全身 + 走动 + 带身份**(非预 crop、非纯人脸库),否则根本测不了"兜底"。

- **起步** `ChokePoint`:走廊走动、多机位、带身份,直接下载,先把端到端融合跑通。
- **主力** `MEVID` / `MARS`:全身视频、跨镜身份,脸天然低清 → 正好测"脸糊靠人形+步态兜底"。`MEVID`(换装)更能逼出步态,`MARS` 更易获取。
- **CSA 弹药**:自录走动片,可摆拍"扭头 / 脸糊仍被认出"。
- (可选)纯人脸库 `SCFace` / `QMUL-SurvFace`:**只用于单独评人脸模块(S0–S4)**,测不了 S5 兜底,不作主力。
- 数据放 **git 仓库外**,`--data` 传入。协议:整帧 → 检测+跟踪 → 三路(脸/人形/步态)→ 融合 → 按 track 比身份。

## 2. 对比项(arm · 逐级叠加)

| arm | 方案组合 | 代码 / 配置 | 状态 |
|---|---|---|---|
| **S0 baseline** | SCRFD + ArcFace-r50,无增强,仅脸 | `FACE_BACKEND=insightface` | ✅ 冻结锚点 |
| **S1 换识别** | AdaFace-IR101 / MagFace / ArcFace-r100 | `_ensure_adaface`;MagFace 待接入 | 🟡 |
| **S2 +超分** | GFPGAN / CodeFormer / GPEN | `_ensure_superres` | 🟡 |
| **S3 +检测对齐** | RetinaFace / YOLOv8-face + 3D-68 | `face_3d_cue`;换检测器待接入 | 🟡 |
| **S4 +多帧融合** | 每条 track 最佳脸 / 多帧聚合 | `app/face.py` | ✅ |
| **S5 +跨路融合** | 脸 + 人形 + 步态,脸糊降权退兜底 | `app/services/multimodal_identity_fusion.py` | ✅ |
| **全栈** | S2 + S1 + S4 + S5 | 上述组合 | 🎯 目标 |

## 3. 指标(按糊脸分桶为主)

- 开集 **Rank-1 / Rank-5**、**TPIR@FPIR**。
- **按人脸质量分桶**(清晰 / 糊 / 极糊 / 无脸)逐桶报准确率。
- 检测召回、单帧端到端延迟。
- **头条结论**:在"脸糊/无脸"子集,**仅人脸(S0–S4)= __% → 加人形+步态融合(S5/全栈)= __%**。

## 4. 怎么跑

1. 数据(整帧全身视频)放仓库外某目录。
2. 跑评测(新脚本 `scripts/eval_identity_e2e.py`,或扩展 `eval_phase3.py` 的 `--manifest`):
   ```powershell
   python scripts/eval_identity_e2e.py --data <视频数据目录> --arms S0,S5,full
   ```
3. 内部走 `app/event_pipeline` 三路认人 → `identity_fusion`(S5)→ 复用 `eval_phase3` 的身份打分(purity/F1/ID-switch);`app/face.py` 武器开关控 S1–S4。
4. 输出 `docs/face_blur_eval_results.json` + `docs/face_blur_ablation.svg`。

## 5. 结论模板

> 在 `<数据集>` 上,脸糊子集身份识别率 **仅人脸 = __% → 脸+人形+步态融合 = __%**;最大单步增益来自 `S__`。

---

## 6. 扩展(后续,同骨架)

- **其他 LANE**:换 provider 即可复用本流程(ReID→Market-1501 / 步态→GREW / OCR→ICDAR)。
- **横向部署**:VM / VM+GPU / +Gateway / +负载均衡 / +CDN / SaaS,矩阵 = 形态 × 并发 × CPU/GPU,看延迟 / 成本 / 弹性。
