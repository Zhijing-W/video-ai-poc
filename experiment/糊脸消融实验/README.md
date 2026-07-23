# 低质量监控身份实验

本目录按“代码、文档、结果、样例数据、新实验”分类。当前论文正式结果只使用协议A和协议B3；
Market-1501、ChokePoint以及早期B/B2结果仅作为历史记录。

## 目录结构

```text
糊脸消融实验/
├─ scripts/                         # 所有实验与绘图脚本
│  ├─ run_mevid_e2e.py             # 协议A：人脸/人形/步态消融
│  ├─ run_mevid_face_b3.py         # 协议B3：独立训练身份校准
│  ├─ mevid_eval_common.py         # MEVID加载、特征与指标
│  ├─ plot_paper_results.py        # 从正式JSON生成论文图
│  └─ ...                          # 旧协议及历史实验脚本
├─ docs/
│  ├─ 实习论文_MEVID多模态身份识别实验.md
│  ├─ 实验设计_MEVID多模态.md
│  └─ history/                     # Market/ChokePoint及旧设计
├─ results/
│  ├─ runs/                        # MEVID实验JSON
│  ├─ paper_figures/               # 论文PNG/SVG
│  └─ legacy_market/               # Market/ChokePoint旧结果
├─ dataset/                        # 旧实验人工分桶样例
└─ 超分实验/
   ├─ 实验设计.md
   ├─ scripts/
   │  └─ run_checkin_superres_abc.py # 兼容CLI入口
   └─ checkin_superres/              # schema-v3正式实现
      ├─ preparation.py              # 固定Gallery/Query
      ├─ embeddings.py               # A/B/C缓存
      ├─ matrix.py                   # A/B1--3/C1--3固定多后端矩阵
      ├─ metrics.py                  # 指标与配对统计
      ├─ visualization.py            # 全量审计图
      ├─ common.py                   # hash/path/manifest
      └─ orchestration.py            # prepare/evaluate编排
```

## 正式结果

```text
results/runs/mevid_e2e_e27_i25_20260710_081808.json
results/runs/mevid_face_b3_train50_r5_test27_20260713_090421.json
```

## 生成论文图表

从仓库根目录执行：

```powershell
python .\experiment\糊脸消融实验\scripts\plot_paper_results.py
```

输出到：

```text
results/paper_figures/
```

## Check-in 超分实验

正式协议见 `超分实验/实验设计.md`。`run_checkin_superres_abc.py prepare` 冻结
check-in 正脸 Gallery 与全部官方 Query；`evaluate` 计算 A/B 一次并从缓存派生 C，
同时输出压缩 embedding、全量比较图和 image manifest。旧
`run_superres_gate.py` 仅保留作历史诊断，不代表当前正式协议。

固定manifest多后端正式实验使用GFPGAN、CodeFormer `w=1`与
Real-ESRGAN x2plus，并把全部输出统一为112×112后进入ArcFace。冻结
recoverable 40条Query的Rank-1为：

| 输入 | Rank-1 |
|---|---:|
| A 原图 | 26/40（65.0%） |
| C1 GFPGAN | 8/40（20.0%） |
| C2 CodeFormer | 11/40（27.5%） |
| C3 Real-ESRGAN | 18/40（45.0%） |

两种纯缩放控制均为26/40。2,126项结果checksum全部通过；正式报告见
`paper/多超分算法适用性报告.pdf`。原始结果归档保存在外部结果目录，不提交模型输出图片和
embedding缓存。
