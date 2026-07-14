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
   └─ 实验设计.md
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

