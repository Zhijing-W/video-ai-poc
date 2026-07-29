# 模型文件管理

模型二进制文件不进入 Git。所有 checkout、Copilot worktree、产品运行和实验共用
`MODEL_ROOT`。默认位置为：

```text
~/.cache/event-monitor/models
```

克隆仓库并安装依赖后执行：

```powershell
python scripts\download_models.py --all
```

只准备某一类时可使用 `--reid`、`--superres`、`--face`。不带类别参数时只准备主
YOLO 检测权重；`--include-optional-yolo` 同时准备姿态和分割权重。

## 自动准备

| 相对 `MODEL_ROOT` 的路径 | 用途 |
|---|---|
| `detection/yolo/yolov8m.pt` | 人体和物体检测 |
| `detection/yolo/yolov8n-pose.pt` | 姿态与躯干区域 |
| `detection/yolo/yolov8m-seg.pt` | 步态剪影分割 |
| `superres/*` | GFPGAN、CodeFormer、Real-ESRGAN |
| `face/insightface` | buffalo_l 检测、关键点与 ArcFace |
| `reid/osnet` | 默认 OSNet-AIN |
| `reid/resnet50` | ImageNet 兜底特征 |
| `reid/clipreid` | 固定官方源码、CLIP 基础权重、MSMT17 checkpoint |
| `reid/siglip2/model` | 固定 Hugging Face 模型快照 |
| `reid/differ` | 固定官方源码与 LTCC checkpoint |

## 按需准备

- InsightFace `buffalo_l`、三个超分模型以及六个人形 ReID 后端由统一脚本准备。
- AdaFace：源码放在 `face/adaface/source/`，权重放在 `face/adaface/model.pt`。
- OpenGait：源码放在 `gait/opengait/source/`，GREW 权重放在同级目录。
- CodeFormer：
  - 默认读取`MODEL_ROOT/superres/codeformer/codeformer-v0.1.0.pth`；
  - `FACE_CODEFORMER_FIDELITY`控制`[0,1]`质量/身份折中，产品默认`1.0`；
  - 源码与权重使用S-Lab License 1.0，当前仅按研究/非商业实验接入，生产使用前必须完成许可确认。
  - 随产品分发的许可证通知见`licenses/CodeFormer-S-Lab-License-1.0.txt`。
- Real-ESRGAN x2plus：
  - 默认读取`MODEL_ROOT/superres/realesrgan/RealESRGAN_x2plus-v0.2.1.pth`；
  - 产品适配器只运行通用x2超分，不启用GFPGAN或其他face enhancement。
- CodeFormer与Real-ESRGAN使用Spandrel按权重自动识别模型架构；  权重在部署准备阶段下载，运行时只从本地加载。模型文件仍不进入Git。
- CR-FIQA：
  - 官方源码放在`face/cr_fiqa/source/`；
  - CR-FIQA(S)权重放在`face/cr_fiqa/32572backbone.pth`；
  - 官方仓库为`https://github.com/fdbtrs/CR-FIQA`；
  - 模型输出是原始回归分数，不是通用0–1概率；当前仅作诊断，不改变质量分桶、超分路由或恢复后验收；
  - `FACE_FIQA_*_THRESH` 必须在独立训练集与客户域完成校准后，才可升级为产品门控；
  - 官方代码为CC BY-NC 4.0，商业使用前必须完成许可确认。

## 人形 ReID 来源与定位

| 后端 | 官方来源与权重 | 产品定位 |
|---|---|---|
| CLIP-ReID | [Syliz517/CLIP-ReID](https://github.com/Syliz517/CLIP-ReID)，源码固定到 `eb1898b72c882875f478bebfc6d41644eece0a5d`，使用官方 MSMT17 ViT-B/16 checkpoint | 标准跨摄像头 ReID；当前官方实现要求 CUDA |
| SigLIP2 | [MarketaJu/siglip2-person-description-reid](https://huggingface.co/MarketaJu/siglip2-person-description-reid)，revision `196e5d6` | 衣着/描述语义特征，作为实验性补充，不替代标准身份 ReID |
| DIFFER | [xliangp/DIFFER](https://github.com/xliangp/DIFFER)，源码固定到 `67acb5d3658d103b4412c8c99f93ee8f085802fe`，使用官方 LTCC `eva02_l_bio_best.pth` | 面向换衣 ReID，方向最匹配但模型最重 |

运行 `python scripts\download_models.py --reid` 会从上述官方来源准备源码、基础
权重和微调 checkpoint。失败项会汇总报错，已成功项不会重复下载，可在网络恢复后直接重跑。
产品运行时只读取本地文件，不会访问 GitHub、Google Drive 或 Hugging Face。

CLIP-ReID 源码为 MIT，DIFFER 源码和 SigLIP2 模型卡标注为 Apache-2.0；实际部署或
再分发前仍需单独核对 checkpoint 所用训练数据集和权重条款。

本地准备示例：

```powershell
git clone --depth 1 https://github.com/fdbtrs/CR-FIQA.git $env:USERPROFILE\.cache\event-monitor\models\face\cr_fiqa\source
python -m pip install gdown
gdown --folder "https://drive.google.com/drive/folders/13bE4LP303XA_IzL1YOgG5eN0c8efHU9h?usp=sharing" `
  -O $env:USERPROFILE\.cache\event-monitor\models\face\cr_fiqa
```

生产部署建议将 `MODEL_ROOT` 指向只读云存储或 PVC 挂载。
