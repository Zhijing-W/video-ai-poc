# 模型文件管理

模型二进制文件不进入 Git。克隆仓库后执行：

```powershell
python scripts\download_models.py --include-optional-yolo
```

## 自动准备

| 文件 | 用途 |
|---|---|
| `yolov8m.pt` | 人体和物体检测 |
| `yolov8n-pose.pt` | 姿态与躯干区域 |
| `yolov8m-seg.pt` | 步态剪影分割 |

## 按需准备

- InsightFace `buffalo_l`：首次启用时由 InsightFace 下载，生产环境可挂载到 `INSIGHTFACE_HOME`。
- AdaFace：仓库放在 `models/AdaFace/`，权重放在其 `pretrained/pretrained_model/model.pt`。
- OpenGait：仓库放在 `models/OpenGait/`，GREW SkeletonGait++ 权重路径见 `.env.example`。
- GFPGAN：通过 `FACE_GFPGAN_WEIGHTS` 指定权重；未指定时由运行库按自身规则查找或下载。
- CodeFormer：
  - 通过`FACE_CODEFORMER_WEIGHTS`指定官方v0.1.0权重；留空时按需下载到
    `~/.cache/event-monitor/superres/codeformer-v0.1.0/`；
  - `FACE_CODEFORMER_FIDELITY`控制`[0,1]`质量/身份折中，产品默认`1.0`；
  - 源码与权重使用S-Lab License 1.0，当前仅按研究/非商业实验接入，生产使用前必须完成许可确认。
  - 随产品分发的许可证通知见`licenses/CodeFormer-S-Lab-License-1.0.txt`。
- Real-ESRGAN x2plus：
  - 通过`FACE_REALESRGAN_X2PLUS_WEIGHTS`指定官方权重；留空时按需下载到
    `~/.cache/event-monitor/superres/realesrgan-x2plus-v0.2.1/`；
  - 产品适配器只运行通用x2超分，不启用GFPGAN或其他face enhancement。
- CodeFormer与Real-ESRGAN使用Spandrel按权重自动识别模型架构；重型依赖和权重只在首次选择对应
  后端时加载。模型文件仍不进入Git。
- CR-FIQA：
  - 官方源码放在`models/CR-FIQA/source/`；
  - CR-FIQA(S)权重放在`models/CR-FIQA/32572backbone.pth`；
  - 官方仓库为`https://github.com/fdbtrs/CR-FIQA`；
  - 模型输出是原始回归分数，不是通用0–1概率；当前仅作诊断，不改变质量分桶、超分路由或恢复后验收；
  - `FACE_FIQA_*_THRESH` 必须在独立训练集与客户域完成校准后，才可升级为产品门控；
  - 官方代码为CC BY-NC 4.0，商业使用前必须完成许可确认。

本地准备示例：

```powershell
git clone --depth 1 https://github.com/fdbtrs/CR-FIQA.git models\CR-FIQA\source
python -m pip install gdown
gdown --folder "https://drive.google.com/drive/folders/13bE4LP303XA_IzL1YOgG5eN0c8efHU9h?usp=sharing" `
  -O models\CR-FIQA
```

生产部署建议将 `models/` 替换为只读云存储或 PVC 挂载。
