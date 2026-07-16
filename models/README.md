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
- CR-FIQA：
  - 官方源码放在`models/CR-FIQA/source/`；
  - CR-FIQA(S)权重放在`models/CR-FIQA/32572backbone.pth`；
  - 官方仓库为`https://github.com/fdbtrs/CR-FIQA`；
  - 模型输出是原始回归分数，不是通用0–1概率；`FACE_FIQA_*_THRESH` 必须按目标数据校准；
  - 官方代码为CC BY-NC 4.0，商业使用前必须完成许可确认。

本地准备示例：

```powershell
git clone --depth 1 https://github.com/fdbtrs/CR-FIQA.git models\CR-FIQA\source
python -m pip install gdown
gdown --folder "https://drive.google.com/drive/folders/13bE4LP303XA_IzL1YOgG5eN0c8efHU9h?usp=sharing" `
  -O models\CR-FIQA
```

生产部署建议将 `models/` 替换为只读云存储或 PVC 挂载。
