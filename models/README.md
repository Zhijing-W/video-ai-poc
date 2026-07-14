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

生产部署建议将 `models/` 替换为只读云存储或 PVC 挂载。
