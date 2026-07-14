# 模型文件管理

模型二进制文件不进入 Git。克隆仓库后执行：

```powershell
python scripts\download_models.py --include-pose
```

| 文件 | 用途 |
|---|---|
| `yolov8m.pt` | YOLO 检测 |
| `yolov8n-pose.pt` | 人体姿态和躯干取色 |

OSNet/ReID 权重由 BoxMOT 根据 `REID_OSNET_WEIGHTS` 管理。生产部署建议把 `models/` 作为只读云存储或 PVC 挂载。
