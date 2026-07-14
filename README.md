# Event Monitor

当前主线应用，用于身份感知的多帧事件分析。

```text
视频或摄像头帧
  -> 检测与跟踪
  -> 人形 ReID、人脸、步态
  -> 事件窗与关键帧
  -> 身份和场景信息打包
  -> 多模态 LLM 事件报告
```

## 启动

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.\.venv\Scripts\python.exe scripts\download_models.py --include-optional-yolo
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

访问 `http://127.0.0.1:8000/event-monitor`。

API 前缀为 `/api/event-monitor`。旧页面地址 `/eventmonitor` 会自动跳转。

运行目录可通过以下环境变量配置：

- `DATA_DIR`
- `OUTPUT_DIR`
- `GALLERY_DIR`

## 数据与模型

- 模型文件统一放在 `models/`，Git 只保存 `models/manifest.json` 和说明。
- 本地视频与数据集统一放在 `data/`，Git 只保存目录说明。
- YOLO 权重可通过 `scripts/download_models.py` 自动准备。
- AdaFace、OpenGait、GFPGAN 等可选模型按照 `models/README.md` 放置。
- MEVID 等大型或受许可约束的数据集不能提交到仓库。
