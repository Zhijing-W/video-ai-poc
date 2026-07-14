# Monitor v1

冻结的旧版应用，包含逐帧监控、目标比对、跟踪、人形 ReID 和轨迹级多帧证据融合。

## 启动

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

访问 `http://127.0.0.1:8000/monitor`。

本应用不再包含 Event Monitor 和 Phase 4 事件窗分析。新功能统一开发在 `../event-monitor/`。
