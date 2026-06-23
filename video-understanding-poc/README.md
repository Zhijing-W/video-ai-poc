# Video Understanding PoC

视频 → ffmpeg 抽帧 → Azure OpenAI vision → 结构化 JSON。

## 本地启动

```powershell
# 1) 建虚拟环境并装依赖
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) 配置 Azure OpenAI
copy .env.example .env
#   编辑 .env，填 AZURE_OPENAI_ENDPOINT / API_KEY / DEPLOYMENT（vision 模型）

# 3) 启动服务
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

打开浏览器：`http://127.0.0.1:8000/` 上传视频即可。

- Swagger 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`GET /health`

> Blob 可选：`.env` 不填 `AZURE_STORAGE_*` 时自动用纯本地存储；填了就自动上传到容器。
