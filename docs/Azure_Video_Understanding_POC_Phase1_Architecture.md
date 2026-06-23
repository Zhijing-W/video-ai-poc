# Azure 视频理解 PoC 第一阶段架构补充文档

> 项目目标：按照 mentor 的 LLM-first MVP 思路，在 Azure 上先搭建一个最小可行视频理解系统。第一阶段不做 YOLO/SSD、ViT/CLIP、Tracking 等增强模块，只验证“视频上传 → Server 抽帧 → Azure OpenAI Vision 总结 → 返回结构化结果”的端到端链路。

---

## 1. 第一阶段范围定义

### 1.1 本阶段要完成的能力

第一阶段的目标不是做完整生产级视频分析系统，而是先验证以下链路是否可行：

```text
VM Client / Browser
    ↓ 上传视频
Azure App Service / Container App 上的 FastAPI Server
    ↓ 保存原始视频
Azure Blob Storage
    ↓ 抽取少量关键帧
FFmpeg / OpenCV
    ↓ 调用多模态大模型
Azure OpenAI / Azure AI Foundry Vision Model
    ↓ 返回结果
Summary / Notification JSON
```

本阶段交付物：

1. 一个可部署在 Azure 上的后端服务。
2. 一个视频上传接口。
3. 视频文件可以保存到 Azure Blob Storage。
4. Server 能从视频中抽取固定数量的帧。
5. 抽取的帧可以传给 Azure OpenAI Vision 模型。
6. 模型返回结构化 JSON，包括 summary、detected_objects、possible_events、notification、confidence。
7. 可以通过 Swagger UI / Postman / 简单 HTML 页面演示。

### 1.2 本阶段暂不实现的能力

以下模块作为后续 Phase 2/3，不放进第一阶段：

- YOLO / SSD object detection
- ViT / CLIP key frame selection
- object tracking / trajectory analysis
- event scoring / weight fusion
- video clip 自动拼接
- 完整 React 前端
- 用户登录、权限系统、多租户隔离
- 生产级异步任务队列

---

## 2. Azure 资源设计

### 2.1 资源总览

| Azure 资源 | 用途 | 第一阶段是否需要 | 建议说明 |
|---|---|---:|---|
| Resource Group | 管理 PoC 所有资源 | 必须 | 建议单独建一个 PoC resource group，方便后续清理 |
| Azure Blob Storage | 存储原始视频、抽帧图片、结果 JSON | 必须 | 存储视频和图片这类非结构化数据 |
| Blob Container | 逻辑存储容器 | 必须 | 可以建一个 container，用路径区分 raw-videos、frames、results |
| Azure App Service / Azure Container App | 托管 FastAPI 后端 | 必须 | 第一阶段推荐 App Service for Containers，部署简单 |
| Azure Container Registry | 存放 Docker image | 推荐 | 如果使用容器部署，则推荐使用 ACR |
| Azure OpenAI / Azure AI Foundry | 部署 vision-capable model | 必须 | 用于图片理解和视频摘要生成 |
| Managed Identity | App 访问 Storage / OpenAI 的身份 | 推荐 | 减少连接字符串和密钥管理 |
| Application Insights | 日志、错误追踪 | 可选但推荐 | 方便 debug 上传、抽帧、模型调用问题 |
| Key Vault | 统一管理 secret | 可选 | 第一版可以先用 App Settings，后续再迁移 |

### 2.2 Blob Storage 路径设计

建议使用一个 container，例如：

```text
video-understanding-poc
```

内部路径：

```text
raw-videos/{video_id}/input.mp4
frames/{video_id}/frame_001.jpg
frames/{video_id}/frame_002.jpg
results/{video_id}/result.json
logs/{video_id}/process_log.json
```

这样后续加入 YOLO/CLIP 时，也可以继续复用相同路径结构：

```text
detections/{video_id}/yolo_result.json
embeddings/{video_id}/clip_frame_embeddings.json
tracks/{video_id}/tracking_result.json
```

---

## 3. 后端服务设计

### 3.1 技术选择

第一阶段推荐使用：

```text
Python 3.10+
FastAPI
Uvicorn / Gunicorn
FFmpeg 或 OpenCV
azure-storage-blob
azure-identity
Azure OpenAI SDK / openai SDK
Docker
```

选择 FastAPI 的原因：

- 上传文件接口开发快。
- 自带 Swagger UI，可以直接用于演示和测试。
- Python 生态适合处理视频、图片和 AI SDK。
- 后续加入 YOLO、OpenCV、CLIP 等模块比较自然。

### 3.2 API 设计

第一版建议只做 3 个接口。

#### 3.2.1 上传并处理视频

```http
POST /upload-video
Content-Type: multipart/form-data
```

输入：

```text
file: mp4 / mov / avi
```

返回：

```json
{
  "video_id": "20260612_demo_001",
  "status": "completed",
  "result_url": "/result/20260612_demo_001",
  "summary": "..."
}
```

第一阶段为了简单，可以同步处理，即上传后直接抽帧、调用 LLM、返回结果。后续如果视频变大，再改成异步任务。

#### 3.2.2 查询处理状态

```http
GET /status/{video_id}
```

返回：

```json
{
  "video_id": "20260612_demo_001",
  "status": "uploaded | extracting_frames | calling_llm | completed | failed",
  "message": "..."
}
```

#### 3.2.3 获取结果

```http
GET /result/{video_id}
```

返回：

```json
{
  "video_id": "20260612_demo_001",
  "summary": "...",
  "detected_objects": [],
  "possible_events": [],
  "notification": "...",
  "confidence": "medium",
  "frames_used": []
}
```

---

## 4. 视频处理流程

### 4.1 视频上传

流程：

```text
Client 上传视频
→ FastAPI 接收 UploadFile
→ 生成 video_id
→ 临时保存到 /tmp/{video_id}/input.mp4
→ 上传原视频到 Blob Storage
```

建议限制：

```text
视频长度：先控制在 30-60 秒以内
文件大小：第一版建议小于 100 MB
格式：优先支持 mp4
```

第一阶段不要过早支持所有格式，避免视频解码问题影响主链路验证。

### 4.2 抽帧策略

第一阶段不做 ViT/CLIP，只做规则抽帧。

建议策略：

```text
每 3-5 秒抽一帧
最多抽 6-10 张
跳过视频最开头和最后过暗/重复的帧
图片统一保存为 jpg
图片宽度可压缩到 768 或 1024 px
```

示例：

```text
00:00:03 → frame_001.jpg
00:00:08 → frame_002.jpg
00:00:13 → frame_003.jpg
00:00:18 → frame_004.jpg
00:00:23 → frame_005.jpg
00:00:28 → frame_006.jpg
```

抽帧可以用 FFmpeg：

```bash
ffmpeg -i input.mp4 -vf fps=1/5 frames/frame_%03d.jpg
```

也可以用 OpenCV 在 Python 中按 timestamp 抽帧。第一阶段推荐 FFmpeg，因为稳定、处理视频格式能力强。部署到 Azure 时，建议使用 Docker，把 FFmpeg 安装进镜像中。

### 4.3 抽帧结果元数据

建议生成 metadata：

```json
{
  "video_id": "20260612_demo_001",
  "frames": [
    {
      "frame_id": "frame_001",
      "timestamp": "00:00:03",
      "local_path": "/tmp/20260612_demo_001/frames/frame_001.jpg",
      "blob_path": "frames/20260612_demo_001/frame_001.jpg"
    }
  ]
}
```

这个 metadata 后续可以给 LLM 或 YOLO/CLIP 使用。

---

## 5. Azure OpenAI Vision 调用设计

### 5.1 模型选择

第一阶段建议使用 Azure OpenAI / Azure AI Foundry 中支持 image input 的 vision-capable chat model，例如：

```text
GPT-4.1 series
GPT-4o series
```

如果 GPT-4.1 access 或 region 不方便，第一版可以先用 GPT-4o / GPT-4o-mini 做视觉理解验证。

### 5.2 输入设计

LLM 输入不应该是整个视频，而是抽取后的关键帧图片。

输入内容：

```text
system prompt
+ user prompt
+ 6-10 张 frame image
+ frame timestamp metadata
```

### 5.3 Prompt 模板

建议第一版使用固定 JSON 输出，方便后端解析。

```text
你是一个视频理解助手。
下面是从同一个视频中按时间顺序抽取的关键帧。
请只根据图片中可见的信息进行总结，不要猜测图片中没有出现的内容。

请输出严格 JSON，字段如下：
{
  "summary": "用 1-3 句话总结视频内容",
  "detected_objects": ["可见对象列表，例如 person, package, pet, vehicle"],
  "possible_events": ["可能发生的事件，例如 person_appears, object_moved"],
  "notification": "面向用户的一句话通知",
  "confidence": "low | medium | high",
  "evidence": [
    {
      "timestamp": "对应帧时间",
      "observation": "该帧中可见的信息"
    }
  ],
  "limitations": "如果信息不足，请说明限制"
}
```

### 5.4 结果保存

LLM 返回结果后，保存到：

```text
results/{video_id}/result.json
```

建议结果结构：

```json
{
  "video_id": "20260612_demo_001",
  "model": "gpt-4.1-or-gpt-4o",
  "frames_used": [
    "frames/20260612_demo_001/frame_001.jpg"
  ],
  "llm_result": {
    "summary": "...",
    "detected_objects": [],
    "possible_events": [],
    "notification": "...",
    "confidence": "medium",
    "evidence": [],
    "limitations": "..."
  }
}
```

---

## 6. Azure 部署方案

### 6.1 推荐部署方式：FastAPI + Docker + Azure App Service for Containers

推荐原因：

- FFmpeg 依赖可以直接打包进 Docker image。
- 本地和云端运行环境一致。
- 后续加入 OpenCV / YOLO / CLIP 时也方便管理依赖。
- App Service 管理简单，适合第一阶段 PoC。

### 6.2 项目目录建议

```text
video-understanding-poc/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py               # 环境变量读取
│   ├── storage.py              # Blob Storage 操作
│   ├── video_processor.py      # FFmpeg/OpenCV 抽帧
│   ├── llm_client.py           # Azure OpenAI 调用
│   ├── schemas.py              # Pydantic schema
│   └── utils.py
├── templates/
│   └── upload.html             # 可选简单上传页面
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
└── docs/
    └── architecture.md
```

### 6.3 Dockerfile 示例

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY templates ./templates

EXPOSE 8000

CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300"]
```

### 6.4 requirements.txt 示例

```text
fastapi
uvicorn
gunicorn
python-multipart
azure-storage-blob
azure-identity
openai
pydantic
opencv-python-headless
```

如果只使用 FFmpeg 命令抽帧，OpenCV 可以暂时不装。

### 6.5 环境变量设计

部署到 Azure App Service 后，在 App Settings 中配置：

```text
AZURE_STORAGE_ACCOUNT_NAME=<storage-account-name>
AZURE_STORAGE_CONTAINER_NAME=video-understanding-poc
AZURE_OPENAI_ENDPOINT=<azure-openai-endpoint>
AZURE_OPENAI_API_KEY=<api-key>              # 第一版可用，后续建议改 Managed Identity
AZURE_OPENAI_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_API_VERSION=<api-version>
MAX_FRAMES=8
FRAME_INTERVAL_SECONDS=5
```

如果使用 Managed Identity 访问 Storage，可以减少 Storage connection string 的使用。

---

## 7. 部署操作步骤

> 以下是建议操作流程。具体命令需要根据实际 Azure subscription、resource group、region 和命名规范调整。

### 7.1 创建 Resource Group

```bash
az group create \
  --name rg-video-understanding-poc \
  --location eastasia
```

### 7.2 创建 Storage Account 和 Container

```bash
az storage account create \
  --name <unique-storage-account-name> \
  --resource-group rg-video-understanding-poc \
  --location eastasia \
  --sku Standard_LRS
```

创建 container：

```bash
az storage container create \
  --name video-understanding-poc \
  --account-name <unique-storage-account-name>
```

### 7.3 创建 Azure OpenAI / Foundry Model Deployment

在 Azure Portal 或 Azure AI Foundry 中：

1. 创建 Azure OpenAI / Foundry 资源。
2. 选择支持 vision input 的模型。
3. 部署模型，例如 GPT-4.1 或 GPT-4o。
4. 记录 endpoint、deployment name、API key 或配置 Managed Identity。

### 7.4 创建 Azure Container Registry

```bash
az acr create \
  --resource-group rg-video-understanding-poc \
  --name <unique-acr-name> \
  --sku Basic
```

登录 ACR：

```bash
az acr login --name <unique-acr-name>
```

构建并推送镜像：

```bash
docker build -t <unique-acr-name>.azurecr.io/video-understanding-poc:v1 .
docker push <unique-acr-name>.azurecr.io/video-understanding-poc:v1
```

### 7.5 创建 App Service Plan 和 Web App

```bash
az appservice plan create \
  --name asp-video-understanding-poc \
  --resource-group rg-video-understanding-poc \
  --is-linux \
  --sku B1
```

创建 Web App：

```bash
az webapp create \
  --resource-group rg-video-understanding-poc \
  --plan asp-video-understanding-poc \
  --name <unique-webapp-name> \
  --deployment-container-image-name <unique-acr-name>.azurecr.io/video-understanding-poc:v1
```

配置 App Settings：

```bash
az webapp config appsettings set \
  --resource-group rg-video-understanding-poc \
  --name <unique-webapp-name> \
  --settings \
  AZURE_STORAGE_ACCOUNT_NAME=<storage-account-name> \
  AZURE_STORAGE_CONTAINER_NAME=video-understanding-poc \
  AZURE_OPENAI_ENDPOINT=<endpoint> \
  AZURE_OPENAI_API_KEY=<key> \
  AZURE_OPENAI_DEPLOYMENT=<deployment-name> \
  MAX_FRAMES=8 \
  FRAME_INTERVAL_SECONDS=5
```

### 7.6 配置 Managed Identity 访问 Storage

推荐后续将 Storage 访问改成 Managed Identity：

1. 给 Web App 开启 System-assigned managed identity。
2. 在 Storage Account 上给这个 identity 分配 `Storage Blob Data Contributor` 权限。
3. 后端使用 `DefaultAzureCredential` 访问 Blob Storage。

第一版如果时间紧，可以先用 connection string 或 account key，但要避免写入代码仓库。

### 7.7 验证部署

访问：

```text
https://<unique-webapp-name>.azurewebsites.net/docs
```

在 Swagger UI 中测试：

```text
POST /upload-video
```

上传一个短视频，检查：

1. App Service 日志是否正常。
2. Blob Storage 中是否出现 raw video。
3. Blob Storage 中是否出现 frames。
4. result JSON 是否生成。
5. API 是否返回 summary。

---

## 8. 本地开发和测试流程

### 8.1 本地环境准备

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

确认 FFmpeg 可用：

```bash
ffmpeg -version
```

### 8.2 本地运行

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

访问：

```text
http://localhost:8000/docs
```

### 8.3 本地测试清单

上传一个 30 秒以内 mp4，检查：

- 是否生成 video_id。
- 本地 `/tmp/{video_id}` 是否有原视频。
- 是否成功抽出 frame 图片。
- frame 图片是否能上传 Blob。
- GPT Vision 是否返回 JSON。
- JSON 是否可以解析。

---

## 9. 第一阶段验收标准

### 9.1 功能验收

| 验收项 | 通过标准 |
|---|---|
| 视频上传 | 可以通过 Swagger/Postman 上传 mp4 |
| Blob 存储 | 原视频和抽帧图片可以在 Blob Storage 中看到 |
| 抽帧 | 能从视频抽出固定数量 jpg 图片 |
| LLM 调用 | 能把抽帧图片传给 vision model |
| 结果返回 | 能返回结构化 JSON |
| 可演示性 | 有一个可访问的 Azure endpoint 或本地 demo |

### 9.2 结果质量验收

第一阶段不追求非常高准确率，但需要满足：

- summary 能描述视频主要内容。
- detected_objects 能列出明显可见对象。
- possible_events 不应过度猜测。
- confidence 可以反映不确定性。
- limitations 能说明图像不足或无法判断的地方。

---

## 10. 需要提前学习的内容

### 10.1 必须提前学习

1. **FastAPI 文件上传**
   - 学习 `UploadFile`、`File`、multipart form-data。
   - 理解 `/docs` Swagger UI 如何测试上传接口。

2. **Azure Blob Storage Python SDK**
   - 学习如何创建 `BlobServiceClient`。
   - 学习如何上传视频、上传图片、下载结果 JSON。
   - 学习 container、blob path、content type 的概念。

3. **Azure App Service / Container 部署**
   - 学习 Dockerfile 基础。
   - 学习如何把 FastAPI 容器部署到 App Service。
   - 学习 App Settings 环境变量配置。

4. **FFmpeg 基础命令**
   - 学习视频抽帧命令。
   - 学习如何控制抽帧间隔、输出图片格式和大小。

5. **Azure OpenAI Vision 调用**
   - 学习如何部署 vision-capable model。
   - 学习 chat completion 中如何传入 image input。
   - 学习如何控制 JSON 输出。

### 10.2 可后续学习

1. **Managed Identity + RBAC**
   - 用于替代连接字符串和密钥。

2. **Application Insights**
   - 用于观察接口错误、模型调用失败、处理耗时。

3. **Azure Queue / Functions**
   - 如果视频处理时间变长，后续改成异步任务。

4. **YOLO / SSD**
   - 用于后续低成本筛选关键目标。

5. **CLIP / ViT**
   - 用于后续关键帧筛选，减少 LLM 调用。

---

## 11. 风险和应对

| 风险 | 可能表现 | 应对 |
|---|---|---|
| Azure OpenAI 模型权限或 region 不支持 | 无法部署 GPT-4.1 / vision model | 先确认可用模型；不行先用 GPT-4o / GPT-4o-mini |
| App Service 中 FFmpeg 不可用 | 抽帧失败 | 使用 Docker，把 FFmpeg 安装进镜像 |
| 视频太大导致上传或处理超时 | 请求失败或等待过久 | 第一版限制视频长度和大小；后续改异步处理 |
| LLM 输出不是合法 JSON | 后端解析失败 | prompt 强制 JSON；temperature 调低；增加 JSON 修复逻辑 |
| 图片太多导致成本高 | 模型调用成本增加 | 最多传 6-10 帧；压缩图片；后续加 YOLO/CLIP 筛选 |
| Blob 权限配置复杂 | 上传失败 | 第一版可以先用 connection string；后续切 Managed Identity |
| 公司网络或权限限制 | Azure 访问不稳定 | 本地先跑通，再部署；保留本地 demo 兜底 |

---

## 12. 下一阶段扩展方向

### Phase 2：加入 YOLO / SSD 做低成本筛选

目标：不要让所有帧都进入 LLM。

```text
Video → 抽帧 → YOLO 检测 person/package/pet → 只把有目标的帧送 LLM
```

输出增强：

```json
{
  "detections": [
    {
      "frame_id": "frame_001",
      "class": "person",
      "bbox": [100, 120, 260, 500],
      "confidence": 0.91
    }
  ]
}
```

### Phase 3：加入 ViT / CLIP 关键帧筛选

目标：减少重复帧，只保留语义变化大的帧。

```text
Frames → CLIP embedding → cosine distance / clustering → selected frames → LLM
```

### Phase 4：加入 Tracking 和 Event Scoring

目标：从“图像描述”升级为“事件判断”。

```text
YOLO detection → ByteTrack / DeepSORT → trajectory → event rule → GPT summary
```

示例事件：

```text
person_appears
package_delivered
object_removed
pet_detected
vehicle_approached
```

---

## 13. 第一阶段最终推荐实现路径

按照优先级推进：

```text
1. 本地 FastAPI 上传接口跑通
2. 本地 FFmpeg 抽帧跑通
3. Blob Storage 上传原视频和帧跑通
4. Azure OpenAI Vision 调用跑通
5. 返回 JSON summary 跑通
6. Docker 化应用
7. 部署到 Azure App Service
8. 用短视频做 demo 测试
9. 整理结果和下一阶段优化建议
```

第一阶段最重要的原则：

> 先跑通端到端闭环，不追求完整模型能力；先验证 LLM-first MVP 是否可行，再逐步加入低成本 CV 模块优化成本和准确率。

---

## 14. 可以对 mentor 说明的版本

可以这样概括：

> 第一阶段我会先做 Azure-hosted LLM-first MVP。用户从 VM Client 或浏览器上传视频，FastAPI Server 部署在 Azure App Service 上，视频和抽帧图片保存到 Azure Blob Storage。Server 使用 FFmpeg 从视频中抽取少量关键帧，然后调用 Azure OpenAI vision-capable model 进行识别和总结，最后返回结构化 JSON。YOLO/SSD 和 ViT/CLIP 暂时不放进第一版，后续根据成本和效果再作为优化模块加入。
