# Azure 上云部署手册（CPU-first，GPU-ready）

> 目标：把 video-understanding-poc 部署到 AKS，先用 CPU 跑通完整链路；GPU 配额到位后一条命令切换。
> 相关代码：`infra/main.bicep`、`infra/deploy.ps1`、`Dockerfile.cpu`、`Dockerfile.gpu`、`charts/video-poc/`

---

## 0. 前置

- **本地工具**：Azure CLI (`az`)、PowerShell 7+、Helm 3、kubectl（`az aks install-cli` 一键装 kubectl；helm 走 `winget install Helm.Helm`）
- **Azure 权限**：目标 Subscription 的 Owner 或 (Contributor + User Access Administrator)
- **推荐订阅**：MCAPS-Hybrid 内部订阅（无 spending limit，能开工单申 GPU）
- **推荐区域**：`southeastasia`（Southeast Asia / 新加坡）
  - 大陆访问 60-100ms
  - **T4 GPU（NCASv3_T4）+ A100（NCADS_A100_v4）都 GA**（Indonesia Central 试过，A10 v4 官方不上架）
  - GPU 配额已批 T4 16 vCPU（工单 `2607070030003714`）
- **Azure OpenAI**：已开通 gpt-4o 部署（本项目现有的 endpoint / key 直接复用）
- **GPU 配额**：MCAPS 订阅默认 0，必须开工单；见 `docs/GPU_QUOTA_STATUS.md`

---

## 1. 架构总览

```
[客户端/浏览器/RTMP]
        │
        ▼
[AKS Ingress (App Routing addon)]
        │
        ├──▶ webapi Deployment (CPU pool, 2-10 replicas, HPA)
        │       ├─ FastAPI + YOLO 门控 + OCR
        │       ├─ 挂 Blob → /models(RO) /data(RO) /results(RW)
        │       └─ 挂 Files → /gallery(RW)
        │
        └──▶ gpu-inference Deployment (GPU pool, 0-3 replicas, disabled by default)
                └─ Face / AdaFace / Gait / ReID / GFPGAN

存储：
  Azure Blob    models / datasets / results / videos
  Azure Files   gallery (FAISS + K/V, RWX)

观测：Log Analytics + Application Insights + Container Insights
```

---

## 2. 首次部署（一条命令）

```powershell
cd C:\Users\t-zhijingwu\Desktop\microsoft\Demo推进\video-understanding-poc

# ⚠️ 关键：先临时挪走 data/ 和 gfpgan/，防止 ACR build 打包卡死
# East Asia 第一次部署就是这里挂了 —— .dockerignore 在 az acr build tar 阶段不完全生效
Move-Item data ..\data_backup -Force -ErrorAction SilentlyContinue
Move-Item gfpgan ..\gfpgan_backup -Force -ErrorAction SilentlyContinue

# 参数：Subscription ID、区域、命名前缀（全局唯一）
pwsh infra\deploy.ps1 `
    -Subscription 260e353b-2845-4f40-ab23-8204e27f3842 `
    -Region southeastasia `
    -Prefix videopoc

# 跑完再挪回来
Move-Item ..\data_backup data -Force
Move-Item ..\gfpgan_backup gfpgan -Force
```

**做了 6 件事**：
1. az login + 切 subscription
2. `az group create -n videopoc-rg -l eastus`
3. `az deployment group create`：Bicep 拉起 ACR / Blob / Files / LAW / AppInsights / AKS
4. `az aks get-credentials`：合并 kubeconfig
5. `az acr build`：**云端 build**（不依赖本地 Docker），推 `video-poc-cpu:<timestamp>` + `:latest`
6. `helm upgrade --install video-poc`：装 Deployment / Service / Ingress / PVC / HPA

**大约耗时**：Bicep 8-12 分钟 + ACR build 8-15 分钟（首次装依赖慢，之后 layer cache 快很多） + Helm 1-2 分钟 = **首次 ~20 分钟**。

---

## 3. 上传模型与数据集

**必做**（否则 pod 会因为 /models 空而崩）：

```powershell
# 模型权重（YOLO + insightface buffalo_l + GFPGAN，共 ~1.5GB，5-10 分钟）
pwsh infra\upload-models.ps1 -Storage <storageAccountName>

# 数据集（ChokePoint ~600MB + Market-1501 ~150MB，5-15 分钟看网速）
pwsh infra\upload-datasets.ps1 -Storage <storageAccountName>
```

`<storageAccountName>` 是 `deploy.ps1` 结尾打印的那个 `Storage = <name> (Blob)`；也可以：
```powershell
az storage account list -g videopoc-rg --query "[?kind=='StorageV2'].name | [0]" -o tsv
```

---

## 4. 验证部署

```powershell
# pod 状态（应该看到 webapi 2/2 Running）
kubectl -n video-poc get pods

# 服务/Ingress 外部 IP
kubectl -n video-poc get svc,ingress

# 打开浏览器
$ip = kubectl -n video-poc get ingress video-poc -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
Start-Process "http://$ip/docs"           # FastAPI 自带 OpenAPI 面板
Start-Process "http://$ip/event-monitor"   # 事件监控页
```

**冒烟测试**：面板选样片视频 → 触发 `/api/event-monitor/understand` → 预期 30 秒内返回事件 JSON。
CPU-only 情况下步态/超分较慢，10 秒视频约 20-60 秒。

---

## 5. GPU 配额到位后（一条命令切换）

> **前置**：确认 GPU 工单已批（见 `docs/GPU_QUOTA_STATUS.md`）。用 `az quota show` 查 limit=32 才算生效。

### 5.1 加 GPU node pool

```powershell
pwsh infra\add-gpu-pool.ps1 -RG videopoc-rg -Cluster <aksName>
```
默认加 `Standard_NC32ads_A10_v4`（1 张 A10 整卡 24GB） Spot 池，min=0 max=3，taint `sku=gpu:NoSchedule` 隔离。

若配额只批了 T4，加 `-Sku Standard_NC4as_T4_v3`；若只有 K80 老配额（全球默认给 48 vCPU），加 `-Sku Standard_NC6`。

### 5.2 build GPU 镜像

```powershell
$ACR = az acr list -g videopoc-rg --query "[0].name" -o tsv
az acr build --registry $ACR `
    --image "video-poc-gpu:latest" `
    --file Dockerfile.gpu .
```

### 5.3 打开 helm gpu 开关

```powershell
helm upgrade video-poc charts\video-poc -n video-poc `
    --set gpu.enabled=true `
    --set gpu.image.repository=$ACR.azurecr.io/video-poc-gpu `
    --set gpu.image.tag=latest `
    --reuse-values
```

### 5.4 验证 GPU 可见

```powershell
kubectl -n video-poc exec deploy/video-poc-gpu -- nvidia-smi
```
应看到 A10 24GB 卡。此时 face/adaface/gait/reid/gfpgan 全部走 CUDA。

---

## 6. 自动扩容三层

| 层 | 工具 | 触发指标 | 位置 |
|---|---|---|---|
| **Pod 水平** | HPA | CPU% ≥ 70 → 扩 pod | `templates/deployment-webapi.yaml` 底部 |
| **Node 水平** | Cluster Autoscaler | Pod pending → 加 node | Bicep 里 `enableAutoScaling: true`（已开） |
| **事件驱动** | KEDA | Blob queue length / 自定义指标 | 后续按需装 `helm install keda kedacore/keda -n keda --create-namespace` |

### GPU 缩到 0（省钱关键）

普通 HPA `minReplicas` 必须 ≥ 1。要缩到 0：
```powershell
helm install keda kedacore/keda -n keda --create-namespace
```
再改 `templates/deployment-gpu.yaml`，把 HPA 换成 `ScaledObject`，触发器用 App Insights queryRate。
（后续单独出 PR，先跑通基础链路）

---

## 7. 日常操作

```powershell
# 改代码后重新部署
az acr build -r $ACR -t video-poc-cpu:latest -f Dockerfile.cpu .
kubectl -n video-poc rollout restart deployment/video-poc-webapi

# 查日志
kubectl -n video-poc logs -l component=webapi --tail=200 -f

# 查 App Insights（浏览器）
az portal browse ...    # 或 az resource show 拿链接

# 缩容
kubectl -n video-poc scale deploy/video-poc-webapi --replicas=1

# 销毁全部（省钱）
az group delete -n videopoc-rg --yes --no-wait
```

---

## 8. 成本预估（Southeast Asia, USD/月）

| 组件 | 配置 | 月成本 |
|---|---|---|
| AKS control plane | Free tier | $0 |
| System pool | 1× D2s_v5 | ~$75 |
| CPU pool | 2× D4s_v5 | ~$290 |
| Storage Blob | 500GB Hot | ~$10 |
| Azure Files Premium | 100GB | ~$16 |
| Log Analytics + AppInsights | ~5GB/月 | ~$15 |
| ACR Basic | 10GB | ~$5 |
| **CPU-only 小计** | | **~$410/月** |
| GPU pool（T4 Spot, autoscale 到 0, ~20 hrs/week） | 1× NC4as_T4_v3 | **~$15-30/月** |
| GPU pool（T4 Spot 8h/天） | 1× NC4as_T4_v3 | ~$60 |
| GPU pool（T4 按需常驻） | 1× NC4as_T4_v3 24/7 | ~$540 |

**暂停模式**（`az aks stop`）：只留存储 + ACR = **~$25/月**，5 分钟起回。

---

## 9. 常见坑（含 East Asia 试跑踩的坑）

- **⚠️ ACR build 卡在本地打包**（Windows 上 East Asia 试跑挂在这里）：
  - 症状：`Packing source code into tar to upload...` 后就一直不动
  - 原因：`.dockerignore` 在 `az acr build` 打包阶段不完全生效，`data/` `gfpgan/` `.git/` 巨大目录被打进 tar
  - 解决：build 前把这三个目录临时挪出项目根（见 §2 首次部署命令）
- **MCAPS 订阅所有区 A10/T4/A100 都返回 `ContactSupport`**：Portal + CLI 自助均被拒，必须开工单；见 `docs/GPU_QUOTA_STATUS.md`
- **Blob CSI 挂载失败**：检查 AKS kubelet MI 是否拿到 Storage Blob Data Owner（Bicep 已授权，验证 `az role assignment list`）
- **拉镜像 401**：检查 `az role assignment list --assignee <kubelet-mi> --all` 是否有 AcrPull
- **GPU pod pending**：检查 gpu node 是否 Ready（`kubectl get nodes -l workload=gpu`）+ NVIDIA device plugin 是否装（AKS 加 GPU pool 会自动装）
- **Azure OpenAI key 泄露到镜像**：`.dockerignore` 已排除 `.env`；用 helm `--set` 或 K8s Secret 注入
