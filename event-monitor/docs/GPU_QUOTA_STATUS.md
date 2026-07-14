# GPU 配额工单状态 & 批准后操作手册

> 记录 GPU 配额工单进度，以及批准后从零到 GPU 推理生效的完整步骤。

---

## 1. 工单信息

### 工单 #2（**已批准并履行** ✅）
| 项 | 值 |
|---|---|
| **工单号** | `2607070030003714` |
| **摘要** | Southeast Asia 处的 标准 NCASv3_T4 系列 vCPU 配额请求 |
| **提交时间** | 2026-07-07 14:58 CST |
| **批准时间** | 2026-07-07 17:00 CST（**响应仅 2 小时！**）|
| **申请** | `Standard NCASv3_T4 Family`（T4 GPU 16GB VRAM）**16 vCPU** @ **Southeast Asia** |
| **配额验证** | `az quota list ... T4Family = 16` ✅ 已生效 |
| **状态** | ✅ **Fulfilled** |

### 工单 #1（已闭环）↩️
| 项 | 值 |
|---|---|
| **工单号** | `2607060030003679` |
| **申请** | `Standard NCADSA10v4 Family`（A10 v4）32 vCPU @ Indonesia Central |
| **结果** | ❌ SKU/Region 组合不支持（support engineer Surabhi B 回复）|
| **根本原因** | 官方 Product Availability by Region 页面确认 NCADSA10v4 不作为独立产品 SKU 上架，只有 A100 v4 / T4 v3 / A10 v5 (visualization) 可选 |
| **处理** | 邮件回复 Surabhi B 抄送新工单号，请求关闭本工单 |

---

## 2. 查看工单当前状态

```powershell
# 切到 MCAPS 订阅
az account set --subscription 260e353b-2845-4f40-ab23-8204e27f3842

# 查当前活跃工单（T4）
az support in-subscription tickets show `
    --ticket-name "2607070030003714" `
    --query "{Name:name, Status:status, Severity:severity}" `
    -o table

# 查所有通信记录（包括 support engineer 的回复）
az support in-subscription tickets communications list `
    --ticket-name "2607070030003714" `
    -o table
```

**状态含义**：
- `Open` = 未处理 / 处理中
- `Closed` = 已批准或已拒绝（看最新 communication 邮件内容）

---

## 3. GPU 批准后：一次性建全套（40 分钟）

假设 support engineer 邮件确认 `Approved`，跑下面这条链就能从零到 GPU 推理生效。

### Step 1：确认配额已到位
```powershell
az quota show `
    --resource-name "Standard NCASv3_T4 Family" `
    --scope "/subscriptions/260e353b-2845-4f40-ab23-8204e27f3842/providers/Microsoft.Compute/locations/southeastasia" `
    --query "properties.limit.value" -o tsv
# 应该显示 16（或 support engineer 批准的数字）
```

### Step 2：部署基础设施（Southeast Asia 全新一套，约 20 分钟）
```powershell
cd C:\Users\t-zhijingwu\Desktop\microsoft\Demo推进\video-understanding-poc
pwsh infra\deploy.ps1 `
    -Subscription 260e353b-2845-4f40-ab23-8204e27f3842 `
    -Region southeastasia `
    -Prefix videopoc
```

**建的东西**：`videopoc-rg` 资源组 + AKS + ACR + Blob + Files Premium + LAW + AppInsights。

⚠️ **首次跑注意**：ACR build 前把 `data/` `gfpgan/` `.git/` 临时挪出项目根（上次 East Asia 就卡在这里）：
```powershell
Move-Item data ..\data_backup -Force
Move-Item gfpgan ..\gfpgan_backup -Force
# 跑完 deploy 再挪回来
Move-Item ..\data_backup data
Move-Item ..\gfpgan_backup gfpgan
```

### Step 3：上传模型 & 数据（15 分钟）
```powershell
# deploy.ps1 输出会告诉你 storage account 名字，形如 videopocst<random>
pwsh infra\upload-models.ps1   -Storage <STORAGE_NAME>
pwsh infra\upload-datasets.ps1 -Storage <STORAGE_NAME>
```

### Step 4：加 GPU 节点池（5 分钟）
```powershell
pwsh infra\add-gpu-pool.ps1 `
    -RG videopoc-rg `
    -Cluster <AKS_NAME> `        # deploy.ps1 输出里有
    -Sku Standard_NC4as_T4_v3    # T4 GPU（1 张 T4 16GB）
# min=0 max=3，Spot，autoscale-to-zero
```

### Step 5：build GPU 镜像 & 打开开关（10 分钟）
```powershell
# build GPU 镜像
az acr build `
    --registry <ACR_NAME> `
    --image video-poc-gpu:latest `
    --file Dockerfile.gpu .

# helm 打开 gpu.enabled
helm upgrade video-poc charts\video-poc -n video-poc `
    --set gpu.enabled=true `
    --set gpu.image.repository=<ACR_LOGIN>/video-poc-gpu `
    --set gpu.image.tag=latest `
    --reuse-values
```

### Step 6：验证 GPU 生效
```powershell
# 1. GPU pod 起来了吗
kubectl -n video-poc get pods -l app=video-poc-gpu -o wide

# 2. Pod 里能看到卡吗
$pod = kubectl -n video-poc get pod -l app=video-poc-gpu -o jsonpath='{.items[0].metadata.name}'
kubectl -n video-poc exec $pod -- nvidia-smi
# 应该输出 A10 详情

# 3. 端到端跑一张糊脸看看多快
kubectl -n video-poc exec $pod -- python -c "
import time, cv2, insightface
app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)
img = cv2.imread('/data/sample_face.jpg')
t = time.time()
faces = app.get(img)
print(f'faces={len(faces)}, took {(time.time()-t)*1000:.1f}ms')
"
# CPU 通常 ~500ms，A10 应该 <50ms
```

---

## 4. 收到"拒绝"或"信息不足"邮件怎么办

**support engineer 常见反问**：

| 问题 | 回答模板 |
|---|---|
| "确认是否内部 POC" | Yes, internal Microsoft POC for CSA customer demo，非商业化 |
| "是否需要 On-Demand 还是 Spot 就够" | Spot is sufficient (autoscale-to-zero when idle) |
| "为什么不用 Azure ML" | AKS gives us multi-service inference (Face + ReID + Gait + OCR) that shares state (FAISS gallery) which fits our architecture better |
| "能否用其他 SKU" | T4 (NCASv3_T4 16 vCPU) is acceptable fallback |
| "为什么不用美区" | Compliance requirement: user is in mainland China, data must stay in APAC region |

**回复通信**：
```powershell
az support in-subscription tickets communications create `
    --ticket-name "gpu-quota-videopoc-202607061454" `
    --communication-name "reply-$(Get-Date -Format 'yyyyMMddHHmm')" `
    --communication-body "回复内容" `
    --communication-subject "Re: GPU quota"
```

---

## 5. 如果 A10 批不下来（备选方案）

按优先级从高到低试：

| 备选 | 命令 |
|---|---|
| **T4** | 工单里加"fallback to T4 v3 16 vCPU"，用 `add-gpu-pool.ps1 -Sku Standard_NC4as_T4_v3` |
| **K80** | 已有 48 vCPU 配额（全球所有区），直接 `add-gpu-pool.ps1 -Sku Standard_NC6`。K80 是老卡但对我们模型有 8-15x 加速 |
| **纯 CPU + KEDA 并行** | 不加 GPU 节点，用 K8s Job 起 5-10 个 CPU pod 并发跑实验，靠 pod 数量弥补速度 |

---

## 6. 关联文档

- **`CLOUD_PROGRESS.md`** — 上云整体进度全景
- **`AZURE_DEPLOY.md`** — 完整部署手册（步骤/成本/常见坑）
- **`cloud-architecture.svg`** — 云上架构图（当前还是 East Asia 版本，Indonesia Central 建好后同步更新）
- **`infra/add-gpu-pool.ps1`** — GPU 节点池一键加脚本
- **`infra/deploy.ps1`** — 完整部署脚本
