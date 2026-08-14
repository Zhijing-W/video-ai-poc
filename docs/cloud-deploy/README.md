# Cloud Deploy 架构图集

这个文件夹装 3 张部署方案图。**当前实际使用的是 `cloud-deploy-single-vm`**，其他两张是历史演进。

## 三张图对比

| 图 | 时期 | 状态 | 说明 |
|---|---|---|---|
| **`cloud-deploy-single-vm.svg`** ⭐ | **当前** | ✅ **正在跑** | 单 VM Spot + Docker，POC 最简方案 |
| `cloud-deploy-simple.svg` | 中间 | ❌ 放弃 | 单 Pod on AKS GPU Node，一开始想的最简 AKS 方案 |
| `cloud-architecture.svg` | 早期 | ❌ 放弃 | 完整 AKS 多 pool 方案（CPU pool + GPU pool 拆分）|

## 演进原因（为什么砍到最简）

### Round 1: 复杂 AKS 方案（`cloud-architecture.svg`）
- **设计**：AKS 集群 + 3 个 node pool（system + cpupool + gpupool）+ 2 个 Deployment（webapi CPU 常驻 + gpu autoscale）+ Blob CSI + Files Premium
- **问题**：代码是 monolithic FastAPI，不支持"CPU 前端 → GPU 后端"分层通信；硬拆需要写 gRPC 胶水
- **结论**：过度设计，YAGNI

### Round 2: 简化 AKS 方案（`cloud-deploy-simple.svg`）
- **设计**：AKS 集群 + system pool + gpupool（NC4as_T4_v3 Spot）+ 单 Pod all-in-one + Blob CSI + Files Premium
- **问题**：MCAPS 内部订阅有 policy `MCAPSGovDenyPolicies/BlockVMSKUs_N`，要求 N 系列（GPU）必须走 Spot；AKS node pool 加 Spot GPU 还有别的 quirks，反复失败
- **结论**：AKS + MCAPS 内部订阅 + GPU 组合坑太多

### Round 3: 单 VM 方案（`cloud-deploy-single-vm.svg`）⭐ 当前
- **设计**：直接开一台 NC4as_T4_v3 Spot VM，装 Docker + NVIDIA Container Toolkit，跑 Docker 容器
- **好处**：绕开 AKS 所有坑；30 分钟从零到 GPU 跑通
- **代价**：少了 K8s "自动扩缩/多副本" 花架子（POC 阶段完全不需要）

## 数据放哪（当前方案）

运行数据保存在 VM OS Disk，容器重建后仍然存在：

| VM 目录 | 容器目录 | 内容 |
|---|---|---|
| `/home/azureuser/vp/models` | `/models` | 模型权重和 CR-FIQA 源码 |
| `/home/azureuser/vp/data` | `/data` | 输入数据 |
| `/home/azureuser/vp/gallery` | `/gallery` | FAISS gallery |
| `/home/azureuser/vp/results` | `/results` | 实验与运行结果 |
| `/home/azureuser/vp/apphome` | `/home/appuser` | 模型缓存和 Python 用户目录 |

产品代码、HTML 和前端静态文件不再从 VM 目录绑定挂载，而是随 Docker 镜像发布，避免宿主机旧代码覆盖新镜像。

## GitHub Actions 自动构建与手动部署

- `.github/workflows/build-gpu-image.yml`
  - 合并到 `main` 且运行时代码发生变化时，使用 Azure OIDC 登录。
  - `Dockerfile.gpu.base` 与 `requirements.txt` 的内容哈希决定依赖镜像标签；相同依赖直接复用，不重复安装 CUDA/PyTorch/Python 包。
  - 依赖变化时优先从现有 `video-poc-gpu:main` 派生，只补新增依赖；手动选择 clean rebuild 才从 NVIDIA CUDA 镜像全量重建。
  - 在 ACR 云端始终构建 `video-poc-gpu:<commit SHA>`；只有 `main` 分支会更新 `video-poc-gpu:main`，feature 手动构建不会污染正式标签。
  - `app/**`、`experiment/**` 或前端代码单独变化时只重建轻量应用层。
  - 构建镜像不需要启动 GPU VM。
- `.github/workflows/deploy-gpu-vm.yml`
  - 在 GitHub Actions 页面手动运行，可部署 `main` 或指定 commit SHA。
  - VM 原本关闭时自动启动，部署和健康检查完成后保持运行。
  - 实验或演示结束后手动执行 `az vm deallocate -g videopoc-rg -n videopoc-gpu-vm`。
  - 新容器启动失败时自动恢复旧容器。

部署脚本使用 VM 托管身份从 ACR 拉取镜像，不依赖 SSH 端口，也不在 GitHub 或 VM 中保存 ACR 密码。`.env` 和模型权重继续保存在 VM，不进入 Git 仓库或镜像。容器内 `MODEL_ROOT=/models`；精度优先的 GPU POC 默认后端 DIFFER 必须已准备在宿主机 `/home/azureuser/vp/models/reid/differ/`。可运行 `python scripts/download_models.py --reid --model-root /home/azureuser/vp/models` 准备全部 ReID 资产；CLIP-ReID 是延迟敏感的显式替代项，显式 `.env` 配置仍可切换 SigLIP2、OSNet 或 `auto`。CPU 和高并发性能尚未验证；在目标演示视频和实际 T4 上用结果页的单 crop Body ReID 调用数、总耗时、均值、P95 及服务端阶段/总耗时完成测量后，再判断 DIFFER 是否构成瓶颈。健康检查成功后保留当前和部署前的镜像用于回退，并清理更旧的 ACR/legacy 应用标签与七天前的构建缓存。

### Optional gait assets

The single-VM bind mount maps `/home/azureuser/vp/models` to `/models`. The
canonical paths for the optional gait YOLO weights are:

```text
/home/azureuser/vp/models/detection/yolo/yolov8n-pose.pt
/home/azureuser/vp/models/detection/yolo/yolov8m-seg.pt
```

From a checkout with the application dependencies installed, provision them
with:

```bash
python scripts/download_models.py --include-optional-yolo \
  --model-root /home/azureuser/vp/models
```

The container defaults use the matching `/models/detection/yolo/...` paths.
When Gait is selected and an optional asset is absent, the run returns an
explicit warning and does not report Gait as having run.

## 未来演进路径

单 VM 撑到什么时候需要升级：
- **多用户并发** → 加 nginx 反代 + 多 VM 手动负载均衡；或回 AKS
- **要给客户看**"生产架构" → 回 AKS（policy 问题解决后）
- **要接 Copilot Studio** → 加 Ingress + Custom Domain + HTTPS
- **要跨机共享 gallery** → 才需要 Azure Files Premium

## 图脚本

每张图对应一个 `scripts/make_*.py` 生成脚本：
- `scripts/make_arch_diagram.py`（早期完整 AKS）
- `scripts/make_cloud_arch_diagram.py`（早期完整 AKS 复刻版）
- `scripts/make_simple_deploy_diagram.py`（简化 AKS）
- `scripts/make_single_vm_diagram.py`（当前，单 VM）

改图 = 改脚本重跑 `python scripts/make_single_vm_diagram.py`。
