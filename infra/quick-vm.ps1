# =============================================================
# 单 GPU VM 一键部署（跳过 AKS，POC 快速验证）
#
# 用法：
#   pwsh infra\quick-vm.ps1 -Subscription <sub-id> -Region southeastasia
#
# 做的事：
#   1. 建一台 Standard_NC4as_T4_v3（1 张 T4，Ubuntu 22.04）
#   2. cloud-init 自动装 Docker + NVIDIA driver + NVIDIA Container Toolkit
#   3. 打开 22 (SSH) + 80 (HTTP) + 8000 (FastAPI) 端口
#   4. 输出：IP + SSH 命令 + 后续 docker 命令
# =============================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Subscription,
    [string]$RG = 'videopoc-rg',
    [string]$Region = 'southeastasia',
    [string]$VmName = 'videopoc-gpu-vm',
    [string]$VmSize = 'Standard_NC4as_T4_v3',
    [string]$AdminUser = 'azureuser',
    [switch]$UseSpot   # 加 -UseSpot 走 Spot 便宜 90%
)
$ErrorActionPreference = 'Stop'

Write-Host "==> 目标：sub=$Subscription  rg=$RG  region=$Region  vm=$VmName ($VmSize)" -ForegroundColor Cyan

# ---- 1. 登录 + 切订阅 ----
Write-Host "`n[1/4] 检查登录 ..." -ForegroundColor Yellow
$acct = az account show 2>$null | ConvertFrom-Json
if (-not $acct -or $acct.id -ne $Subscription) {
    az account set --subscription $Subscription
}

# ---- 2. 确保 RG 存在 ----
Write-Host "`n[2/4] 确保 RG 存在 ..." -ForegroundColor Yellow
$rgState = az group show -n $RG --query "properties.provisioningState" -o tsv 2>$null
if (-not $rgState) {
    az group create -n $RG -l $Region -o none
    Write-Host "  ✓ 已建 $RG"
} else {
    Write-Host "  ✓ 已存在"
}

# ---- 3. 写 cloud-init（VM 启动时自动跑）----
Write-Host "`n[3/4] 生成 cloud-init（装 Docker + NVIDIA Container Toolkit）..." -ForegroundColor Yellow
$cloudInit = @'
#cloud-config
package_update: true
package_upgrade: false
packages:
  - ca-certificates
  - curl
  - gnupg
  - lsb-release
  - build-essential

runcmd:
  # Install Docker
  - install -m 0755 -d /etc/apt/keyrings
  - curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  - chmod a+r /etc/apt/keyrings/docker.gpg
  - echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list
  - apt-get update
  - DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  - usermod -aG docker azureuser
  # Install NVIDIA driver (Ubuntu 22.04 has ubuntu-drivers)
  - DEBIAN_FRONTEND=noninteractive apt-get install -y ubuntu-drivers-common
  - ubuntu-drivers install --gpgpu
  # Install NVIDIA Container Toolkit (docker + GPU)
  - curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  - curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  - apt-get update
  - DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
  - nvidia-ctk runtime configure --runtime=docker
  - systemctl restart docker
  # Marker file to know when setup is complete
  - echo "setup complete at $(date)" > /var/log/setup.done
'@

$cloudInitPath = "$env:TEMP\cloud-init-gpu.yaml"
Set-Content -Path $cloudInitPath -Value $cloudInit -Encoding UTF8
Write-Host "  ✓ cloud-init 写到 $cloudInitPath"

# ---- 4. 创建 VM ----
Write-Host "`n[4/4] 创建 VM（3-5 分钟）..." -ForegroundColor Yellow
$priorityArgs = @()
if ($UseSpot) {
    $priorityArgs = @('--priority', 'Spot', '--max-price', '-1', '--eviction-policy', 'Deallocate')
    Write-Host "  → 使用 Spot（便宜 90%，但可能被 evict）" -ForegroundColor DarkYellow
}

az vm create `
    --resource-group $RG `
    --name $VmName `
    --image "Ubuntu2204" `
    --size $VmSize `
    --admin-username $AdminUser `
    --generate-ssh-keys `
    --custom-data $cloudInitPath `
    --public-ip-sku Standard `
    --os-disk-size-gb 128 `
    --nsg-rule SSH `
    @priorityArgs `
    -o json | ConvertFrom-Json | Tee-Object -Variable vmInfo | Out-Null

$publicIp = $vmInfo.publicIpAddress
Write-Host "`n  ✓ VM 已创建" -ForegroundColor Green
Write-Host "    Public IP: $publicIp" -ForegroundColor Green

# ---- 5. 打开额外端口 ----
Write-Host "`n[5/5] 打开 HTTP/8000 端口 ..." -ForegroundColor Yellow
az vm open-port -g $RG -n $VmName --port 80 --priority 900 -o none
az vm open-port -g $RG -n $VmName --port 8000 --priority 910 -o none
Write-Host "  ✓ 端口 22/80/8000 已开"

# ---- 输出使用说明 ----
Write-Host "`n==================================================================" -ForegroundColor Cyan
Write-Host "  ✓ VM 已建好。cloud-init 还在后台装 Docker + NVIDIA (~10 分钟)" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  SSH 上机（等 3 分钟让 VM 起来）：" -ForegroundColor Yellow
Write-Host "     ssh $AdminUser@$publicIp" -ForegroundColor Gray
Write-Host ""
Write-Host "  检查 cloud-init 是否装完（登进 VM 后跑）：" -ForegroundColor Yellow
Write-Host "     ls -la /var/log/setup.done       # 有此文件 = 装完了" -ForegroundColor Gray
Write-Host "     nvidia-smi                       # 应看到 T4" -ForegroundColor Gray
Write-Host "     docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi" -ForegroundColor Gray
Write-Host ""
Write-Host "  拉 POC 代码 + 跑：" -ForegroundColor Yellow
Write-Host "     (下一步给你)" -ForegroundColor Gray
Write-Host ""
Write-Host "  用完暂停（省钱，$0/hr）：" -ForegroundColor Yellow
Write-Host "     az vm deallocate -g $RG -n $VmName" -ForegroundColor Gray
Write-Host ""
Write-Host "  恢复：" -ForegroundColor Yellow
Write-Host "     az vm start -g $RG -n $VmName" -ForegroundColor Gray
Write-Host "=================================================================="
