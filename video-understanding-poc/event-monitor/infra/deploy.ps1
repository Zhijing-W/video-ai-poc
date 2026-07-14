# =============================================================
# 视频理解 POC — 一键部署到 Azure
#
# 用法：
#   pwsh infra\deploy.ps1 -Subscription <sub-id> -Region eastus -Prefix videopoc
#
# 全流程：
#   1. az login + set subscription
#   2. 建 Resource Group
#   3. 部署 Bicep（ACR + Storage + Files + LAW + AKS）
#   4. az aks get-credentials 拿 kubeconfig
#   5. ACR build 镜像（云端 build，避免本地 Docker）
#   6. helm upgrade —— 装/升 workload
#
# 后续（GPU 配额批下来后）：
#   pwsh infra\add-gpu-pool.ps1 -Cluster <aks> -RG <rg>
# =============================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Subscription,

    [string]$Region = 'eastus',
    [string]$Prefix = 'videopoc',
    [string]$RgSuffix = 'rg',           # RG 名 = <Prefix>-<RgSuffix>
    [string]$CpuSku = 'Standard_D4s_v5',
    [int]$CpuNodes = 2,

    [switch]$SkipInfra,                 # 只 build+deploy 镜像，不动 Bicep
    [switch]$SkipImage,                 # 只跑 Bicep，不 build 镜像
    [switch]$SkipHelm                   # 只 build 镜像，不部署 helm
)

$ErrorActionPreference = 'Stop'
$RG = "$Prefix-$RgSuffix"
$root = Split-Path -Parent $PSScriptRoot   # repo 根

Write-Host "==> 目标：sub=$Subscription  rg=$RG  region=$Region  prefix=$Prefix" -ForegroundColor Cyan

# ------------------------ 1. az login + sub ---------------------
Write-Host "`n[1/6] 检查 Azure 登录状态 ..." -ForegroundColor Yellow
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account -or $account.id -ne $Subscription) {
    az login --output none
    az account set --subscription $Subscription
}
$adminObjectId = az ad signed-in-user show --query id -o tsv

# ------------------------ 2. Resource Group ---------------------
Write-Host "`n[2/6] 创建 Resource Group ..." -ForegroundColor Yellow
az group create -n $RG -l $Region -o none

# ------------------------ 3. Bicep 部署 -------------------------
if (-not $SkipInfra) {
    Write-Host "`n[3/6] 部署 Bicep（ACR + Storage + Files + AKS，约 8-12 分钟）..." -ForegroundColor Yellow
    $deployName = "videopoc-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    $result = az deployment group create `
        -g $RG -n $deployName `
        -f "$PSScriptRoot\main.bicep" `
        -p namePrefix=$Prefix cpuVmSize=$CpuSku cpuNodeCount=$CpuNodes adminObjectId=$adminObjectId `
        -o json | ConvertFrom-Json

    $global:ACR_LOGIN = $result.properties.outputs.acrLoginServer.value
    $global:ACR_NAME  = $result.properties.outputs.acrName.value
    $global:AKS_NAME  = $result.properties.outputs.aksName.value
    $global:STORAGE   = $result.properties.outputs.storageAccountName.value
    $global:FILES     = $result.properties.outputs.filesStoreName.value
    $global:AI_CONN   = $result.properties.outputs.appInsightsConnStr.value
    Write-Host "  ACR = $ACR_LOGIN" -ForegroundColor Green
    Write-Host "  AKS = $AKS_NAME" -ForegroundColor Green
    Write-Host "  Storage = $STORAGE (Blob) / $FILES (Files)" -ForegroundColor Green
} else {
    # 从现有 RG 找资源名
    $global:ACR_NAME  = az acr list -g $RG --query "[0].name" -o tsv
    $global:ACR_LOGIN = az acr list -g $RG --query "[0].loginServer" -o tsv
    $global:AKS_NAME  = az aks list -g $RG --query "[0].name" -o tsv
    $global:STORAGE   = az storage account list -g $RG --query "[?kind=='StorageV2'].name | [0]" -o tsv
    $global:FILES     = az storage account list -g $RG --query "[?kind=='FileStorage'].name | [0]" -o tsv
    Write-Host "  复用已有资源：ACR=$ACR_LOGIN  AKS=$AKS_NAME" -ForegroundColor Green
}

# ------------------------ 4. kubeconfig -------------------------
Write-Host "`n[4/6] 拉 AKS 凭据 ..." -ForegroundColor Yellow
az aks get-credentials -g $RG -n $AKS_NAME --overwrite-existing -o none

# ------------------------ 5. ACR build 镜像 ---------------------
if (-not $SkipImage) {
    $tag = Get-Date -Format 'yyyyMMdd-HHmm'
    Write-Host "`n[5/6] ACR 云端 build CPU 镜像 tag=$tag（首次 8-12 分钟）..." -ForegroundColor Yellow
    az acr build `
        --registry $ACR_NAME `
        --image "video-poc-cpu:$tag" `
        --image "video-poc-cpu:latest" `
        --file "$root\Dockerfile.cpu" `
        $root
    $global:IMAGE_TAG = $tag
    Write-Host "  镜像：${ACR_LOGIN}/video-poc-cpu:$tag" -ForegroundColor Green
} else {
    $global:IMAGE_TAG = 'latest'
}

# ------------------------ 6. Helm 部署 --------------------------
if (-not $SkipHelm) {
    Write-Host "`n[6/6] Helm 部署 workload ..." -ForegroundColor Yellow
    $chart = "$root\charts\video-poc"
    if (-not (Test-Path $chart)) {
        Write-Host "  (跳过：$chart 不存在，先手动 kubectl apply / 完善 chart)" -ForegroundColor DarkYellow
    } else {
        helm upgrade --install video-poc $chart `
            -n video-poc --create-namespace `
            --set image.repository="$ACR_LOGIN/video-poc-cpu" `
            --set image.tag="$IMAGE_TAG" `
            --set storage.blobAccount="$STORAGE" `
            --set storage.filesAccount="$FILES" `
            --set appInsights.connectionString="$AI_CONN" `
            --wait --timeout 10m
        Write-Host "  查看 pod：kubectl -n video-poc get pods" -ForegroundColor Green
        Write-Host "  查看服务：kubectl -n video-poc get svc" -ForegroundColor Green
    }
}

Write-Host "`n✓ 部署完成。" -ForegroundColor Cyan
Write-Host "下一步：" -ForegroundColor Cyan
Write-Host "  - 上传模型：pwsh infra\upload-models.ps1 -Storage $STORAGE" -ForegroundColor Gray
Write-Host "  - 上传数据集：pwsh infra\upload-datasets.ps1 -Storage $STORAGE" -ForegroundColor Gray
Write-Host "  - GPU 配额到位后：pwsh infra\add-gpu-pool.ps1 -RG $RG -Cluster $AKS_NAME" -ForegroundColor Gray
