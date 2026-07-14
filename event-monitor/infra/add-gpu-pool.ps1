# =============================================================
# GPU 配额批下来后，一键加 GPU node pool（不动 CPU pool、不重建集群）
#
# 用法：
#   pwsh infra\add-gpu-pool.ps1 -RG videopoc-rg -Cluster <aks-name>
#
# 之后升级 helm：
#   helm upgrade video-poc charts\video-poc -n video-poc `
#     --set gpu.enabled=true --set gpu.image.tag=<gpu-tag> --reuse-values
# =============================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$RG,
    [Parameter(Mandatory=$true)][string]$Cluster,

    # 默认 T4 v3（Southeast Asia GA，配额易批）；若批到 A10 v4 换 Standard_NC32ads_A10_v4；A100 换 Standard_NC24ads_A100_v4
    [string]$Sku = 'Standard_NC4as_T4_v3',
    [int]$MinNodes = 0,           # ← 缩到 0 是省钱关键
    [int]$MaxNodes = 3,
    [switch]$OnDemand             # 默认 Spot；加 -OnDemand 走按需
)
$ErrorActionPreference = 'Stop'

$priorityArgs = @()
if (-not $OnDemand) {
    $priorityArgs = @('--priority', 'Spot', '--spot-max-price', '-1', '--eviction-policy', 'Delete')
}

Write-Host "==> 加 GPU pool  sku=$Sku  min=$MinNodes max=$MaxNodes  " `
           ($(if($OnDemand){'按需'}else{'Spot'})) -ForegroundColor Cyan

az aks nodepool add `
    --resource-group $RG `
    --cluster-name $Cluster `
    --name gpupool `
    --node-vm-size $Sku `
    --node-count $MinNodes `
    --min-count $MinNodes `
    --max-count $MaxNodes `
    --enable-cluster-autoscaler `
    --node-taints 'sku=gpu:NoSchedule' `
    --labels workload=gpu `
    --mode User `
    @priorityArgs `
    -o table

Write-Host "`n✓ GPU pool 已加。接下来：" -ForegroundColor Green
Write-Host "  1) build GPU 镜像：" -ForegroundColor Gray
Write-Host "     az acr build -r <acrName> -t video-poc-gpu:latest -f Dockerfile.gpu ." -ForegroundColor Gray
Write-Host "  2) 打开 helm gpu 开关：" -ForegroundColor Gray
Write-Host "     helm upgrade video-poc charts\video-poc -n video-poc \``
     --set gpu.enabled=true --set gpu.image.repository=<acr>/video-poc-gpu \``
     --set gpu.image.tag=latest --reuse-values" -ForegroundColor Gray
Write-Host "  3) 验证 GPU pod 拿到卡：" -ForegroundColor Gray
Write-Host "     kubectl -n video-poc exec deploy/video-poc-gpu -- nvidia-smi" -ForegroundColor Gray
