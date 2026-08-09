# =============================================================
# 一次性把本地模型权重上传到 Azure Blob，供 AKS pod 挂 /models 使用
#
# 用法：
#   pwsh infra\upload-models.ps1 -Storage <storageAccountName>
# =============================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Storage,
    [string]$Container = 'models',
    [string]$ModelRoot = "$env:USERPROFILE\.cache\event-monitor\models"
)
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Write-Host "==> 上传模型到 $Storage/$Container" -ForegroundColor Cyan

$differRoot = Join-Path $ModelRoot "reid\differ"
$requiredDifferAssets = @(
    @{ path = Join-Path $differRoot "source"; label = "DIFFER 官方源码目录"; type = "Container" }
    @{ path = Join-Path $differRoot "config.yml"; label = "DIFFER 配置文件"; type = "Leaf" }
    @{ path = Join-Path $differRoot "eva02_l_bio_best.pth"; label = "DIFFER checkpoint"; type = "Leaf" }
)
$missingDifferAssets = @(
    foreach ($asset in $requiredDifferAssets) {
        if (-not (Test-Path -LiteralPath $asset.path -PathType $asset.type)) {
            "$($asset.label): $($asset.path)"
        }
    }
)
if ($missingDifferAssets.Count -gt 0) {
    $details = $missingDifferAssets -join [Environment]::NewLine
    throw @"
缺少默认 ReID 后端 DIFFER 的必需资产，未执行任何上传：
$details
请先运行 python scripts\download_models.py --reid --model-root "$ModelRoot"，或通过 -ModelRoot 指向完整模型目录。
"@
}

# 可选目标不存在时跳过；ReID 上传整个现有树以保留所有显式后端和 auto 回退。
$targets = @(
    @{ src = "$root\yolov8m.pt";        dest = "ultralytics/yolov8m.pt" }
    @{ src = "$root\yolov8m-seg.pt";    dest = "ultralytics/yolov8m-seg.pt" }
    @{ src = "$root\yolov8n-pose.pt";   dest = "ultralytics/yolov8n-pose.pt" }
    @{ src = "$env:USERPROFILE\.insightface\models\buffalo_l"; dest = "insightface/models/buffalo_l"; recursive = $true }
    @{ src = "$root\gfpgan";            dest = "gfpgan";        recursive = $true }
    @{ src = "$env:USERPROFILE\.cache\torch\hub\checkpoints"; dest = "torch/hub/checkpoints"; recursive = $true }
    @{ src = "$ModelRoot\reid"; dest = "reid"; recursive = $true }
)

$uploaded = 0
$skipped = 0
foreach ($t in $targets) {
    if (-not (Test-Path $t.src)) {
        Write-Host "  ! 跳过（不存在）：$($t.src)" -ForegroundColor DarkYellow
        $skipped++
        continue
    }
    Write-Host "  → $($t.src)  →  $Container/$($t.dest)" -ForegroundColor Yellow
    if ($t.recursive) {
        az storage blob upload-batch `
            --account-name $Storage `
            --auth-mode login `
            --destination $Container `
            --destination-path $t.dest `
            --source $t.src `
            --overwrite -o none
    } else {
        az storage blob upload `
            --account-name $Storage `
            --auth-mode login `
            --container-name $Container `
            --name $t.dest `
            --file $t.src `
            --overwrite -o none
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Azure CLI 上传失败（退出码 $LASTEXITCODE）：$($t.src) -> $Container/$($t.dest)。请检查 az 登录、Storage RBAC、容器名称和网络后重试。"
    }
    $uploaded++
}
Write-Host "✓ 模型上传完成：成功上传 $uploaded 个目标，跳过 $skipped 个不存在的可选目标；完整 ReID 树已上传。" -ForegroundColor Green
