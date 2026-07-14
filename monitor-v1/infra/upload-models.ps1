# =============================================================
# 一次性把本地模型权重上传到 Azure Blob，供 AKS pod 挂 /models 使用
#
# 用法：
#   pwsh infra\upload-models.ps1 -Storage <storageAccountName>
# =============================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Storage,
    [string]$Container = 'models'
)
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Write-Host "==> 上传模型到 $Storage/$Container" -ForegroundColor Cyan

# 需要上传的文件/目录（存在才上传）
$targets = @(
    @{ src = "$root\yolov8m.pt";        dest = "ultralytics/yolov8m.pt" }
    @{ src = "$root\yolov8m-seg.pt";    dest = "ultralytics/yolov8m-seg.pt" }
    @{ src = "$root\yolov8n-pose.pt";   dest = "ultralytics/yolov8n-pose.pt" }
    @{ src = "$env:USERPROFILE\.insightface\models\buffalo_l"; dest = "insightface/models/buffalo_l"; recursive = $true }
    @{ src = "$root\gfpgan";            dest = "gfpgan";        recursive = $true }
    @{ src = "$env:USERPROFILE\.cache\torch\hub\checkpoints"; dest = "torch/hub/checkpoints"; recursive = $true }
)

foreach ($t in $targets) {
    if (-not (Test-Path $t.src)) {
        Write-Host "  ! 跳过（不存在）：$($t.src)" -ForegroundColor DarkYellow
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
}
Write-Host "✓ 模型上传完成。" -ForegroundColor Green
