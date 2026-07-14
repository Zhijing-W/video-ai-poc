# =============================================================
# 把实验数据集（ChokePoint / Market-1501 等）上传到 Azure Blob，
# AKS pod 挂 /data 只读跑实验。
#
# 用法：
#   pwsh infra\upload-datasets.ps1 -Storage <storageAccountName>
# =============================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Storage,
    [string]$Container = 'datasets'
)
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Write-Host "==> 上传数据集到 $Storage/$Container" -ForegroundColor Cyan

$targets = @(
    @{ src = "$root\data\external\chokepoint";           dest = "chokepoint" }
    @{ src = "$root\data\external\Market-1501-v15.09.15"; dest = "market1501" }
    @{ src = "$root\data\samples";                       dest = "samples" }   # 演示视频
)

foreach ($t in $targets) {
    if (-not (Test-Path $t.src)) {
        Write-Host "  ! 跳过（不存在）：$($t.src)" -ForegroundColor DarkYellow
        continue
    }
    $sizeGB = [math]::Round(((Get-ChildItem $t.src -Recurse -File -EA SilentlyContinue | Measure-Object Length -Sum).Sum/1GB), 2)
    Write-Host "  → $($t.src)  →  $Container/$($t.dest)  (${sizeGB} GB)" -ForegroundColor Yellow
    az storage blob upload-batch `
        --account-name $Storage `
        --auth-mode login `
        --destination $Container `
        --destination-path $t.dest `
        --source $t.src `
        --overwrite -o none
}
Write-Host "✓ 数据集上传完成。" -ForegroundColor Green
