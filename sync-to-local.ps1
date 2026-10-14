# sync-to-local.ps1 — 把 bundle 里的 skill 同步到本地 DSH skill 目录
# 用法：cd 到 dsh-skill-ballistic，运行  .\sync-to-local.ps1
# 说明：只覆盖、不删除（保留本地 .env 真实密钥）；排除 __pycache__

$src = Join-Path $PSScriptRoot "skills\ballistic-simulation"
$dst = Join-Path $env:USERPROFILE ".dsh\skills\ballistic-simulation"

if (-not (Test-Path $src)) {
    Write-Host "错误：找不到 $src" -ForegroundColor Red
    exit 1
}
New-Item -ItemType Directory -Force -Path $dst | Out-Null
robocopy $src $dst /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ | Out-Null

Write-Host "已同步到: $dst" -ForegroundColor Green
Write-Host "（DSH 会自动热加载，无需重启）"