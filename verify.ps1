# 验证 cordis.patch.yml 语法
# 前提：ballistic-test profile 已创建（上次运行已建好）。dump-config 只打印、直接退出，不启动服务。
$dsh = "D:\deepseek\resources\runtime\cli\bin\dsh.cmd"
$profile = "ballistic-test"

Write-Host "== 验证 cordis.patch.yml（profile: $profile）==" -ForegroundColor Cyan
$out = & $dsh --profile $profile --patch "$PSScriptRoot\cordis.patch.yml" --dump-config 2>&1
$out | Out-File "$PSScriptRoot\verify-output.txt" -Encoding utf8
$out | Select-Object -First 50

Write-Host "`n== 检查结果 ==" -ForegroundColor Cyan
$joined = $out -join "`n"
if ($joined -match "ballistic" -or $joined -match "skill") {
    Write-Host "OK: 检测到 skill 相关配置，patch 语法有效，可以继续上架。" -ForegroundColor Green
} else {
    Write-Host "未检测到 skill 配置。把 verify-output.txt 内容发我，我修 cordis.patch.yml。" -ForegroundColor Yellow
}