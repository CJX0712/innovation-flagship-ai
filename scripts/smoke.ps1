# ifai 冒烟测试 (Windows PowerShell) — 跑通核心 CLI 子命令, 验证端到端可运行。
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
Write-Host "== ifai smoke (ps1) =="
& $py -m ifai.cli version
& $py -m ifai.cli doctor
& $py -m ifai.cli tokenize --text "人工智能 模型"
& $py -m ifai.cli search --query "注意力"
& $py -m ifai.cli demo --greedy --max-new 8
& $py -m ifai.cli eval
Write-Host "== smoke OK =="
