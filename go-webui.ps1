$ErrorActionPreference = "Stop"
chcp 65001
Set-Location $PSScriptRoot
uv run --locked gpt-sovits-webui zh_CN
pause
