@echo off
set "SCRIPT_DIR=%~dp0"
cd /d "%~dp0."
uv run --locked gpt-sovits-webui zh_CN
pause
