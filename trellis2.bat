@echo off
rem Run a trellis2-lowvram command with the installed environment: trellis2 doctor
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "UV=uv"
if exist "%~dp0.tools\uv\uv.exe" (
  set "UV=%~dp0.tools\uv\uv.exe"
  set "UV_PYTHON_INSTALL_DIR=%~dp0.tools\python"
  set "UV_CACHE_DIR=%~dp0.tools\uv-cache"
)
"%UV%" run --frozen --no-dev trellis2-lowvram %*
