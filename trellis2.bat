@echo off
rem Run a trellis2-lowvram command with the installed environment: trellis2 doctor
rem --no-sync: never touch the environment here (install/update do that). Syncing while another
rem window (e.g. the web UI) runs fails because Windows locks the running .exe.
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo 아직 설치되지 않았습니다. 먼저 install.bat 을 더블클릭해 설치해 주세요.
  echo Not installed yet. Double-click install.bat first.
  echo.
  exit /b 1
)
set "UV=uv"
if exist "%~dp0.tools\uv\uv.exe" (
  set "UV=%~dp0.tools\uv\uv.exe"
  set "UV_PYTHON_INSTALL_DIR=%~dp0.tools\python"
  set "UV_CACHE_DIR=%~dp0.tools\uv-cache"
)
"%UV%" run --no-sync python -m trellis2_lowvram %*
