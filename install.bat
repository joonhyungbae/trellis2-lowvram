@echo off
rem Double-click to install. Uses PowerShell; no admin rights needed.
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1"
echo.
pause
