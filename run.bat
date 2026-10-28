@echo off
rem Double-click to open the web interface in your browser. Close this window to stop it.
cd /d "%~dp0"
call "%~dp0trellis2.bat" ui %*
pause
