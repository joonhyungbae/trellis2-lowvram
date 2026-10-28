@echo off
rem Drag one or more pictures onto this file to make 3D models (balanced quality).
rem Results go to the outputs folder, which opens when it is done.
chcp 65001 >nul
cd /d "%~dp0"
if "%~1"=="" (
  echo 사진 파일을 이 파일 위로 끌어다 놓으세요. 여러 장도 됩니다.
  echo Drag picture files onto this file. Several at once is fine.
  echo.
  pause
  exit /b 1
)
call "%~dp0trellis2.bat" generate %* --yes --open
echo.
pause
