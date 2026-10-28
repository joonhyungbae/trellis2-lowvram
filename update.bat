@echo off
chcp 65001 >nul
cd /d "%~dp0"
set HAVE_GIT=0
if exist .git where git >/dev/null 2>/dev/null && set HAVE_GIT=1
if "%HAVE_GIT%"=="1" (
  git pull --ff-only
  git submodule update --init third_party/TRELLIS.2
) else (
  echo git 이 없어 코드를 자동으로 받지 못했습니다. 새 zip 을 받아 이 폴더에 덮어쓴 뒤 다시 실행해 주세요.
  echo git is not available. Download the new zip, extract it over this folder, then run this again.
)
call "%~dp0install.bat"
