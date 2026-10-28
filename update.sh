#!/usr/bin/env bash
# Update this tool: new code, then the same install steps (already-installed parts are skipped).
set -euo pipefail
cd "$(dirname "$0")"
if [ -d .git ] && command -v git >/dev/null; then
  git pull --ff-only
  git submodule update --init third_party/TRELLIS.2
else
  echo "git 이 없어 코드를 자동으로 받지 못했습니다. 새 zip 을 받아 이 폴더에 덮어쓴 뒤 다시 실행해 주세요."
  echo "git is not available. Download the new zip, extract it over this folder, then run this again."
fi
exec ./install.sh
