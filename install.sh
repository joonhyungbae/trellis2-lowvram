#!/usr/bin/env bash
# One-step installer for Ubuntu. Everything goes inside this folder (.tools, .venv, models, logs);
# delete the folder to uninstall. Usage: ./install.sh
# If a step fails, it prints what that step needs and how to retry (messages in src/.../i18n).
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

UV_VERSION="0.11.8"
UPSTREAM_COMMIT="75fbf0183001ed9876c8dbb35de6b68552ee08bd"

mkdir -p logs
LOG="logs/install-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

export UV_INSTALL_DIR="$ROOT/.tools/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.tools/python"
export UV_CACHE_DIR="$ROOT/.tools/uv-cache"
export UV_NO_MODIFY_PATH=1
export PATH="$UV_INSTALL_DIR:$PATH"

STEP=0

# Translated text needs Python, which exists once uv is installed (step 2 on).
msg() {
  PYTHONPATH="$ROOT/src" uv run -q --no-project --managed-python --python 3.11 \
    python -m trellis2_lowvram.i18n "$@" 2>/dev/null || return 1
}

fail() {
  trap - ERR
  echo
  echo "[FAIL] step $STEP: $1"
  if [ "$STEP" -le 1 ] || ! msg "install.step$STEP.fail"; then
    echo "  [1단계: uv 받기] astral.sh 에서 설치 도구를 받지 못했습니다. 인터넷 연결과, 학교·회사 네트워크라면 astral.sh·github.com 이 막혀 있지 않은지 확인해 주세요."
    echo "  [Step 1: uv] Could not download the installer tool from astral.sh. Check your internet connection and, on a school or office network, that astral.sh and github.com are not blocked."
  fi
  echo
  msg install.retry || echo "Run ./install.sh again after fixing it. / 고친 뒤 ./install.sh 를 다시 실행해 주세요."
  msg install.log "log=$ROOT/$LOG" || echo "log: $ROOT/$LOG"
  exit 1
}
trap 'fail "$BASH_COMMAND (line $LINENO)"' ERR

STEP=1; echo "== 1/6 uv $UV_VERSION"
if ! "$UV_INSTALL_DIR/uv" --version 2>/dev/null | grep -q "$UV_VERSION"; then
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
fi
uv --version

STEP=2; echo "== 2/6 PC check (before the large download)"
PYTHONPATH="$ROOT/src" uv run --no-project --managed-python --python 3.11 --with nvidia-ml-py --with psutil \
  python -m trellis2_lowvram doctor --pre || fail "PC check"

STEP=3; echo "== 3/6 TRELLIS.2 source @ ${UPSTREAM_COMMIT:0:7}"
if [ ! -f third_party/TRELLIS.2/trellis2/__init__.py ]; then
  if [ -d .git ] && command -v git >/dev/null; then
    git submodule update --init third_party/TRELLIS.2
  else
    mkdir -p third_party/TRELLIS.2
    curl -LsSf "https://github.com/microsoft/TRELLIS.2/archive/$UPSTREAM_COMMIT.tar.gz" \
      | tar -xz --strip-components=1 -C third_party/TRELLIS.2
  fi
fi

STEP=4; echo "== 4/6 Python packages (PyTorch etc., several GB)"
uv sync --frozen --no-dev --python 3.11

STEP=5; echo "== 5/6 CUDA extensions"
uv run --frozen --no-dev trellis2-lowvram install-extensions

STEP=6; echo "== 6/6 Hugging Face access (DINOv3) and models"
MODELS_OK=1
if ! uv run --frozen --no-dev trellis2-lowvram download; then
  # Usually: no token or not approved yet. login explains the steps; then try once more.
  uv run --frozen --no-dev trellis2-lowvram login || true
  if ! uv run --frozen --no-dev trellis2-lowvram download; then
    MODELS_OK=0
    echo
    msg install.step6.fail || true
  fi
fi

trap - ERR
uv run --frozen --no-dev trellis2-lowvram doctor --json "logs/doctor.json" || true

# Menu entry for the web interface (opens a terminal, like run.sh). Not verified on a desktop.
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
if mkdir -p "$APPS" 2>/dev/null; then
  cat > "$APPS/trellis2-lowvram.desktop" <<DESKTOP || true
[Desktop Entry]
Type=Application
Name=trellis2-lowvram
Comment=Picture to 3D (TRELLIS.2)
Exec="$ROOT/run.sh"
Path=$ROOT
Terminal=true
Categories=Graphics;
DESKTOP
fi

echo
if [ "$MODELS_OK" = 1 ]; then msg install.done || true; else msg install.done_need_models || true; fi
echo
echo "log: $ROOT/$LOG"
