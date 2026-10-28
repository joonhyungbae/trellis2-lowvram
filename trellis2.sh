#!/usr/bin/env bash
# Run a trellis2-lowvram command with the installed environment: ./trellis2.sh doctor
cd "$(dirname "$0")"
UV=uv
if [ -x .tools/uv/uv ]; then
  # Same locations install.sh used, so nothing is downloaded again.
  UV="$PWD/.tools/uv/uv"
  export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python" UV_CACHE_DIR="$PWD/.tools/uv-cache"
fi
# --no-sync: never touch the environment here (install/update do that). Syncing while another
# window (e.g. the web UI) runs fails on Windows because the running .exe is locked.
exec "$UV" run --no-sync python -m trellis2_lowvram "$@"
