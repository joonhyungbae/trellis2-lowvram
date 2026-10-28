#!/usr/bin/env bash
# Open the web interface in your browser: ./run.sh   (stop with Ctrl+C)
cd "$(dirname "$0")"
exec ./trellis2.sh ui "$@"
