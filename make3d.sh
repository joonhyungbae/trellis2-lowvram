#!/usr/bin/env bash
# Make 3D models from pictures: ./make3d.sh photo1.jpg [photo2.png ...]  (results in outputs/)
cd "$(dirname "$0")"
[ $# -gt 0 ] || { echo "usage: ./make3d.sh picture [picture ...]"; exit 1; }
args=()
for f in "$@"; do args+=("$(cd "$OLDPWD" && realpath "$f")"); done
exec ./trellis2.sh generate "${args[@]}" --yes --open
