"""Install the four CUDA extensions from prebuilt wheels.

Wheels are built by .github/workflows/build-wheels.yml from pinned sources (wheels/sources.toml)
and attached to a GitHub Release. wheels/manifest.json pins each file by URL and sha256; this
module refuses a download whose hash does not match.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

from .config import REPO_ROOT

log = logging.getLogger(__name__)

MANIFEST = REPO_ROOT / "wheels" / "manifest.json"


def platform_key() -> str:
    return {"Windows": "win_amd64", "Linux": "linux_x86_64"}.get(platform.system(), "unsupported")


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def entries_for(manifest: dict, key: str) -> list[dict]:
    return manifest.get("wheels", {}).get(key, [])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(entry: dict, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / entry["file"]
    if not (dest.exists() and sha256(dest) == entry["sha256"]):
        log.info("download %s", entry["url"])
        urllib.request.urlretrieve(entry["url"], dest)
    got = sha256(dest)
    if got != entry["sha256"]:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {entry['file']}: {got}")
    return dest


def install(dest_dir: Path) -> list[str]:
    """Returns installed file names. Uses `uv pip` when available (install scripts set UV)."""
    entries = entries_for(load_manifest(), platform_key())
    if not entries:
        raise RuntimeError(f"no prebuilt wheels for {platform_key()}")
    files = [fetch(e, dest_dir) for e in entries]
    # --no-deps: o_voxel declares cumesh/flex_gemm as git URLs; we install them ourselves.
    cmd = ["uv", "pip", "install", "--python", sys.executable, "--no-deps", *map(str, files)]
    log.info("%s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return [f.name for f in files]
