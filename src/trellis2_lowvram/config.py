"""Paths and environment that must be fixed before torch or upstream code is imported."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Repo root: src/trellis2_lowvram/config.py -> ../../..
REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = REPO_ROOT / "third_party" / "TRELLIS.2"
UPSTREAM_COMMIT = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"


@dataclass(frozen=True)
class Paths:
    home: Path
    models: Path
    logs: Path
    outputs: Path


def get_paths() -> Paths:
    """Every location can be moved with an environment variable; nothing is hardcoded elsewhere."""
    home = Path(os.environ.get("TRELLIS2_LOWVRAM_HOME", REPO_ROOT)).expanduser().resolve()
    models = Path(os.environ.get("TRELLIS2_LOWVRAM_MODELS", home / "models")).expanduser().resolve()
    logs = Path(os.environ.get("TRELLIS2_LOWVRAM_LOGS", home / "logs")).expanduser().resolve()
    outputs = (
        Path(os.environ.get("TRELLIS2_LOWVRAM_OUTPUTS", home / "outputs")).expanduser().resolve()
    )
    return Paths(home=home, models=models, logs=logs, outputs=outputs)


def prepare_environment() -> Paths:
    """Call once at process start, before importing torch, transformers or trellis2."""
    paths = get_paths()
    # Keep the Hugging Face cache and token inside our models folder so that deleting the
    # install folder removes everything, and the student's global HF setup is not touched.
    os.environ.setdefault("HF_HOME", str(paths.models / "huggingface"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    if UPSTREAM_DIR.is_dir() and str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    return paths
