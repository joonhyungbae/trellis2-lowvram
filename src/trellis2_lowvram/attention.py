"""Attention backend selection.

Upstream reads ATTN_BACKEND / SPARSE_ATTN_BACKEND once, at import time
(trellis2/modules/attention/config.py, trellis2/modules/sparse/config.py), so `apply()` must run
before anything under `trellis2` is imported.

Upstream sparse (varlen) attention supports only xformers / flash_attn / flash_attn_3; there is
no SDPA path. Dense attention may fall back to SDPA. See docs/PLAN.md 2.4.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Backend:
    dense: str
    sparse: str | None  # None: no usable sparse backend, generation cannot run


def _importable(name: str) -> bool:
    if importlib.util.find_spec(name) is None:
        return False
    try:
        importlib.import_module(name)
        return True
    except Exception as e:  # a wheel built for another torch/CUDA fails here, not in find_spec
        log.debug("backend %s present but failed to import: %r", name, e)
        return False


def _flash_attn_supported() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)


def choose_backend(force: str | None = None) -> Backend:
    force = force or os.environ.get("TRELLIS2_LOWVRAM_ATTN")
    if force:
        sparse = force if force in ("flash_attn", "xformers") else None
        return Backend(dense=force, sparse=sparse)
    if _importable("flash_attn") and _flash_attn_supported():
        return Backend("flash_attn", "flash_attn")
    if _importable("xformers.ops"):
        return Backend("xformers", "xformers")
    return Backend("sdpa", None)


def apply(backend: Backend) -> None:
    os.environ["ATTN_BACKEND"] = backend.dense
    if backend.sparse:
        os.environ["SPARSE_ATTN_BACKEND"] = backend.sparse
    log.info("attention backend: dense=%s sparse=%s", backend.dense, backend.sparse)
