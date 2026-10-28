"""Turn an exception into "what happened / what to do" (HANDOFF rule 4) for the CLI and web UI.

Only causes we can name with confidence are mapped; anything else stays "unexpected error, please
send the log". Out-of-memory is not here: generate.make_one handles it as StageOOM.
"""

from __future__ import annotations

import errno
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Explained:
    msg: str  # i18n key
    fix: str  # i18n key
    params: dict[str, Any] = field(default_factory=dict)


def _chain(e: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def _hf_errors() -> tuple[tuple[type, ...], tuple[type, ...]]:
    """(missing or not allowed, cannot connect). Empty when huggingface_hub is not installed."""
    try:
        from huggingface_hub.errors import (
            GatedRepoError,
            LocalEntryNotFoundError,
            OfflineModeIsEnabled,
            RepositoryNotFoundError,
        )
    except ImportError:
        return (), ()
    return (GatedRepoError, RepositoryNotFoundError, LocalEntryNotFoundError), (
        OfflineModeIsEnabled,
    )


def explain(e: BaseException) -> Explained | None:
    from .images import BadImage

    hf_missing, hf_offline = _hf_errors()
    for x in _chain(e):
        if isinstance(x, BadImage):
            return Explained("error.bad_image", "fix.bad_image", {"path": x.path.name})
        if isinstance(x, OSError) and x.errno == errno.ENOSPC:
            return Explained("error.no_space", "fix.disk.low")
        if isinstance(x, ImportError):
            return Explained("error.components", "fix.reinstall", {"name": x.name or "?"})
        if hf_offline and isinstance(x, hf_offline):
            return Explained("error.network", "fix.hf.network")
        if type(x).__name__ in ("ConnectionError", "ConnectTimeout", "ReadTimeout"):
            return Explained("error.network", "fix.hf.network")
        if hf_missing and isinstance(x, hf_missing):
            return Explained("error.models", "fix.models.missing")
        text = str(x)
        if "Torch not compiled with CUDA" in text or "No CUDA GPUs are available" in text:
            return Explained("error.no_gpu", "fix.torch.no_cuda")
    return None
