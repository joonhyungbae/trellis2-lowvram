"""Per-stage VRAM / time measurement, OOM capture, and the simulated VRAM cap.

Upstream already moves each stage's model to the GPU and back (`low_vram=True`). What it does not
do is tell you how much memory each stage needed. StageMonitor wraps the stage methods of a
pipeline instance and records, per call: wall time, peak allocated, peak reserved, and device-wide
used memory. These numbers are the only basis for any "fits in N GB" claim (HANDOFF rule 3).
"""

from __future__ import annotations

import functools
import gc
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

log = logging.getLogger(__name__)

MIB = 1024**2
GIB = 1024**3


@dataclass
class StageRecord:
    stage: str
    seconds: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    device_used_mib: float  # whole device at stage end (includes other processes and the context)
    ok: bool = True


class StageOOM(RuntimeError):
    """A stage ran out of GPU memory. The pipeline's models have been moved back to the CPU."""

    def __init__(self, stage: str, detail: str):
        super().__init__(f"out of GPU memory in stage {stage}: {detail}")
        self.stage = stage
        self.detail = detail


@dataclass
class StageMonitor:
    records: list[StageRecord] = field(default_factory=list)
    on_stage: Callable[[str], None] | None = None  # progress callback (UI)
    on_oom: Callable[[], None] | None = None  # free everything when a stage fails

    def wrap(self, stage: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.on_stage:
                self.on_stage(stage)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                out = fn(*args, **kwargs)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError as e:
                self._record(stage, t0, ok=False)
                log.warning("OOM in %s: %s", stage, e)
                if self.on_oom:
                    self.on_oom()
                release()
                raise StageOOM(stage, str(e).splitlines()[0]) from e
            self._record(stage, t0, ok=True)
            release()
            return out

        return wrapper

    def _record(self, stage: str, t0: float, ok: bool) -> None:
        free, total = torch.cuda.mem_get_info()
        r = StageRecord(
            stage=stage,
            seconds=round(time.perf_counter() - t0, 2),
            peak_allocated_mib=round(torch.cuda.max_memory_allocated() / MIB),
            peak_reserved_mib=round(torch.cuda.max_memory_reserved() / MIB),
            device_used_mib=round((total - free) / MIB),
            ok=ok,
        )
        self.records.append(r)
        log.info("stage %s", asdict(r))

    def peak_mib(self) -> float:
        return max((r.peak_reserved_mib for r in self.records), default=0)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.records]


def release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def apply_vram_cap(cap_gib: float | None) -> dict[str, Any]:
    """Simulate a smaller card by capping what PyTorch may allocate on device 0.

    The CUDA context and other processes sit outside this cap, so a cap is not identical to a
    real card of that size (docs/PLAN.md 0: we use 14.5 to stand in for 16GB).
    """
    total = torch.cuda.get_device_properties(0).total_memory
    info: dict[str, Any] = {"device_total_gib": round(total / GIB, 2), "cap_gib": cap_gib}
    if cap_gib:
        fraction = min(1.0, cap_gib * GIB / total)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        info["fraction"] = round(fraction, 4)
        log.info("VRAM cap %.2f GiB (fraction %.4f of %.2f GiB)", cap_gib, fraction, total / GIB)
    return info
