"""One generation job: image -> (several seeds) -> GLB/OBJ + a VRAM/time report.

Shared by the CLI (`generate`, `benchmark`), scripts/gpu_smoke_test.py and the web UI (make_one).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .env_check import os_name
from .memory import StageMonitor, StageOOM, apply_vram_cap
from .presets import Settings, lower

log = logging.getLogger(__name__)

# ask(stage, current, proposed) -> True to retry with `proposed`
AskRetry = Callable[[str, Settings, Settings], bool]


@dataclass
class SeedResult:
    seed: int
    ok: bool
    settings: dict[str, Any]
    files: list[str] = field(default_factory=list)
    seconds: float = 0.0
    oom_stage: str | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Job:
    image: Path
    out_dir: Path
    settings: Settings
    seeds: list[int]
    vram_cap_gib: float | None = None
    obj: bool = True
    remove_background: bool = True


def make_one(
    gen: Any,
    image: Any,
    seed: int,
    settings: Settings,
    out_base: Path,
    ask_retry: AskRetry | None = None,
    remove_background: bool = True,
    obj: bool = True,
) -> list[SeedResult]:
    """Generate one seed and export it, lowering settings after an OOM when ask_retry agrees.

    After an OOM with gpu_memory "high" the first step is "low", taken without asking.

    An OOM while exporting retries only the export with the mesh already made; an OOM while
    generating reruns the generation. Every attempt is returned (the last one is the outcome).
    """
    monitor = gen.monitor
    attempts: list[SeedResult] = []
    s, mesh = settings, None
    while True:
        start, t0 = len(monitor.records), time.perf_counter()
        try:
            if mesh is None:
                mesh = gen.generate(image.copy(), seed, s, remove_background)
            files = gen.export(mesh, out_base, s, obj=obj)
            attempts.append(
                SeedResult(
                    seed,
                    True,
                    asdict(s),
                    [str(f) for f in files],
                    round(time.perf_counter() - t0, 1),
                    stages=monitor.to_dicts()[start:],
                )
            )
            return attempts
        except StageOOM as e:
            attempts.append(
                SeedResult(
                    seed,
                    False,
                    asdict(s),
                    [],
                    round(time.perf_counter() - t0, 1),
                    oom_stage=e.stage,
                    stages=monitor.to_dicts()[start:],
                )
            )
            proposed = lower(s, e.stage)
            if proposed is None:
                return attempts
            # Giving back kept GPU memory lowers nothing the user sees: retry without asking.
            same_quality = proposed.gpu_memory != s.gpu_memory
            if not same_quality and (ask_retry is None or not ask_retry(e.stage, s, proposed)):
                return attempts
            if not e.stage.startswith("export") or proposed.pipeline_type != s.pipeline_type:
                mesh = None  # the mesh itself has to be made again
            s = proposed


def run_job(
    job: Job,
    ask_retry: AskRetry | None = None,
    on_stage: Callable[[str], None] | None = None,
    gen: Any = None,
) -> dict[str, Any]:
    """Pass `gen` to reuse an already loaded pipeline (several images in one session)."""
    import torch

    from .attention import choose_backend
    from .images import open_image
    from .pipeline import Generator

    cap = apply_vram_cap(job.vram_cap_gib)
    t_load = time.perf_counter()
    if gen is None:
        gen = Generator(job.settings.pipeline_type, StageMonitor(on_stage=on_stage))
    monitor = gen.monitor
    load_seconds = round(time.perf_counter() - t_load, 1)
    image = open_image(job.image)

    results: list[SeedResult] = []
    for seed in job.seeds:
        base = job.out_dir / f"{job.image.stem}_seed{seed}"
        results += make_one(
            gen, image, seed, job.settings, base, ask_retry, job.remove_background, job.obj
        )

    return {
        "tool": "trellis2-lowvram",
        "version": __version__,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "image": job.image.name,
        "system": {
            "os": os_name(),
            "gpu": torch.cuda.get_device_name(0),
            "driver_cuda": torch.version.cuda,
            "torch": torch.__version__,
            "attention": asdict(choose_backend()),
            **cap,
        },
        "load_seconds": load_seconds,
        "results": [asdict(r) for r in results],
        "peak_reserved_mib": monitor.peak_mib(),
    }


def write_report(report: dict[str, Any], out_dir: Path, name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"{name}.json"
    mp = out_dir / f"{name}.md"
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    mp.write_text(report_markdown(report), encoding="utf-8")
    return jp, mp


def report_markdown(r: dict[str, Any]) -> str:
    sysinfo = r["system"]
    cap = f"{sysinfo['cap_gib']} GiB (simulated)" if sysinfo.get("cap_gib") else "none"
    lines = [
        f"## {sysinfo['gpu']} / {sysinfo['os']}",
        "",
        f"- trellis2-lowvram {r['version']}, torch {sysinfo['torch']}, "
        f"attention {sysinfo['attention']['dense']}/{sysinfo['attention']['sparse']}",
        f"- device memory {sysinfo['device_total_gib']} GiB, cap {cap}",
        f"- image `{r['image']}`, model load {r['load_seconds']} s",
        f"- peak reserved over the run: {r['peak_reserved_mib']} MiB",
        "",
    ]
    for res in r["results"]:
        st = res["settings"]
        status = "ok" if res["ok"] else f"OOM in {res['oom_stage']}"
        lines += [
            f"### seed {res['seed']}: {status} ({res['seconds']} s)",
            "",
            f"pipeline {st['pipeline_type']}, steps {st['steps']}, texture {st['texture_size']}, "
            f"decimation {st['decimation_target']}",
            "",
            "| stage | s | peak alloc MiB | peak reserved MiB | device used MiB |",
            "|---|---:|---:|---:|---:|",
        ]
        for x in res["stages"]:
            mark = "" if x["ok"] else " (OOM)"
            lines.append(
                f"| {x['stage']}{mark} | {x['seconds']} | {x['peak_allocated_mib']} | "
                f"{x['peak_reserved_mib']} | {x['device_used_mib']} |"
            )
        lines.append("")
    return "\n".join(lines)
