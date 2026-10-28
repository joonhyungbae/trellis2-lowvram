"""One generation job: image -> (several seeds) -> GLB/OBJ + a VRAM/time report.

Shared by the CLI (`generate`, `benchmark`), scripts/gpu_smoke_test.py and, later, the web UI.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
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


def run_job(
    job: Job,
    ask_retry: AskRetry | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    from .attention import choose_backend
    from .pipeline import Generator

    cap = apply_vram_cap(job.vram_cap_gib)
    monitor = StageMonitor(on_stage=on_stage)
    t_load = time.perf_counter()
    gen = Generator(job.settings.pipeline_type, monitor)
    load_seconds = round(time.perf_counter() - t_load, 1)
    image = Image.open(job.image)

    results: list[SeedResult] = []
    for seed in job.seeds:
        s = job.settings
        while True:
            start = len(monitor.records)
            t0 = time.perf_counter()
            try:
                mesh = gen.generate(image.copy(), seed, s)
                base = job.out_dir / f"{job.image.stem}_seed{seed}"
                files = gen.export(mesh, base, s, obj=job.obj)
                del mesh
                results.append(
                    SeedResult(
                        seed,
                        True,
                        asdict(s),
                        [str(f) for f in files],
                        round(time.perf_counter() - t0, 1),
                        stages=monitor.to_dicts()[start:],
                    )
                )
                break
            except StageOOM as e:
                proposed = lower(s, e.stage)
                results.append(
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
                if proposed is None or ask_retry is None or not ask_retry(e.stage, s, proposed):
                    break
                s = proposed

    return {
        "tool": "trellis2-lowvram",
        "version": __version__,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "image": job.image.name,
        "system": {
            "os": f"{platform.system()} {platform.release()}",
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
