"""Generate the example gallery (docs/examples) from upstream sample images.

    uv run --frozen python scripts/make_examples.py OUT_DIR [--vram-cap 14.5] [--preset balanced]

Loads the pipeline once, then for each image writes <name>.glb and <name>.jpg (preview next to
the input) plus examples.json with time and peak VRAM per example. GLBs are large (20-40MB);
they go to a GitHub release, the previews into docs/examples/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from trellis2_lowvram import attention
from trellis2_lowvram.config import UPSTREAM_DIR, prepare_environment

# name -> upstream assets/example_image file (MIT, microsoft/TRELLIS.2). Chosen to show both
# what works and what is hard: organic, hard-surface, scene, thin parts, character, transparency.
EXAMPLES = {
    "bust": "39488b45bb4820ff0f31bb07cb8d0a19ebd991adbcb22a10fc89ee41c59219ee",
    "pineapple": "5a020584b95cf3db3b6420e9b09fb93e7c0f4046e61076e5b4c65c63dc1f5837",
    "typewriter": "65433d02fc56dae164719ec29cb9646c0383aa1d0e24f0bb592899f08428d68e",
    "village": "4bc7abe209c8673dd3766ee4fad14d40acbed02d118e7629f645c60fd77313f1",
    "maple": "e134444178eae855cfdefb9e5259d076df5e34f780ee44d4ad604483ff69cc74",
    "stroller": "e513fcd6c897b249fc4bff54268b4d0bbab6403503ecf3846d92feb892536e5e",
    "mermaid": "8ce83f6a28910e755902de10918672e77dd23476f43f0f1521c48667de6cea84",
    "fishbowl": "b358d0eb96a68ac4ba1f2fb6d44ea2225f95fdfbf9cf4e0da08650c3704f1d23",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--vram-cap", type=float)
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    prepare_environment()
    attention.apply(attention.choose_backend())
    import torch

    from trellis2_lowvram.memory import StageMonitor, StageOOM, apply_vram_cap
    from trellis2_lowvram.pipeline import Generator
    from trellis2_lowvram.presets import resolve
    from trellis2_lowvram.preview import render

    cap = apply_vram_cap(args.vram_cap)
    settings = resolve(args.preset)
    monitor = StageMonitor()
    gen = Generator(settings.pipeline_type, monitor)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, stem in EXAMPLES.items():
        if args.only and name not in args.only:
            continue
        src = UPSTREAM_DIR / "assets" / "example_image" / f"{stem}.webp"
        from PIL import Image

        start, t0 = len(monitor.records), time.perf_counter()
        try:
            mesh = gen.generate(Image.open(src), 42, settings)
            gen.export(mesh, args.out / name, settings, obj=False)
            ok, oom = True, None
        except StageOOM as e:
            ok, oom = False, e.stage
        secs = round(time.perf_counter() - t0, 1)
        stages = monitor.to_dicts()[start:]
        if ok:
            render(str(args.out / f"{name}.glb"), str(args.out / f"{name}.jpg"), str(src))
        rows.append(
            {
                "name": name,
                "source": src.name,
                "ok": ok,
                "oom_stage": oom,
                "seconds": secs,
                "peak_reserved_mib": max((s["peak_reserved_mib"] for s in stages), default=0),
                "stages": stages,
            }
        )
        print(name, "ok" if ok else f"OOM {oom}", secs, rows[-1]["peak_reserved_mib"], flush=True)

    (args.out / "examples.json").write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                **cap,
                "settings": asdict(settings),
                "seed": 42,
                "examples": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
