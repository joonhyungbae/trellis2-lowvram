"""GPU smoke test for testers: environment report + one measured generation.

    uv run --frozen python scripts/gpu_smoke_test.py [--vram-cap 14.5] [--preset fast]
    (Windows, after install.bat: .tools/uv/uv.exe run --frozen python scripts/gpu_smoke_test.py)

Writes logs/smoke-<time>/ with doctor.json, benchmark.json and benchmark.md. The markdown is
ready to paste into docs/BENCHMARKS.md (report form at the bottom of that file).
"""

from __future__ import annotations

import argparse
import sys
import time

from trellis2_lowvram.cli import main as cli_main
from trellis2_lowvram.config import get_paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vram-cap", type=float)
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    folder = get_paths().logs / f"smoke-{time.strftime('%Y%m%d-%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=True)
    cli_main(["doctor", "--json", str(folder / "doctor.json")])
    bench = ["benchmark", "--preset", args.preset, "--seeds", str(args.seeds), "--out", str(folder)]
    if args.vram_cap:
        bench += ["--vram-cap", str(args.vram_cap)]
    code = cli_main(bench)
    print(f"\nAttach this folder to your report / 이 폴더를 제보에 첨부해 주세요: {folder}")
    return code


if __name__ == "__main__":
    sys.exit(main())
