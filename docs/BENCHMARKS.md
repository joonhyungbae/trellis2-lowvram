# Benchmarks

Only measured numbers go here. Each row comes from a `benchmark` report (JSON + Markdown) made by
`trellis2-lowvram benchmark` or `scripts/gpu_smoke_test.py`. "Simulated" means a PyTorch allocation
cap on a bigger card (`--vram-cap`), which is **not** the same as a real card of that size: the
CUDA context, the display and other programs sit outside the cap.

Peak = the highest `max_memory_reserved` of any single stage (PyTorch's own memory, per stage).
Device used = whole-card usage at the end of the heaviest stage, including other programs.

## Results

| GPU (VRAM) | OS | Cap | Input | Pipeline / preset | Result | Time (s) | Peak reserved (MiB) | Device used (MiB) | Date |
|---|---|---|---|---|---|---:|---:|---:|---|
| RTX 4090 (24GB) | Ubuntu 24.04 | 14.5 GiB simulated | T.png (transparent) | 512 / balanced | ok | 86.2 (first run) | 2786 | 6234 | 2026-10-28 |
| RTX 4090 (24GB) | Ubuntu 24.04 | 14.5 GiB simulated | T.png on a flat background (JPG, background removal) | 512 / balanced | ok | 44.4 | 3634 | 7068 | 2026-10-28 |
| RTX 3080 (10GB) | Windows 11 | none (real card) | T.png (transparent) | 512 / balanced | ok | 123.5 (first run) | 3594 | 4800 | 2026-10-28 |
| RTX 3080 (10GB) | Windows 11 | none (real card) | T.png on a flat background, 2 seeds | 512 / quality (texture 4096, 1M faces) | ok, ok | 108.2 / 78.6 | 4634 | 5826 | 2026-10-28 |
| RTX 4090 (24GB) | Ubuntu 24.04 | 14.5 GiB simulated | T.png on a flat background | 1024_cascade / balanced | ok | 114.2 | 8524 | 12543 | 2026-10-28 |
| RTX 3080 (10GB) | Windows 11 | none (real card) | T.png on a flat background | 1024_cascade / balanced | ok, **spilled into shared memory** | 314.0 | 12334 | 10240 (full) | 2026-10-28 |
| RTX 4090 (24GB) | Ubuntu 24.04 | 14.5 GiB simulated | 8 sample images ([examples](examples/README.md)) | 512 / balanced | 8 ok | 29.6-52.9 | 2728-6468 | | 2026-10-28 |

Notes:
- Model loading (from disk to RAM) is not included in Time: 56 s (Linux) and 91 s (Windows) above.
- The first generation on a PC compiles Triton kernels for FlexGEMM, which makes `shape_decoder`
  much slower once (47 s vs 9 s on the 4090). Later runs reuse the cache.
- On the RTX 4090 the "device used" column includes about 3.3 GB used by other programs.
- **Windows does not stop at the card's limit.** The NVIDIA driver lends system RAM ("shared GPU
  memory") when VRAM runs out, so 1024_cascade on the 10GB RTX 3080 finished, but its decoders
  reserved 12.3 GB and the run took about 2.5x longer than 512. Linux drivers have no such
  fallback, so there the same run is expected to stop with out-of-memory, which the tool reports
  and offers to retry at 512 (not tested on a 10GB Linux card).
- 1024_cascade peaks in the shape/texture decoders and in GLB baking (about 8-12 GB);
  512 stays under about 6.5 GB in every run so far.
- Not measured yet: a real 16GB card.

## Per-stage example (RTX 3080 10GB, Windows 11, 512 / balanced)

| stage | s | peak alloc MiB | peak reserved MiB | device used MiB |
|---|---:|---:|---:|---:|
| image_cond | 5.25 | 1213 | 1250 | 2440 |
| sparse_structure | 11.01 | 2715 | 2954 | 4152 |
| shape_flow | 9.85 | 2701 | 2860 | 4058 |
| texture_flow | 4.75 | 2703 | 2862 | 4060 |
| shape_decoder | 56.78 | 2296 | 3308 | 4514 |
| texture_decoder | 0.71 | 2424 | 3594 | 4800 |
| export_simplify | 0.0 | 169 | 2188 | 3394 |
| export_glb | 29.0 | 1858 | 1978 | 3184 |

## Report your GPU

1. Install with `install.bat` / `install.sh`.
2. Run the smoke test:
   - Windows: `.tools\uv\uv.exe run --frozen python scripts\gpu_smoke_test.py`
   - Ubuntu: `.tools/uv/uv run --frozen python scripts/gpu_smoke_test.py`
3. Open an issue titled `benchmark: <GPU> <VRAM> <OS>` and attach the `logs/smoke-<time>/`
   folder it prints (doctor.json, benchmark.json, benchmark.md). Paste benchmark.md into the issue
   text. Remove the generated `.glb` / `_obj` files if they are too large to attach.
