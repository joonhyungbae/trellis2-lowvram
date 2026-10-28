# Upstream pins

This tool wraps upstream code without copying or editing it. Everything it depends on is pinned here.

| What | Where pinned | Pin |
|---|---|---|
| microsoft/TRELLIS.2 (Python package `trellis2`) | `third_party/TRELLIS.2` submodule, `UPSTREAM_COMMIT` in `install.sh` / `scripts/install.ps1` / `config.py` | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` (2026-06-05) |
| O-Voxel (inside TRELLIS.2) | `wheels/sources.toml` | same commit |
| FlexGEMM | `wheels/sources.toml` | `6dd94a859c26ee8246888502eada3dd8ad85532e` |
| CuMesh | `wheels/sources.toml` | `12289e1062f0603f2f0d0771b02e1395d247f26f` |
| nvdiffrast | `wheels/sources.toml` | `253ac4fcea7de5f396371124af597e6cc957bfae` (v0.4.0) |
| Python packages | `uv.lock` | torch 2.8.0+cu128, xformers 0.0.32.post2, transformers 4.57.3, ... |
| Model weights | downloaded at install from Hugging Face (`models.py`) | `microsoft/TRELLIS.2-4B` main |

## Patches

None. `patches/` (for TRELLIS.2) and `wheels/patches/<ext>/` (for extensions) are applied in
sorted order; if one stops applying, install / the wheel build stops with the failing file name.

## Upstream code this wrapper relies on

If any of these change, update the wrapper before moving the pin.

- `ATTN_BACKEND` / `SPARSE_ATTN_BACKEND` read at import time
  (`trellis2/modules/attention/config.py`, `trellis2/modules/sparse/config.py`)
- `pipeline.json` keys under `args.models`, `image_cond_model`, `rembg_model`
- `trellis2.models.from_pretrained(path)` resolving `<org>/<repo>/<file>`
- `Trellis2ImageTo3DPipeline` stage methods and `low_vram` attribute (wrapped in M2)

## Moving the pin

1. `git -C third_party/TRELLIS.2 fetch && git -C third_party/TRELLIS.2 checkout <new>`
2. Update the commit in this file, `config.py`, `install.sh`, `scripts/install.ps1`, `wheels/sources.toml`.
3. Read the upstream diff for the items listed above.
4. Rebuild wheels (Actions: build-wheels) and update `wheels/manifest.json`.
5. Run `scripts/gpu_smoke_test.py` and compare against `docs/BENCHMARKS.md`.
