# HANDOFF: trellis2-lowvram

Read this first if you are an AI coding agent (Claude Code, Cursor, Codex, ...) or a new
contributor. It is the single entry point: what the project is, the rules, where things are,
what is verified, and what comes next. Keep it current when you change any of these.

## 1. What this is

An **unofficial wrapper** around [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
(image to 3D mesh with PBR textures, MIT). Goal: run the **same 4B model, unchanged**, on 16GB
consumer NVIDIA GPUs (RTX 30/40/50), with a one-click installer that non-experts (art students)
can use on Windows 10/11 (first) and Ubuntu 22.04/24.04 (second). macOS is out of scope.

The value is memory management, measurement and install automation. It is **not** a smaller or
quantized model; never describe it that way.

## 2. Rules (do not break)

1. **Wrapper, not fork.** Upstream lives in `third_party/TRELLIS.2` (git submodule, pinned).
   Never edit it. Changes go in `patches/` (applied at install) or runtime monkeypatches in
   `src/trellis2_lowvram/`. Pins and the update procedure: `docs/UPSTREAM.md`.
2. **Read before you use.** Upstream API, stages and dependencies must be confirmed in the
   actual code, not guessed. `docs/PLAN.md` section 2 lists what has been confirmed.
3. **Nothing unverified is stated as fact.** "Works on 16GB" may only be written with a
   measured report. Mark anything unmeasured as *unverified* in docs, comments and commits.
   There is no real 16GB card available: GPU checks run on an RTX 4090 24GB with a PyTorch
   allocation cap (14.5GB) and must be labeled "simulated" (PLAN section 0).
4. **All user-facing text in Korean and English**, via `src/trellis2_lowvram/i18n/{ko,en}.json`.
   Code refers to keys only. A unit test enforces identical keys and placeholders.
   Error messages: what happened, why (if known), what the user should do.
5. **Pin everything.** `uv.lock` for Python, commit hashes in `wheels/sources.toml` for CUDA
   extensions, submodule commit for upstream.
6. **Licenses.** This repo is MIT. Never commit model weights. Before redistributing any binary,
   confirm its license allows it. nvdiffrast is non-commercial (research/evaluation) and its
   wheels must carry the license text (`wheels/build.py` enforces this).
7. **Hardcoded paths are banned.** All locations come from `config.get_paths()` and can be moved
   with `TRELLIS2_LOWVRAM_HOME/_MODELS/_LOGS/_OUTPUTS`.
8. Use `logging`, not `print`, except for user-facing output in `cli.py` (`say()`).
9. Conventional Commits (`feat:`, `fix:`, `docs:`, `build:` ...), small commits.
10. Work milestone by milestone. Summarize at the end of each and get the maintainer's
    confirmation before starting the next.

## 3. Status

| Milestone | Scope | State |
|---|---|---|
| M0 | Research and design (`docs/PLAN.md`) | done, approved |
| M1 | Installer, `doctor`, HF access, model download, CUDA extension wheels | done (pending maintainer review) |
| M2 | Memory-managed generation CLI, per-stage VRAM report, OOM retry | **in progress**: generate/benchmark work on Linux and Windows; OOM retry verified (simulated cap) |
| M3 | Gradio web UI (upload, presets, multi-seed, preview, GLB/OBJ, ko/en) | **in progress**: works in headless Chrome on Linux; real-browser 3D view on Windows unverified |
| M4 | READMEs, TROUBLESHOOTING, BENCHMARKS, THIRD_PARTY_LICENSES, v0.1.0 | not started |

Verified so far (Linux, RTX 4090):
- Full `doctor` passes: GPU, driver, torch 2.8.0+cu128 sees the GPU, the four CUDA extensions
  import (built locally and in Actions from `wheels/sources.toml`), xformers chosen, DINOv3 access.
- `login` (token saved under `models/huggingface/`) and `download --pipeline 512` (about 11GB).
- `from trellis2.pipelines import Trellis2ImageTo3DPipeline` imports with our environment.
- Install failure guidance: a forced step-3 failure prints the step message, retry hint and log path.
- 21 CPU unit tests pass locally and in GitHub Actions (Ubuntu and Windows).

Verified on Windows 11 (pc2: RTX 3080 10GB, driver 610.88, ZIP download without git, C:\trellis2-lowvram):
- `scripts/install.ps1` steps 1-6 end to end, including the `wheels-v1` extension wheels and models.
- Full `doctor` passes except the expected 10GB VRAM warning.
- Upstream pipeline, o_voxel, cumesh, flex_gemm import; nvdiffrast CUDA context opens.
- Step-5 failure guidance shown correctly when the manifest was still empty.
- `make3d.bat` (drag-and-drop path as argument) with a Korean path containing spaces, while the
  web UI was running in another window: ok, 97.6 s.

Not verified yet:
- `install.bat` double-click itself (tested via `install.ps1` over SSH; same script).
- Interactive `login` prompt on a Windows console (token was passed with `--token`).
- A real 16GB card. A real (not simulated) OOM; on Windows the driver spills into shared memory instead.
- `generate` interactive OOM prompt. The web UI's rotatable Model3D view in a real browser
  (Gradio 6 Model3D uses Babylon.js WebGPU; headless Chrome has no WebGPU and cannot screenshot it).

M2 measurements: docs/BENCHMARKS.md. Gallery: docs/examples (GLBs in release `examples-v1`,
regenerate with `scripts/make_examples.py`, previews with `trellis2_lowvram.preview`).

Gotchas found so far:
- uv must use its own managed Python (`python-preference = "only-managed"`): a conda Python ships
  an old libstdc++ and the extensions fail with `GLIBCXX_3.4.30 not found`.
- uv searches `--extra-index-url` before `--index-url`. PyPI's Windows torch is CPU-only, so the
  CUDA index must be the extra one (`wheels/build.py`), or extensions fail with "CUDA_HOME not set".
- Windows extension builds need `DISTUTILS_USE_SDK=1` with an activated MSVC environment.
- `huggingface_hub` 0.36 `login()` has no `skip_if_logged_in`.
- `o-voxel/` has no license file of its own; the wheel gets the TRELLIS.2 root LICENSE.
- Upstream `to_glb` bakes alpha but always writes `alphaMode=OPAQUE`; `pipeline.set_alpha_mode`
  switches to BLEND when >1% of used texels are see-through (glass, jellyfish).
- Transparent containers: only the shell is generated; contents seen through glass are lost.
- Windows NVIDIA drivers spill to shared system memory instead of OOM (slow, not an error).
- Gradio sends usage analytics by default; `config.prepare_environment` sets GRADIO_ANALYTICS_ENABLED=False.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is ignored on Windows ("not supported on this
  platform" warning). All Windows measurements in BENCHMARKS were taken without it.
- A sync that fails midway (old shim, locked .exe) can leave the CUDA extensions uninstalled;
  `doctor` reports it and `install` / `trellis2 install-extensions` fixes it.
- Shims (`trellis2.bat/.sh`) use `uv run --no-sync`: a sync while the web UI runs fails on Windows
  (`trellis2-lowvram.exe` is locked, os error 32). Only install/update change the environment.
- `pkill -f <pattern>` also kills the shell running it (its command line contains the pattern); kill by PID.
- `trimesh.load(glb)` with default `process=True` takes minutes on these meshes; use `process=False`.

## 4. Map

```
install.sh / install.bat -> scripts/install.ps1   one-click install (uv, Python 3.11, deps, wheels, HF, models)
update.sh / update.bat                           pull + rerun install
trellis2.sh / trellis2.bat                       user shim: runs `trellis2-lowvram <cmd>` with the installed uv env
run.sh / run.bat                                 opens the web UI (`trellis2 ui`)
make3d.sh / make3d.bat                           drag pictures onto it -> `generate <files> --yes --open`
pyproject.toml, uv.lock                          Python deps (torch 2.8.0+cu128, xformers, transformers 4.57.3)
third_party/TRELLIS.2/                           upstream @ 75fbf01 (submodule; zip fallback without git)
src/trellis2_lowvram/
  config.py        paths; prepare_environment() sets HF_HOME, PYTORCH_CUDA_ALLOC_CONF, sys.path BEFORE torch import
  env_check.py     hardware checks via NVML/psutil (no torch), software checks after install; pure judge functions
  attention.py     flash_attn -> xformers; dense may fall back to sdpa, sparse cannot (no upstream SDPA varlen path)
  hf_access.py     token stored under our HF_HOME; DINOv3 gated-access check
  models.py        which weight files each pipeline type needs; downloads only those
  extensions.py    installs prebuilt CUDA wheels listed in wheels/manifest.json (sha256 verified)
  cli.py           doctor | login | download | install-extensions | generate | benchmark | ui
  generate.py      make_one (one seed, OOM ladder, export-only retry), run_job, reports
  pipeline.py      Generator: selective loading, stage wrappers, export, set_alpha_mode, OBJ
  memory.py        StageMonitor, StageOOM, apply_vram_cap
  presets.py       fast | balanced | quality, lower() ladder
  preview.py       4-view nvdiffrast render (UI gallery, docs)
  ui/app.py        Gradio UI
  i18n/            ko.json, en.json
wheels/sources.toml, build.py, manifest.json     pinned extension sources, builder, release hashes
.github/workflows/tests.yml, build-wheels.yml     CPU tests + lint; wheel builds (manual dispatch)
docs/PLAN.md       design, confirmed upstream facts, decisions, milestone criteria (Korean)
docs/UPSTREAM.md   pins and how to move them
```

## 5. Commands

```bash
uv sync --frozen                                   # dev env (needs CUDA wheels for full doctor)
uv run trellis2-lowvram doctor [--pre] [--json f]  # environment check
uv run trellis2-lowvram login                      # HF token + DINOv3 access
uv run trellis2-lowvram download [--pipeline 512 1024_cascade]
PYTHONPATH=src uv run --no-project --python 3.11 --with pytest python -m pytest -q
uvx ruff@0.14.0 check src tests wheels scripts && uvx ruff@0.14.0 format --check src tests wheels scripts
python wheels/build.py --out dist/wheels           # needs CUDA 12.8 nvcc + compiler; makes its own venv
```

Actions: `gh workflow run build-wheels.yml -f release_tag=wheels-vN` builds Windows and Linux
wheels into a **new** release (it refuses to overwrite one: builds are not byte-identical and
`wheels/manifest.json` pins sha256). Then download the release, check hashes against `*.jsonl`,
and write `wheels/manifest.json`. Current: `wheels-v1`. Windows needs the patches in `wheels/patches/`.

## 6. Facts about upstream that shape the design

Details and sources: `docs/PLAN.md` section 2.
- Upstream already has `low_vram=True` (default): each stage's model is moved to GPU and back.
  Our work is measurement, selective loading, activation peaks (decoders, `to_glb`), OOM handling.
- `ATTN_BACKEND` / `SPARSE_ATTN_BACKEND` are read at import time: set them before importing `trellis2`.
- `pipeline.json` default is `1024_cascade`; we default to `512` (1024 is experimental, 1536 blocked).
- DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`) is gated with **manual** approval. Each
  user requests access with their own account; weights are never redistributed.
- Upstream background removal is `briaai/RMBG-2.0` (gated, non-commercial). We use
  `ZhengPeng7/BiRefNet` (MIT) with the same upstream class.
- GLB texture baking needs nvdiffrast (non-commercial license). nvdiffrec is not used.
- FlexGEMM kernels are Triton; Windows uses `triton-windows`.

Verified OOM handling: `generate --pipeline 1024_cascade --vram-cap 7 -y` on the 4090 hit OOM in
export_glb, retried with texture 1024, hit OOM in texture_decoder, retried at 512, succeeded.

## 7. Next steps

1. Finish the extension wheel builds (local Linux, then Actions for Windows/Linux); publish to a
   release; fill `wheels/manifest.json`; full `doctor` must pass on Linux.
2. Hand `install.bat` to a Windows tester; fix from their `logs/install-*.log`.
3. M2 leftovers: after an OOM in export, retry only the export with the mesh already made
   (today the whole generation reruns). Measure a real 16GB card when one is available.
4. M3: confirm the web UI (`run.bat`) in a real browser on Windows; offloading-strength control
   from the original spec is not implemented (upstream low_vram is the only mode so far).
