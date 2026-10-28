"""Build the CUDA extension wheels from wheels/sources.toml.

Needs: a CUDA 12.8 toolkit (nvcc on PATH or CUDA_HOME), a C++ compiler (MSVC on Windows),
git, and uv. Creates its own build venv with the pinned torch, so it never touches the app env.

    python wheels/build.py --out dist/wheels [--only cumesh]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCHES = HERE / "patches"


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def checkout(ext: dict, work: Path) -> Path:
    dest = work / ext["name"]
    if not dest.exists():
        run(["git", "init", "-q", str(dest)])
        run(["git", "-C", str(dest), "remote", "add", "origin", ext["repo"]])
    run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", ext["commit"]])
    run(["git", "-C", str(dest), "checkout", "-q", "--force", "FETCH_HEAD"])
    # Submodules come from several hosts (GitLab for eigen) that sometimes refuse under load.
    sub = ["git", "-C", str(dest), "submodule", "update", "-q", "--init", "--recursive"]
    for attempt in range(4):
        try:
            run(sub + (["--depth", "1"] if attempt < 2 else []))
            break
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            time.sleep(30 * (attempt + 1))
    for patch in sorted((PATCHES / ext["name"]).glob("*.patch")):
        run(["git", "-C", str(dest), "apply", "--whitespace=nowarn", str(patch)])
    src = dest / ext.get("subdir", "")
    if lic := ext.get("license_from"):
        # The package folder has no license file of its own; ship the repository's.
        shutil.copy(dest / lic, src / "LICENSE")
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("dist/wheels"))
    ap.add_argument("--work", type=Path, default=Path("build/ext-src"))
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    cfg = tomllib.loads((HERE / "sources.toml").read_text(encoding="utf-8"))
    out, work = args.out.resolve(), args.work.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    venv = work / ".venv"
    if not venv.exists():
        run(["uv", "venv", "-q", "--python", "3.11", str(venv)])
    py = str(venv_python(venv))
    run(
        [
            "uv",
            "pip",
            "install",
            "-q",
            "--python",
            py,
            # uv searches --extra-index-url first. Put the CUDA index there: PyPI's Windows
            # torch is CPU-only, which makes every CUDA extension fail with "CUDA_HOME not set".
            "--extra-index-url",
            cfg["cuda_index"],
            f"torch=={cfg['torch']}",
            "setuptools>=64",
            "wheel",
            "ninja",
            "numpy",
            "pip",
        ]
    )

    has_cuda = subprocess.run(
        [py, "-c", "import torch,sys; sys.exit(0 if torch.version.cuda else 1)"]
    ).returncode
    if has_cuda != 0:
        raise SystemExit("build venv has a CPU-only torch; check the index order")

    env = dict(os.environ)
    env["TORCH_CUDA_ARCH_LIST"] = cfg["cuda_arch_list"]
    env["BUILD_TARGET"] = "cuda"
    if os.name == "nt":
        # MSVC is already activated (vcvars / msvc-dev-cmd); torch refuses to build otherwise.
        env["DISTUTILS_USE_SDK"] = "1"
    env.setdefault("MAX_JOBS", str(max(1, (os.cpu_count() or 2) // 2)))
    env["PATH"] = str(Path(py).parent) + os.pathsep + env["PATH"]  # ninja from the venv

    built: list[Path] = []
    failed: list[str] = []
    for ext in cfg["ext"]:
        if args.only and ext["name"] not in args.only:
            continue
        src = checkout(ext, work)
        before = set(out.glob("*.whl"))
        try:
            run(
                [
                    py,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-build-isolation",
                    "--no-deps",
                    "-w",
                    str(out),
                    str(src),
                ],
                env=env,
            )
        except subprocess.CalledProcessError:
            # Keep going so one CI run shows every extension that fails on this platform.
            failed.append(ext["name"])
            continue
        built += sorted(set(out.glob("*.whl")) - before)

    for whl in built:
        # Redistribution condition (nvdiffrast 3.1): the wheel must carry the license text.
        with zipfile.ZipFile(whl) as z:
            if not any("LICENSE" in n.upper() for n in z.namelist()):
                raise SystemExit(f"{whl.name}: no license file inside the wheel")
        digest = hashlib.sha256(whl.read_bytes()).hexdigest()
        print(json.dumps({"file": whl.name, "sha256": digest}))
    if failed:
        print("FAILED: " + ", ".join(failed), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
