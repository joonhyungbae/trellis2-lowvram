"""Hand downloaded models to another PC (a teacher to students) and take them in there.

A shared folder has the Hugging Face cache layout, so a PC that takes it in can load the models
without a token or DINOv3 approval of its own (loading is offline first, pipeline.hub_offline):

    <folder>/huggingface/hub/models--<org>--<name>/refs/main
    <folder>/huggingface/hub/models--<org>--<name>/snapshots/<commit>/<files>
    <folder>/DINOv3-LICENSE.md   the DINOv3 License requires a copy with every copy of the weights
    <folder>/README.txt
    <folder>/files.json          every file under huggingface/hub with its size

Folders do not always arrive in one piece. Google Drive, for one, downloads a folder as numbered
zip parts and hands out files over 2GB on their own, outside their folders and renamed
(`ss_flow_img_dit_1_3B_64_bf16-008.safetensors`); people also extract each part somewhere else.
import_ therefore gathers every hub it finds and puts such stray files back by name and size.

Only refs and snapshots are copied, with symlinks resolved, so the folder works on any drive.
No Hugging Face token is ever copied. Nothing here imports huggingface_hub (unit tests run without).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

log = logging.getLogger(__name__)

SHARE_DIR = "shared-models"  # default folder name, next to the install (in .gitignore)
LICENSE_NAME = "DINOv3-LICENSE.md"
MANIFEST_NAME = "files.json"
# Added to file names on the way: Google Drive parts "-008", browsers and Windows " (1)".
_ADDED_SUFFIX = re.compile(r"(-\d{3}| \(\d+\))$")


class NotAShare(ValueError):
    """The folder has no models in Hugging Face cache layout."""


def repo_dir(hub: Path, repo: str) -> Path:
    return hub / ("models--" + repo.replace("/", "--"))


def commit_of(hub: Path, repo: str) -> str:
    return (repo_dir(hub, repo) / "refs" / "main").read_text(encoding="utf-8").strip()


def snapshot(hub: Path, repo: str) -> Path:
    return repo_dir(hub, repo) / "snapshots" / commit_of(hub, repo)


def find_hubs(folder: Path) -> list[Path]:
    """Every Hugging Face hub folder inside folder (one per extracted zip part, or just one)."""
    hubs = {
        ref.parents[2]
        for ref in folder.rglob("refs/main")
        if ref.parents[1].name.startswith("models--")
    }
    if not hubs:
        raise NotAShare(str(folder))
    return sorted(hubs)


def original_name(name: str) -> str:
    """'x_bf16-008.safetensors' or 'x_bf16 (1).safetensors' -> 'x_bf16.safetensors'."""
    stem, dot, ext = name.rpartition(".")
    if not dot:
        return _ADDED_SUFFIX.sub("", name)
    return _ADDED_SUFFIX.sub("", stem) + dot + ext


def _copy(src: Path, dst: Path, move: bool = False) -> int:
    """Copy (or move) one file; skip it when dst already has the same size. Returns bytes."""
    size = src.stat().st_size  # follows the symlinks of a Hugging Face cache
    if dst.is_file() and dst.stat().st_size == size:
        if move:
            src.unlink()
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Via a temporary name: a copy cut off halfway must not look like a finished file.
    part = dst.with_name(dst.name + ".part")
    if move:
        shutil.move(str(src), str(part))
    else:
        shutil.copyfile(src, part)
    os.replace(part, dst)
    return size


def export(
    hub: Path,
    dest: Path,
    files: Iterable[tuple[str, str]],
    whole_repos: Iterable[str],
    license_repo: str,
    readme: str,
    on_file: Callable[[str], None] | None = None,
) -> int:
    """Copy the given (repo, path) files and every file of whole_repos from hub into dest.

    Returns the bytes copied (files already in dest are skipped, so a rerun is quick).
    """
    jobs: list[tuple[str, str]] = list(files)
    for repo in whole_repos:
        snap = snapshot(hub, repo)
        jobs += [
            (repo, p.relative_to(snap).as_posix()) for p in sorted(snap.rglob("*")) if p.is_file()
        ]
    out_hub = dest / "huggingface" / "hub"
    manifest: dict[str, int] = {}
    copied = 0
    for repo in dict.fromkeys(r for r, _ in jobs):
        ref = repo_dir(out_hub, repo) / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(commit_of(hub, repo), encoding="utf-8")
    for repo, name in jobs:
        if on_file:
            on_file(name)
        rel = Path(repo_dir(hub, repo).name, "snapshots", commit_of(hub, repo), name)
        copied += _copy(hub / rel, out_hub / rel)
        manifest[rel.as_posix()] = (out_hub / rel).stat().st_size
    lic = snapshot(hub, license_repo) / "LICENSE.md"
    if lic.is_file():
        shutil.copyfile(lic, dest / LICENSE_NAME)
    (dest / "README.txt").write_text(readme, encoding="utf-8")
    (dest / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return copied


def _expected(folder: Path, hubs: list[Path]) -> dict[str, int | None]:
    """hub-relative path -> size of every file the folder should hold.

    From files.json when there is one. Older folders have none: then each weight `<x>.json` in a
    snapshot's ckpts/ implies `<x>.safetensors` beside it (that is how TRELLIS.2 checkpoints come).
    """
    manifests = sorted(folder.rglob(MANIFEST_NAME))
    if manifests:
        return json.loads(manifests[0].read_text(encoding="utf-8"))
    want: dict[str, int | None] = {}
    for hub in hubs:
        for p in hub.glob("models--*/snapshots/*/**/*"):
            if not p.is_file():
                continue
            want[p.relative_to(hub).as_posix()] = p.stat().st_size
            weights = p.with_suffix(".safetensors")
            in_ckpts = "ckpts" in p.relative_to(hub).parts
            if p.suffix == ".json" and in_ckpts and not weights.is_file():
                want.setdefault(weights.relative_to(hub).as_posix(), None)
    return want


def import_(
    folder: Path, hub: Path, move: bool = False, on_file: Callable[[str], None] | None = None
) -> tuple[int, list[str]]:
    """Take a shared folder into this install's cache.

    Returns (files added, names still missing). A file counts as missing when it is in neither
    a hub nor among the stray files, or when its size differs from files.json (cut off).
    The shared refs/main wins: the shared snapshot is complete for what it was made for.
    """
    hubs = [h for h in find_hubs(folder) if h.resolve() != hub.resolve()]
    if not hubs:
        return 0, []
    expected = _expected(folder, hubs)
    found: dict[str, Path] = {}  # hub-relative path -> where it is in the shared folder
    for src_hub in hubs:
        for repo in sorted(src_hub.glob("models--*")):
            ref = repo / "refs" / "main"
            if ref.is_file():
                out = hub / ref.relative_to(src_hub)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(ref.read_text(encoding="utf-8").strip(), encoding="utf-8")
            for p in repo.glob("snapshots/*/**/*"):
                if p.is_file() and not p.name.endswith(".part"):
                    found.setdefault(p.relative_to(src_hub).as_posix(), p)
    inside = {p.resolve() for p in found.values()}
    strays: dict[str, list[Path]] = {}
    for p in folder.rglob("*"):
        if p.is_file() and p.resolve() not in inside:
            strays.setdefault(original_name(p.name), []).append(p)
    added, missing = 0, []
    for rel in sorted(set(expected) | set(found)):
        size = expected.get(rel)
        src = found.get(rel)
        if src is None or (size is not None and src.stat().st_size != size):
            name = rel.rsplit("/", 1)[-1]
            fits = [p for p in strays.get(name, []) if size is None or p.stat().st_size == size]
            src = fits[0] if fits else None
            if src is not None:
                log.info("stray file %s -> %s", src, rel)
        dst = hub / rel
        if src is None:
            if not (dst.is_file() and (size is None or dst.stat().st_size == size)):
                missing.append(rel.rsplit("/", 1)[-1])
            continue
        if on_file:
            on_file(dst.name)
        existed = dst.is_file()
        _copy(src, dst, move)
        added += not existed
    return added, missing
