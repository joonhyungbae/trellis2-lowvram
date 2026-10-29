"""Hand downloaded models to another PC (a teacher to students) and take them in there.

A shared folder has the Hugging Face cache layout, so a PC that takes it in can load the models
without a token or DINOv3 approval of its own (loading is offline first, pipeline.hub_offline):

    <folder>/huggingface/hub/models--<org>--<name>/refs/main
    <folder>/huggingface/hub/models--<org>--<name>/snapshots/<commit>/<files>
    <folder>/DINOv3-LICENSE.md   the DINOv3 License requires a copy with every copy of the weights
    <folder>/README.txt

Only refs and snapshots are copied, with symlinks resolved, so the folder works on any drive.
No Hugging Face token is ever copied. Nothing here imports huggingface_hub (unit tests run without).
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

log = logging.getLogger(__name__)

SHARE_DIR = "shared-models"  # default folder name, next to the install (in .gitignore)
LICENSE_NAME = "DINOv3-LICENSE.md"


class NotAShare(ValueError):
    """The folder has no models in Hugging Face cache layout."""


def repo_dir(hub: Path, repo: str) -> Path:
    return hub / ("models--" + repo.replace("/", "--"))


def commit_of(hub: Path, repo: str) -> str:
    return (repo_dir(hub, repo) / "refs" / "main").read_text(encoding="utf-8").strip()


def snapshot(hub: Path, repo: str) -> Path:
    return repo_dir(hub, repo) / "snapshots" / commit_of(hub, repo)


def find_hub(folder: Path) -> Path:
    """Accept the shared folder itself, its huggingface folder, or the hub folder."""
    for hub in (folder / "huggingface" / "hub", folder / "hub", folder):
        if hub.is_dir() and any(hub.glob("models--*/refs/main")):
            return hub
    raise NotAShare(str(folder))


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
    lic = snapshot(hub, license_repo) / "LICENSE.md"
    if lic.is_file():
        shutil.copyfile(lic, dest / LICENSE_NAME)
    (dest / "README.txt").write_text(readme, encoding="utf-8")
    return copied


def import_(
    folder: Path, hub: Path, move: bool = False, on_file: Callable[[str], None] | None = None
) -> int:
    """Take a shared folder into this install's cache. Returns the number of files added.

    The shared refs/main wins: the shared snapshot is complete for what it was made for.
    """
    src_hub = find_hub(folder)
    if src_hub.resolve() == hub.resolve():
        return 0
    added = 0
    for src_repo in sorted(src_hub.glob("models--*")):
        ref = src_repo / "refs" / "main"
        if not ref.is_file():
            continue
        commit = ref.read_text(encoding="utf-8").strip()
        snap = src_repo / "snapshots" / commit
        for p in sorted(snap.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(src_hub)
            if on_file:
                on_file(p.name)
            dst = hub / rel
            existed = dst.is_file()
            _copy(p, dst, move)
            added += not existed
        out_ref = hub / ref.relative_to(src_hub)
        out_ref.parent.mkdir(parents=True, exist_ok=True)
        out_ref.write_text(commit, encoding="utf-8")
    return added
