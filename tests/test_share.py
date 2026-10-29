import json
from pathlib import Path

import pytest

from trellis2_lowvram import share


def make_hub(root: Path) -> Path:
    """A tiny Hugging Face cache: one repo with two files, one repo copied whole."""
    hub = root / "huggingface" / "hub"
    for repo, commit, files in (
        ("org/big", "c1", {"pipeline.json": "{}", "ckpts/a.json": "a", "ckpts/b.json": "bb"}),
        ("facebook/dino", "c2", {"config.json": "{}", "LICENSE.md": "license", "m.bin": "xyz"}),
    ):
        d = share.repo_dir(hub, repo)
        (d / "refs").mkdir(parents=True)
        (d / "refs" / "main").write_text(commit)
        for name, text in files.items():
            p = d / "snapshots" / commit / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    (hub.parent / "token").write_text("hf_secret")
    return hub


def test_export_copies_only_what_is_asked_and_never_the_token(tmp_path):
    hub = make_hub(tmp_path / "src")
    dest = tmp_path / "shared"
    files = [("org/big", "pipeline.json"), ("org/big", "ckpts/a.json")]
    n = share.export(hub, dest, files, ["facebook/dino"], "facebook/dino", "readme")
    out = dest / "huggingface" / "hub"
    assert (share.snapshot(out, "org/big") / "ckpts" / "a.json").read_text() == "a"
    assert not (share.snapshot(out, "org/big") / "ckpts" / "b.json").exists()
    assert (share.snapshot(out, "facebook/dino") / "m.bin").read_text() == "xyz"
    assert (dest / share.LICENSE_NAME).read_text() == "license"
    assert (dest / "README.txt").read_text() == "readme"
    manifest = json.loads((dest / share.MANIFEST_NAME).read_text())
    assert manifest["models--org--big/snapshots/c1/ckpts/a.json"] == 1
    assert not any(p.name == "token" for p in dest.rglob("*"))
    assert n == len("{}") + len("a") + len("{}") + len("license") + len("xyz")
    # A rerun skips files that are already there.
    assert share.export(hub, dest, files, ["facebook/dino"], "facebook/dino", "readme") == 0


@pytest.mark.parametrize("move", [False, True])
def test_import_into_an_empty_install(tmp_path, move):
    hub = make_hub(tmp_path / "src")
    dest = tmp_path / "shared"
    share.export(hub, dest, [("org/big", "pipeline.json")], ["facebook/dino"], "facebook/dino", "")
    mine = tmp_path / "student" / "huggingface" / "hub"
    seen = []
    assert share.import_(dest, mine, move, seen.append) == (4, [])
    assert share.commit_of(mine, "org/big") == "c1"
    assert (share.snapshot(mine, "facebook/dino") / "config.json").read_text() == "{}"
    assert "m.bin" in seen
    left = [p for p in (dest / "huggingface").rglob("*") if p.is_file() and "snapshots" in p.parts]
    assert bool(left) is not move
    # Taking the same folder in again adds nothing.
    assert share.import_(dest, mine) == (0, [])


def test_find_hubs_accepts_inner_folders_and_rejects_others(tmp_path):
    hub = make_hub(tmp_path / "s")
    assert share.find_hubs(tmp_path / "s") == [hub]
    assert share.find_hubs(hub) == [hub]
    (tmp_path / "nothing").mkdir()
    with pytest.raises(share.NotAShare):
        share.find_hubs(tmp_path / "nothing")


def test_import_of_own_cache_is_a_no_op(tmp_path):
    hub = make_hub(tmp_path / "s")
    assert share.import_(tmp_path / "s", hub) == (0, [])


@pytest.mark.parametrize(
    "name, original",
    [
        (
            "ss_flow_img_dit_1_3B_64_bf16-008.safetensors",
            "ss_flow_img_dit_1_3B_64_bf16.safetensors",
        ),
        ("model (1).safetensors", "model.safetensors"),
        ("model.safetensors", "model.safetensors"),
        ("dinov3-vitl16.json", "dinov3-vitl16.json"),
    ],
)
def test_original_name(name, original):
    assert share.original_name(name) == original


def split_like_google_drive(dest: Path, with_manifest: bool) -> Path:
    """Two zip parts extracted into their own folders; ckpts/b.json's weights came on their own."""
    hub = make_hub(dest / "teacher")
    b = share.snapshot(hub, "org/big") / "ckpts" / "b.safetensors"
    b.write_text("weights")
    files = [
        ("org/big", "pipeline.json"),
        ("org/big", "ckpts/b.json"),
        ("org/big", "ckpts/b.safetensors"),
    ]
    share.export(hub, dest / "full", files, ["facebook/dino"], "facebook/dino", "")
    got = dest / "shared-models"
    part1 = got / "part-001" / "huggingface" / "hub"
    part2 = got / "part-002" / "huggingface" / "hub"
    full = dest / "full" / "huggingface" / "hub"
    for p in full.rglob("*"):
        if p.is_file():
            rel = p.relative_to(full)
            if p.name == "b.safetensors":
                (got / "b-003.safetensors").write_bytes(p.read_bytes())
                continue
            target = (part1 if "org--big" in str(rel) else part2) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
    if with_manifest:
        (got / share.MANIFEST_NAME).write_bytes((dest / "full" / share.MANIFEST_NAME).read_bytes())
    return got


@pytest.mark.parametrize("with_manifest", [True, False])
def test_import_puts_stray_big_files_back(tmp_path, with_manifest):
    got = split_like_google_drive(tmp_path, with_manifest)
    mine = tmp_path / "student" / "hub"
    added, missing = share.import_(got, mine)
    assert missing == []
    assert (share.snapshot(mine, "org/big") / "ckpts" / "b.safetensors").read_text() == "weights"
    assert (share.snapshot(mine, "facebook/dino") / "m.bin").read_text() == "xyz"


@pytest.mark.parametrize("with_manifest", [True, False])
def test_import_names_what_is_missing(tmp_path, with_manifest):
    got = split_like_google_drive(tmp_path, with_manifest)
    (got / "b-003.safetensors").unlink()
    _, missing = share.import_(got, tmp_path / "student" / "hub")
    assert missing == ["b.safetensors"]


def test_import_rejects_a_cut_off_file(tmp_path):
    got = split_like_google_drive(tmp_path, with_manifest=True)
    (got / "b-003.safetensors").write_text("wei")
    _, missing = share.import_(got, tmp_path / "student" / "hub")
    assert missing == ["b.safetensors"]


def test_partial_copy_is_not_taken_for_a_finished_file(tmp_path, monkeypatch):
    src = tmp_path / "a.bin"
    src.write_bytes(b"12345")
    dst = tmp_path / "out" / "a.bin"

    def broken(a, b):
        Path(b).write_bytes(b"12")
        raise OSError("disk pulled out")

    monkeypatch.setattr(share.shutil, "copyfile", broken)
    with pytest.raises(OSError):
        share._copy(src, dst)
    assert not dst.exists()
