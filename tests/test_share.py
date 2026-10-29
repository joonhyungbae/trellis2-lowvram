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
    assert share.import_(dest, mine, move, seen.append) == 4
    assert share.commit_of(mine, "org/big") == "c1"
    assert (share.snapshot(mine, "facebook/dino") / "config.json").read_text() == "{}"
    assert "m.bin" in seen
    left = [p for p in (dest / "huggingface").rglob("*") if p.is_file() and "snapshots" in p.parts]
    assert bool(left) is not move
    # Taking the same folder in again adds nothing.
    assert share.import_(dest, mine) == 0


def test_find_hub_accepts_inner_folders_and_rejects_others(tmp_path):
    hub = make_hub(tmp_path / "s")
    assert share.find_hub(tmp_path / "s") == hub
    assert share.find_hub(hub.parent) == hub
    assert share.find_hub(hub) == hub
    with pytest.raises(share.NotAShare):
        share.find_hub(tmp_path / "nothing")


def test_import_of_own_cache_is_a_no_op(tmp_path):
    hub = make_hub(tmp_path / "s")
    assert share.import_(tmp_path / "s", hub) == 0


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
