import pytest

from trellis2_lowvram import cli, i18n


@pytest.fixture(autouse=True)
def english():
    i18n.set_lang("en")
    yield
    i18n.set_lang(None)


def test_generate_options():
    a = cli.build_parser().parse_args(["generate", "a.jpg", "b.heic", "--seeds", "3", "-y"])
    assert (a.input, a.seeds, a.yes, a.preset, a.output) == (
        ["a.jpg", "b.heic"],
        3,
        True,
        "balanced",
        None,
    )


def test_unreadable_pictures_are_reported_and_skipped(tmp_path, capsys):
    Image = pytest.importorskip("PIL.Image")
    good = tmp_path / "good.png"
    Image.new("RGB", (4, 4)).save(good)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"\x00" * 10)
    gone = tmp_path / "gone.png"
    assert cli._readable([good, broken, gone]) == [good]
    out = capsys.readouterr().out
    assert "broken.jpg" in out and "gone.png" in out
