import errno

from trellis2_lowvram.errors import explain
from trellis2_lowvram.images import BadImage


def test_bad_image(tmp_path):
    x = explain(BadImage(tmp_path / "a.jpg"))
    assert (x.msg, x.params) == ("error.bad_image", {"path": "a.jpg"})


def test_disk_full_found_through_the_cause_chain():
    try:
        try:
            raise OSError(errno.ENOSPC, "No space left on device")
        except OSError as inner:
            raise RuntimeError("export failed") from inner
    except RuntimeError as outer:
        assert explain(outer).msg == "error.no_space"


def test_missing_component():
    x = explain(ModuleNotFoundError("No module named 'cumesh'", name="cumesh"))
    assert (x.msg, x.fix, x.params["name"]) == ("error.components", "fix.reinstall", "cumesh")


def test_no_gpu():
    x = explain(AssertionError("Torch not compiled with CUDA enabled"))
    assert x.msg == "error.no_gpu"


def test_unknown_stays_unexpected():
    assert explain(ValueError("something new")) is None
