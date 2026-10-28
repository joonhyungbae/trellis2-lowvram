"""make_one's OOM ladder with a fake pipeline (needs torch to import; skipped in CPU-only CI)."""

import pytest

pytest.importorskip("torch")

from trellis2_lowvram.generate import make_one  # noqa: E402
from trellis2_lowvram.memory import StageOOM  # noqa: E402
from trellis2_lowvram.presets import PRESETS  # noqa: E402


class FakeImage:
    def copy(self):
        return self


class FakeMonitor:
    records: list = []

    def to_dicts(self):
        return []


class FakeGen:
    """Fails the stages listed in `fail` once each, in order."""

    def __init__(self, fail):
        self.monitor = FakeMonitor()
        self.fail = list(fail)
        self.generated = 0
        self.exported = []

    def generate(self, image, seed, s, remove_background):
        self.generated += 1
        if self.fail and not self.fail[0].startswith("export"):
            raise StageOOM(self.fail.pop(0), "test")
        return "mesh"

    def export(self, mesh, out_base, s, obj=True):
        if self.fail and self.fail[0].startswith("export"):
            raise StageOOM(self.fail.pop(0), "test")
        self.exported.append(s.texture_size)
        return [f"{out_base}.glb"]


def test_export_oom_retries_only_the_export(tmp_path):
    gen = FakeGen(["export_glb"])
    attempts = make_one(gen, FakeImage(), 42, PRESETS["quality"], tmp_path / "a", lambda *a: True)
    assert [a.ok for a in attempts] == [False, True]
    assert gen.generated == 1  # the mesh was kept
    assert gen.exported == [2048]


def test_generation_oom_makes_the_mesh_again(tmp_path):
    gen = FakeGen(["shape_flow"])
    attempts = make_one(gen, FakeImage(), 42, PRESETS["balanced"], tmp_path / "a", lambda *a: True)
    assert attempts[-1].ok and gen.generated == 2


def test_declined_retry_stops(tmp_path):
    gen = FakeGen(["export_glb"])
    attempts = make_one(gen, FakeImage(), 42, PRESETS["quality"], tmp_path / "a", lambda *a: False)
    assert [a.ok for a in attempts] == [False]
    assert attempts[0].oom_stage == "export_glb"
