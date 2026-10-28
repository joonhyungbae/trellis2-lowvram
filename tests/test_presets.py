from dataclasses import replace

from trellis2_lowvram.presets import PRESETS, Settings, lower, resolve


def test_export_oom_lowers_texture_then_faces_then_gives_up():
    s = PRESETS["quality"]
    seen = []
    while (s := lower(s, "export_glb")) is not None:
        seen.append((s.texture_size, s.decimation_target))
    assert seen == [(2048, 1_000_000), (1024, 1_000_000), (1024, 200_000)]


def test_generation_oom_drops_1024_first():
    s = replace(PRESETS["balanced"], pipeline_type="1024_cascade")
    s = lower(s, "shape_flow")
    assert (s.pipeline_type, s.texture_size) == ("512", 2048)
    assert lower(s, "shape_flow").texture_size == 1024
    assert lower(lower(s, "shape_flow"), "shape_flow") is None


def test_resolve_ignores_unset_overrides():
    assert resolve("balanced", pipeline_type=None, steps=None) == Settings()
    assert resolve("fast", steps=20).steps == 20
