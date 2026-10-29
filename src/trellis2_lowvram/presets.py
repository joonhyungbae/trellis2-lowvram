"""Quality presets and the "lower the settings" ladder used after an out-of-memory error.

Values start from upstream's own defaults (app.py sliders: 12 steps, texture 2048, decimation
500k; example.py: texture 4096, decimation 1M). They are provisional until measured
(docs/BENCHMARKS.md); change them only with a benchmark report.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Settings:
    pipeline_type: str = "512"
    steps: int = 12  # same for sparse structure, shape and texture samplers
    texture_size: int = 2048
    decimation_target: int = 500_000
    max_num_tokens: int = 49152  # upstream default; only matters for 1024_cascade
    tex_guidance: float = 1.0  # texture sampler guidance_strength (upstream pipeline.json: 1.0)
    # "low": every stage moves its model to the GPU and back (upstream low_vram).
    # "high": models that fit stay on the GPU between stages and runs (Generator.keep_on_gpu).
    gpu_memory: str = "low"


PRESETS: dict[str, Settings] = {
    "fast": Settings(steps=8, texture_size=1024, decimation_target=200_000),
    "balanced": Settings(),
    "quality": Settings(texture_size=4096, decimation_target=1_000_000),
}


GPU_MEMORY = ("low", "high")


def resolve(preset: str, **overrides: object) -> Settings:
    s = PRESETS[preset]
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(s, **clean) if clean else s


def lower(s: Settings, stage: str) -> Settings | None:
    """One step down after an OOM in `stage`. None when there is nothing left to lower.

    Export (texture baking) failures lower the texture first; generation failures lower the
    resolution first, since texture size does not affect the earlier stages.
    """
    if s.gpu_memory == "high":  # first give back what we kept on the GPU; same quality
        return replace(s, gpu_memory="low")
    if stage.startswith("export"):
        if s.texture_size > 1024:
            return replace(s, texture_size=s.texture_size // 2)
        if s.decimation_target > 200_000:
            return replace(s, decimation_target=200_000)
        return None
    if s.pipeline_type != "512":
        return replace(s, pipeline_type="512")
    if s.texture_size > 1024:  # also frees a little room for the later export
        return replace(s, texture_size=s.texture_size // 2)
    return None
