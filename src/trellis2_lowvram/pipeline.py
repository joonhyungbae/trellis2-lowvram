"""Wrapper around upstream `Trellis2ImageTo3DPipeline`: selective loading, measured stages, export.

Import this module only after `config.prepare_environment()` and `attention.apply()`, because
upstream reads its backend settings when `trellis2` is first imported.

Differences from upstream `from_pretrained` + `run` (docs/PLAN.md 3):
- only the models the chosen pipeline type runs are loaded (others are loaded on demand);
- background removal uses BiRefNet (MIT) instead of RMBG-2.0 (gated, non-commercial);
- every stage method is wrapped by StageMonitor (time, peak VRAM, OOM -> StageOOM);
- optionally, models that fit stay on the GPU instead of moving every stage (keep_on_gpu).
The model itself, its weights, samplers and parameters are upstream's.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from . import models as weights
from .memory import StageMonitor, StageOOM
from .presets import Settings

log = logging.getLogger(__name__)

# Upstream stage methods, in run() order, with the names used in reports.
STAGES = (
    ("preprocess_image", "background_removal"),
    ("get_cond", "image_cond"),
    ("sample_sparse_structure", "sparse_structure"),
    ("sample_shape_slat", "shape_flow"),
    ("sample_shape_slat_cascade", "shape_flow"),
    ("sample_tex_slat", "texture_flow"),
    ("decode_shape_slat", "shape_decoder"),
    ("decode_tex_slat", "texture_decoder"),
)

NVDIFFRAST_FACE_LIMIT = 16_777_216

# GPU memory left free for the stages themselves when models stay on the GPU. Measured beside
# the weights at 512 on an RTX 3080: under 1.8GB up to texture 2048 / 500k faces (export_glb).
# Larger settings are not measured; an OOM there drops back to moving models (presets.lower).
GPU_WORK_RESERVE = 3 * 1024**3

_INIT_FUNCS = (
    "uniform_",
    "normal_",
    "trunc_normal_",
    "constant_",
    "ones_",
    "zeros_",
    "eye_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "orthogonal_",
)


@contextlib.contextmanager
def skip_random_init() -> Iterator[None]:
    """Make torch.nn.init a no-op while models are built.

    Upstream builds each model with training initialization (random values for ~1.3B weights on
    the CPU) and then overwrites every weight from the checkpoint: about 15 s per flow model on
    the RTX 3080 PC, 55 of 94 s of loading. All parameters are in the checkpoints (checked:
    no missing parameter keys); the one missing buffer, rope_phases, is computed, not random.
    """
    saved = {n: getattr(torch.nn.init, n) for n in _INIT_FUNCS if hasattr(torch.nn.init, n)}
    for n in saved:
        setattr(torch.nn.init, n, lambda tensor, *a, **k: tensor)
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(torch.nn.init, n, f)


@contextlib.contextmanager
def hub_offline() -> Iterator[None]:
    """Use the local Hugging Face cache without asking the Hub about every file (~1 s each).

    Raises if something is not cached; the caller then loads again online.
    """
    from huggingface_hub import constants

    saved = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        constants.HF_HUB_OFFLINE = (
            saved  # upstream example.py / app.py simplify to this before baking
        )


class Generator:
    def __init__(
        self,
        pipeline_type: str = "512",
        monitor: StageMonitor | None = None,
        on_load: Callable[[int, int, str], None] | None = None,
    ):
        """on_load(done, total, part) reports loading progress; part is a key for i18n "load.*"."""
        self.monitor = monitor or StageMonitor()
        self.on_load = on_load or (lambda done, total, part: None)
        try:
            with hub_offline():
                self._build(pipeline_type)
        except Exception as e:  # something not cached yet: same steps, asking the Hub
            log.info("offline load failed (%r); loading online", e)
            self._build(pipeline_type)
        self.pipe.cuda()  # low_vram: records the device only; models stay on the CPU

        self.kept: dict[str, Any] = {}  # models staying on the GPU (keep_on_gpu)
        self.monitor.on_oom = self.offload_all
        for method, stage in STAGES:
            setattr(self.pipe, method, self.monitor.wrap(stage, getattr(self.pipe, method)))

    def _build(self, pipeline_type: str) -> None:
        from trellis2.modules import image_feature_extractor
        from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg, samplers

        self.config = weights.load_pipeline_config()
        cfg = self.config
        missing = [name for name, _ in STAGES if not hasattr(Trellis2ImageTo3DPipeline, name)]
        if missing:  # the upstream pin moved without updating this wrapper (docs/UPSTREAM.md)
            raise RuntimeError(f"upstream pipeline has no {missing}")
        total = 2 + len(weights.weights_for(pipeline_type, cfg))

        def sampler(key: str):
            return getattr(samplers, cfg[key]["name"])(**cfg[key]["args"])

        self.on_load(0, total, "image_cond")
        image_cond = getattr(image_feature_extractor, cfg["image_cond_model"]["name"])(
            **cfg["image_cond_model"]["args"]
        )
        self.on_load(1, total, "rembg")
        rembg_model = rembg.BiRefNet(model_name=weights.REMBG_REPO)
        self.pipe = Trellis2ImageTo3DPipeline(
            models={},
            sparse_structure_sampler=sampler("sparse_structure_sampler"),
            shape_slat_sampler=sampler("shape_slat_sampler"),
            tex_slat_sampler=sampler("tex_slat_sampler"),
            sparse_structure_sampler_params=cfg["sparse_structure_sampler"]["params"],
            shape_slat_sampler_params=cfg["shape_slat_sampler"]["params"],
            tex_slat_sampler_params=cfg["tex_slat_sampler"]["params"],
            shape_slat_normalization=cfg["shape_slat_normalization"],
            tex_slat_normalization=cfg["tex_slat_normalization"],
            image_cond_model=image_cond,
            rembg_model=rembg_model,
            low_vram=True,
            default_pipeline_type=pipeline_type,
        )
        self.ensure(pipeline_type, offset=2, total=total)
        self.on_load(total, total, "done")

    def ensure(self, pipeline_type: str, offset: int = 0, total: int | None = None) -> None:
        """Load (once) every model this pipeline type needs."""
        from trellis2 import models as upstream_models

        refs = weights.weights_for(pipeline_type, self.config)
        for i, ref in enumerate(refs):
            if ref.key not in self.pipe.models:
                self.on_load(offset + i, total or len(refs), ref.key)
                log.info("load %s from %s/%s", ref.key, ref.repo_id, ref.name)
                with skip_random_init():
                    m = upstream_models.from_pretrained(f"{ref.repo_id}/{ref.name}")
                m.eval()
                self.pipe.models[ref.key] = m

    def _movable(self) -> list[tuple[str, Any]]:
        """Models upstream moves to the GPU and back, in the order worth keeping.

        Every move costs about 0.25 s per GB each way (RTX 3080: 2.44GB flow model 0.6 s
        up, 0.6 s back), so small models that move often go first, then flows in run order.
        """
        p = self.pipe
        order = [("image_cond", p.image_cond_model), ("rembg", p.rembg_model)]
        names = [k for k in p.models if "decoder" in k] + [k for k in p.models if "flow" in k]
        return [(n, m) for n, m in order if m is not None] + [(k, p.models[k]) for k in names]

    def keep_on_gpu(self, high: bool) -> list[str]:
        """high: keep the models that fit on the GPU (leaving GPU_WORK_RESERVE); else none.

        Upstream calls model.cpu() after each stage; for a kept model that call does nothing.
        Returns the names kept.
        """
        if not high:
            self.release_gpu()
            return []
        free = torch.cuda.mem_get_info()[0] - GPU_WORK_RESERVE
        for name, m in self._movable():
            if name in self.kept:
                continue
            size = _nbytes(m)
            if size > free:
                continue
            try:
                m.to(torch.device("cuda"))
            except torch.cuda.OutOfMemoryError:  # something else took the GPU meanwhile
                m.cpu()
                break
            m.cpu = _stay  # instance attribute: hides the class's cpu() for this model only
            self.kept[name] = m
            free -= size
        log.info("kept on GPU: %s", ", ".join(self.kept) or "-")
        return list(self.kept)

    def release_gpu(self) -> None:
        for m in self.kept.values():
            del m.cpu
            m.cpu()
        if self.kept:
            log.info("released from GPU: %s", ", ".join(self.kept))
            self.kept = {}
            torch.cuda.empty_cache()

    def offload_all(self) -> None:
        self.release_gpu()
        for m in self.pipe.models.values():
            m.cpu()
        self.pipe.image_cond_model.cpu()
        if self.pipe.rembg_model is not None:
            self.pipe.rembg_model.cpu()

    def generate(
        self, image: Image.Image, seed: int, s: Settings, remove_background: bool = True
    ) -> Any:
        """remove_background=False keeps the picture as is (no cut-out, no centering crop)."""
        self.ensure(s.pipeline_type)
        self.keep_on_gpu(s.gpu_memory == "high")
        steps = {"steps": s.steps}
        if not remove_background:
            image = image.convert("RGB")
        try:
            meshes = self.pipe.run(
                image,
                seed=seed,
                preprocess_image=remove_background,
                sparse_structure_sampler_params=steps,
                shape_slat_sampler_params=steps,
                tex_slat_sampler_params={**steps, "guidance_strength": s.tex_guidance},
                pipeline_type=s.pipeline_type,
                max_num_tokens=s.max_num_tokens,
            )
        except torch.cuda.OutOfMemoryError as e:
            # Steps we do not wrap: decode_latent's hole filling, the final empty_cache.
            self.offload_all()
            raise StageOOM("decode_latent", str(e).splitlines()[0]) from e
        return meshes[0]

    def export(self, mesh: Any, out_base: Path, s: Settings, obj: bool = True) -> list[Path]:
        import o_voxel

        mon = self.monitor
        mon.wrap("export_simplify", mesh.simplify)(NVDIFFRAST_FACE_LIMIT)
        glb = mon.wrap("export_glb", o_voxel.postprocess.to_glb)(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=s.decimation_target,
            texture_size=s.texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
        )
        set_alpha_mode(glb)
        out_base.parent.mkdir(parents=True, exist_ok=True)
        paths = [out_base.with_suffix(".glb")]
        glb.export(paths[0], extension_webp=False)
        if obj:
            paths += export_obj(paths[0])
        return paths


def _stay(*args: Any, **kwargs: Any) -> None:
    """cpu() of a model kept on the GPU."""


def _nbytes(m: Any) -> int:
    module = m if isinstance(m, torch.nn.Module) else m.model  # upstream DINOv3 / BiRefNet wrap
    tensors = list(module.parameters()) + list(module.buffers())
    return sum(t.numel() * t.element_size() for t in tensors)


# Share of used texels that must be see-through before the material is marked transparent.
TRANSPARENT_SHARE = 0.01


def set_alpha_mode(mesh: Any) -> str:
    """Mark the material BLEND when the baked texture really has see-through parts.

    Upstream to_glb bakes an alpha channel (glass, jellyfish...) but always writes
    alphaMode=OPAQUE, so every viewer shows glass as a solid shell. We look at the alpha of
    the texels the mesh actually uses (unused texels are inpainted) and switch only then.
    """
    import numpy as np

    mat = mesh.visual.material
    tex = np.asarray(mat.baseColorTexture)
    if tex.ndim != 3 or tex.shape[-1] != 4:
        return mat.alphaMode
    h, w = tex.shape[:2]
    uv = np.asarray(mesh.visual.uv)
    rows = ((1 - uv[:, 1]) * (h - 1)).astype(int).clip(0, h - 1)
    cols = (uv[:, 0] * (w - 1)).astype(int).clip(0, w - 1)
    share = float((tex[rows, cols, 3] < 250).mean())
    if share > TRANSPARENT_SHARE:
        mat.alphaMode = "BLEND"
    log.info("alpha: %.1f%% of used texels see-through -> %s", share * 100, mat.alphaMode)
    return mat.alphaMode


def export_obj(glb_path: Path) -> list[Path]:
    """OBJ + MTL + texture next to the GLB, for tools that do not read GLB.

    OBJ has no PBR: only the base color texture carries over (metallic/roughness are lost).
    """
    import trimesh

    # process=False: merging vertices on a 500k-face mesh takes minutes and changes nothing here
    scene = trimesh.load(glb_path, force="scene", process=False)
    mesh = next(iter(scene.geometry.values()))
    folder = glb_path.with_suffix("")
    folder = folder.parent / (folder.name + "_obj")
    folder.mkdir(exist_ok=True)
    obj_path = folder / (glb_path.stem + ".obj")
    mesh.export(obj_path)
    return sorted(folder.iterdir())
