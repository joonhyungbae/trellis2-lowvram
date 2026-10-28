"""Wrapper around upstream `Trellis2ImageTo3DPipeline`: selective loading, measured stages, export.

Import this module only after `config.prepare_environment()` and `attention.apply()`, because
upstream reads its backend settings when `trellis2` is first imported.

Differences from upstream `from_pretrained` + `run` (docs/PLAN.md 3):
- only the models the chosen pipeline type runs are loaded (others are loaded on demand);
- background removal uses BiRefNet (MIT) instead of RMBG-2.0 (gated, non-commercial);
- every stage method is wrapped by StageMonitor (time, peak VRAM, OOM -> StageOOM).
The model itself, its weights, samplers and parameters are upstream's.
"""

from __future__ import annotations

import logging
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

NVDIFFRAST_FACE_LIMIT = 16_777_216  # upstream example.py / app.py simplify to this before baking


class Generator:
    def __init__(self, pipeline_type: str = "512", monitor: StageMonitor | None = None):
        from trellis2.modules import image_feature_extractor
        from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg, samplers

        self.monitor = monitor or StageMonitor()
        self.config = weights.load_pipeline_config()
        cfg = self.config
        missing = [name for name, _ in STAGES if not hasattr(Trellis2ImageTo3DPipeline, name)]
        if missing:  # the upstream pin moved without updating this wrapper (docs/UPSTREAM.md)
            raise RuntimeError(f"upstream pipeline has no {missing}")

        def sampler(key: str):
            return getattr(samplers, cfg[key]["name"])(**cfg[key]["args"])

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
            image_cond_model=getattr(image_feature_extractor, cfg["image_cond_model"]["name"])(
                **cfg["image_cond_model"]["args"]
            ),
            rembg_model=rembg.BiRefNet(model_name=weights.REMBG_REPO),
            low_vram=True,
            default_pipeline_type=pipeline_type,
        )
        self.ensure(pipeline_type)
        self.pipe.cuda()  # low_vram: records the device only; models stay on the CPU

        self.monitor.on_oom = self.offload_all
        for method, stage in STAGES:
            setattr(self.pipe, method, self.monitor.wrap(stage, getattr(self.pipe, method)))

    def ensure(self, pipeline_type: str) -> None:
        """Load (once) every model this pipeline type needs."""
        from trellis2 import models as upstream_models

        for ref in weights.weights_for(pipeline_type, self.config):
            if ref.key not in self.pipe.models:
                log.info("load %s from %s/%s", ref.key, ref.repo_id, ref.name)
                m = upstream_models.from_pretrained(f"{ref.repo_id}/{ref.name}")
                m.eval()
                self.pipe.models[ref.key] = m

    def offload_all(self) -> None:
        for m in self.pipe.models.values():
            m.cpu()
        self.pipe.image_cond_model.cpu()
        if self.pipe.rembg_model is not None:
            self.pipe.rembg_model.cpu()

    def generate(self, image: Image.Image, seed: int, s: Settings) -> Any:
        self.ensure(s.pipeline_type)
        steps = {"steps": s.steps}
        try:
            meshes = self.pipe.run(
                image,
                seed=seed,
                sparse_structure_sampler_params=steps,
                shape_slat_sampler_params=steps,
                tex_slat_sampler_params=steps,
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
