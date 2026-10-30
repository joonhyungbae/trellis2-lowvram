"""Render a GLB from four angles next to its input image (web UI, docs and reports).

    uv run --frozen python -m trellis2_lowvram.preview model.glb preview.jpg [input.png]

The web UI shows these images next to its interactive 3D view, because that view needs WebGPU,
which some browsers (and every headless test browser) do not have.

Base color texture (with its alpha, unless the material is OPAQUE) and simple directional
shading, rasterized with nvdiffrast (needs a GPU). A quick look, not a PBR render:
metallic/roughness are ignored.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
from PIL import Image

from .images import open_image

SIZE = 512
BG = 0.93
LAYERS = 4


def load(glb: str):
    scene = trimesh.load(glb, force="scene", process=False)  # process=True takes minutes
    mesh = next(iter(scene.geometry.values()))
    dev = "cuda"
    v = torch.tensor(mesh.vertices, dtype=torch.float32, device=dev)
    v = v - (v.max(0).values + v.min(0).values) / 2
    v = v / v.abs().max()
    mat = mesh.visual.material
    tex_img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    tex = torch.tensor(np.array(tex_img.convert("RGBA")), dtype=torch.float32, device=dev)[None]
    mode = getattr(mat, "alphaMode", None)
    if mode in (None, "OPAQUE"):  # viewers ignore alpha then, too
        tex[..., 3] = 255
    elif mode == "MASK":  # fully shown above the cutoff, hidden below
        cutoff = getattr(mat, "alphaCutoff", None) or 0.5
        tex[..., 3] = (tex[..., 3] >= cutoff * 255).float() * 255
    f = torch.tensor(mesh.faces, dtype=torch.int64, device=dev)
    return (
        v,
        f.int(),
        torch.tensor(mesh.visual.uv, dtype=torch.float32, device=dev),
        vertex_normals(v, f),
        (tex / 255).contiguous(),
    )


def vertex_normals(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals. trimesh's fall back to a slow Python loop without scipy."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = torch.linalg.cross(b - a, c - a)
    n = torch.zeros_like(v)
    for i in range(3):
        n.index_add_(0, f[:, i], fn)
    return torch.nn.functional.normalize(n, dim=-1)


def view(ctx, v, f, uv, n, tex, yaw: float, pitch: float) -> np.ndarray:
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], device=v.device)
    rx = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], device=v.device)
    m = rx @ ry
    p, nn = v @ m.T, n @ m.T
    dist, zoom = 3.0, 2.2
    w = dist - p[:, 2]
    near, far = 1.0, 5.0  # depth: NDC z linear in view distance
    z = w * ((w - near) / (far - near) * 2 - 1)
    clip = torch.stack([p[:, 0] * zoom, p[:, 1] * zoom, z, w], 1)[None]
    light = torch.nn.functional.normalize(torch.tensor([0.4, 0.6, 1.0], device=v.device), dim=0)
    # Depth peeling: composite up to LAYERS surfaces front to back, so glass shows what is inside.
    out = torch.zeros((1, SIZE, SIZE, 3), device=v.device)
    remaining = torch.ones((1, SIZE, SIZE, 1), device=v.device)
    with dr.DepthPeeler(ctx, clip.contiguous(), f, (SIZE, SIZE)) as peeler:
        for _ in range(LAYERS):
            rast, _ = peeler.rasterize_next_layer()
            hit = rast[..., 3:4] > 0
            if not hit.any():
                break
            t, _ = dr.interpolate(uv[None], rast, f)
            t = torch.stack([t[..., 0], 1 - t[..., 1]], -1).contiguous()
            rgba = dr.texture(tex, t, filter_mode="linear")
            nrm, _ = dr.interpolate(nn[None].contiguous(), rast, f)
            nrm = torch.nn.functional.normalize(nrm, dim=-1)
            col = rgba[..., :3] * (0.45 + 0.55 * (nrm @ light).abs()[..., None])
            a = torch.where(hit, rgba[..., 3:4], torch.zeros_like(remaining))
            out = out + remaining * a * col
            remaining = remaining * (1 - a)
    img = out + remaining * BG
    return (img[0].flip(0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def render(glb: str, out: str, ref: str | None = None) -> str:
    v, f, uv, n, tex = load(glb)
    ctx = dr.RasterizeCudaContext()
    angles = [(0, 0), (math.pi / 2, 0), (math.pi, 0), (math.pi / 4, -0.5)]
    tiles = [Image.fromarray(view(ctx, v, f, uv, n, tex, a, b)) for a, b in angles]
    if ref:
        r = open_image(Path(ref)).convert("RGBA")  # HEIC, phone rotation
        bg = Image.new("RGBA", r.size, (int(BG * 255),) * 3 + (255,))
        bg.alpha_composite(r)
        k = SIZE / max(bg.size)  # keep its shape; a tall cactus is not stretched sideways
        bg = bg.resize((max(1, round(bg.width * k)), max(1, round(bg.height * k))))
        tile = Image.new("RGB", (SIZE, SIZE), (int(BG * 255),) * 3)
        tile.paste(bg.convert("RGB"), ((SIZE - bg.width) // 2, (SIZE - bg.height) // 2))
        tiles.insert(0, tile)
    sheet = Image.new("RGB", (SIZE * len(tiles), SIZE), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * SIZE, 0))
    sheet.save(out, quality=88)
    return out


if __name__ == "__main__":
    render(*sys.argv[1:4])
