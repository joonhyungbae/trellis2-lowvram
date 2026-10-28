"""Which weight files a pipeline type needs, and downloading only those.

Upstream `Pipeline.from_pretrained` loads all 8 models listed in pipeline.json. We resolve the
same paths the same way (`<repo>/<v>` first, then `v` as a full `<org>/<repo>/<file>` path) but
only for the models the chosen pipeline type actually runs (see `Trellis2ImageTo3DPipeline.run`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

TRELLIS_REPO = "microsoft/TRELLIS.2-4B"
# Background removal: upstream pipeline.json uses briaai/RMBG-2.0 (gated, non-commercial).
# We default to the MIT-licensed BiRefNet that upstream's BiRefNet class also loads (PLAN Q5).
REMBG_REPO = "ZhengPeng7/BiRefNet"

MODELS_BY_PIPELINE: dict[str, tuple[str, ...]] = {
    "512": (
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_decoder",
        "tex_slat_flow_model_512",
        "tex_slat_decoder",
    ),
    "1024_cascade": (
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "shape_slat_decoder",
        "tex_slat_flow_model_1024",
        "tex_slat_decoder",
    ),
}
PIPELINE_TYPES = tuple(MODELS_BY_PIPELINE)


@dataclass(frozen=True)
class WeightRef:
    key: str
    repo_id: str
    name: str  # path inside the repo without extension; files are <name>.json / <name>.safetensors


def resolve(key: str, value: str, repo_id: str = TRELLIS_REPO) -> WeightRef:
    """Mirror upstream: "ckpts/x" lives in repo_id; "org/repo/ckpts/x" names another repo."""
    parts = value.split("/")
    if len(parts) > 2 and not value.startswith("ckpts/"):
        return WeightRef(key, f"{parts[0]}/{parts[1]}", "/".join(parts[2:]))
    return WeightRef(key, repo_id, value)


def load_pipeline_config(repo_id: str = TRELLIS_REPO) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "pipeline.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["args"]


def weights_for(pipeline_type: str, config: dict) -> list[WeightRef]:
    return [resolve(k, config["models"][k]) for k in MODELS_BY_PIPELINE[pipeline_type]]


def download(pipeline_types: tuple[str, ...] = ("512",)) -> None:
    """Fetch only what the given pipeline types need. Resumes partial downloads."""
    from huggingface_hub import hf_hub_download, snapshot_download

    config = load_pipeline_config()
    refs: dict[tuple[str, str], WeightRef] = {}
    for pt in pipeline_types:
        for r in weights_for(pt, config):
            refs[(r.repo_id, r.name)] = r
    for r in refs.values():
        for ext in (".json", ".safetensors"):
            log.info("download %s/%s%s", r.repo_id, r.name, ext)
            hf_hub_download(r.repo_id, r.name + ext)

    image_cond = config["image_cond_model"]["args"]["model_name"]
    log.info("download %s", image_cond)
    snapshot_download(image_cond, allow_patterns=["*.json", "*.safetensors", "*.md"])
    log.info("download %s", REMBG_REPO)
    snapshot_download(REMBG_REPO, allow_patterns=["*.json", "*.py", "*.safetensors", "*.md"])
