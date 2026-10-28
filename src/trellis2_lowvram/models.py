"""Which weight files a pipeline type needs, and downloading only those.

Upstream `Pipeline.from_pretrained` loads all 8 models listed in pipeline.json. We resolve the
same paths the same way (`<repo>/<v>` first, then `v` as a full `<org>/<repo>/<file>` path) but
only for the models the chosen pipeline type actually runs (see `Trellis2ImageTo3DPipeline.run`).
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable
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


def download(
    pipeline_types: tuple[str, ...] = ("512",),
    on_file: Callable[[int, int, str], None] | None = None,
) -> None:
    """Fetch only what the given pipeline types need. Resumes partial downloads.

    on_file(done, total, name) is called before each file (progress for the web UI).
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    config = load_pipeline_config()
    refs: dict[tuple[str, str], WeightRef] = {}
    for pt in pipeline_types:
        for r in weights_for(pt, config):
            refs[(r.repo_id, r.name)] = r
    image_cond = config["image_cond_model"]["args"]["model_name"]
    jobs: list[tuple[str, Callable[[], object]]] = []
    for r in refs.values():
        for ext in (".json", ".safetensors"):
            jobs.append(
                (f"{r.name}{ext}", functools.partial(hf_hub_download, r.repo_id, r.name + ext))
            )
    jobs.append(
        (
            image_cond,
            functools.partial(
                snapshot_download, image_cond, allow_patterns=["*.json", "*.safetensors", "*.md"]
            ),
        )
    )
    jobs.append(
        (
            REMBG_REPO,
            functools.partial(
                snapshot_download,
                REMBG_REPO,
                allow_patterns=["*.json", "*.py", "*.safetensors", "*.md"],
            ),
        )
    )
    for i, (name, fetch) in enumerate(jobs):
        log.info("download %s", name)
        if on_file:
            on_file(i, len(jobs), name.split("/")[-1])
        fetch()
    if on_file:
        on_file(len(jobs), len(jobs), "")


def missing_files(pipeline_type: str = "512") -> list[str]:
    """Files a pipeline type needs that are not in the local cache (no network)."""
    from huggingface_hub import try_to_load_from_cache

    def cached(repo: str, name: str) -> bool:
        return isinstance(try_to_load_from_cache(repo, name), str)

    if not cached(TRELLIS_REPO, "pipeline.json"):
        return [f"{TRELLIS_REPO}/pipeline.json"]
    import json

    with open(try_to_load_from_cache(TRELLIS_REPO, "pipeline.json"), encoding="utf-8") as f:
        config = json.load(f)["args"]
    want = [
        (r.repo_id, r.name + ext)
        for r in weights_for(pipeline_type, config)
        for ext in (".json", ".safetensors")
    ]
    image_cond = config["image_cond_model"]["args"]["model_name"]
    want += [(image_cond, "config.json"), (image_cond, "model.safetensors")]
    want += [
        (REMBG_REPO, "config.json"),
        (REMBG_REPO, "model.safetensors"),
        (REMBG_REPO, "birefnet.py"),
    ]
    return [f"{repo}/{name}" for repo, name in want if not cached(repo, name)]
