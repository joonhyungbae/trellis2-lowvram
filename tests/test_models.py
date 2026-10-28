from trellis2_lowvram.models import MODELS_BY_PIPELINE, TRELLIS_REPO, resolve, weights_for

CONFIG = {
    "models": {
        "sparse_structure_decoder": "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
        "sparse_structure_flow_model": "ckpts/ss_flow_img_dit_1_3B_64_bf16",
        "shape_slat_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
        "shape_slat_flow_model_512": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
        "shape_slat_flow_model_1024": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
        "tex_slat_decoder": "ckpts/tex_dec_next_dc_f16c32_fp16",
        "tex_slat_flow_model_512": "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16",
        "tex_slat_flow_model_1024": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
    }
}


def test_resolve_local_and_foreign_repo():
    r = resolve("a", "ckpts/x")
    assert (r.repo_id, r.name) == (TRELLIS_REPO, "ckpts/x")
    r = resolve("b", "microsoft/TRELLIS-image-large/ckpts/y")
    assert (r.repo_id, r.name) == ("microsoft/TRELLIS-image-large", "ckpts/y")


def test_512_never_needs_1024_weights():
    names = {w.name for w in weights_for("512", CONFIG)}
    assert not any("1024" in n for n in names)
    assert len(names) == 6


def test_every_listed_model_exists_in_config():
    for keys in MODELS_BY_PIPELINE.values():
        assert set(keys) <= set(CONFIG["models"])
