import pytest

pytest.importorskip("torch")
np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")
Image = pytest.importorskip("PIL.Image")

from trellis2_lowvram.pipeline import set_alpha_mode  # noqa: E402


def mesh_with_alpha(alphas: list[int]):
    """One triangle per alpha value, each reading its own texel of a 1-row texture."""
    n = len(alphas)
    tex = np.full((1, n, 4), 200, dtype=np.uint8)
    tex[0, :, 3] = alphas
    u = np.repeat(np.arange(n) / max(n - 1, 1), 3)
    uv = np.stack([u, np.ones(3 * n)], 1)
    vertices = np.random.default_rng(0).random((3 * n, 3))
    faces = np.arange(3 * n).reshape(n, 3)
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(tex), alphaMode="OPAQUE"
    )
    visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    return trimesh.Trimesh(vertices, faces, visual=visual, process=False)


@pytest.mark.parametrize(
    "alphas, mode",
    [
        ([255] * 100, "OPAQUE"),
        ([60] * 10 + [255] * 90, "BLEND"),  # glass in a solid object
        ([60] * 90 + [255] * 10, "MASK"),  # a painted cut-out: soft alpha all over
        ([0] * 100, "OPAQUE"),  # MASK would hide everything
    ],
)
def test_alpha_mode(alphas, mode):
    assert set_alpha_mode(mesh_with_alpha(alphas)) == mode


def test_mask_drops_faces_nobody_would_see():
    # A ground sheet the model marked invisible (alpha 0) under a soft-alpha cactus.
    mesh = mesh_with_alpha([0] * 60 + [80] * 40)
    assert set_alpha_mode(mesh) == "MASK"
    assert len(mesh.faces) == 40
    assert mesh.visual.material.alphaCutoff == pytest.approx(0.1)


def test_tex_guidance_off_stays_upstream():
    from trellis2_lowvram.pipeline import tex_guidance

    assert tex_guidance(1.0) == {"guidance_strength": 1.0}
    assert tex_guidance(3.0)["guidance_interval"] == [0.0, 1.0]
