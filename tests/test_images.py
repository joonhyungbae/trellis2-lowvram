import pytest

Image = pytest.importorskip("PIL.Image")

from trellis2_lowvram.images import BadImage, frame_cutout, open_image  # noqa: E402


def test_phone_portrait_is_turned_upright(tmp_path):
    # Stored 40 wide x 20 high with "rotate 90" (orientation 6), the way phones save portraits.
    path = tmp_path / "portrait.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (40, 20), "red").save(path, exif=exif)
    assert Image.open(path).size == (40, 20)
    assert open_image(path).size == (20, 40)


def test_transparent_png_keeps_alpha(tmp_path):
    path = tmp_path / "cutout.png"
    Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(path)
    assert open_image(path).mode == "RGBA"


def test_not_a_picture(tmp_path):
    path = tmp_path / "notes.jpg"
    path.write_text("not an image", encoding="utf-8")
    with pytest.raises(BadImage) as e:
        open_image(path)
    assert e.value.path == path


def test_iphone_heic(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    path = tmp_path / "IMG_0001.HEIC"
    pillow_heif.from_pillow(Image.new("RGB", (16, 12), "blue")).save(path)
    assert open_image(path).size == (16, 12)


def test_cutout_frame_keeps_soft_edges():
    # A solid trunk 2 wide and faint twigs (alpha 60) reaching further up and out.
    im = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    im.paste((200, 100, 0, 255), (19, 20, 21, 36))
    im.paste((255, 0, 0, 60), (10, 6, 30, 20))
    out = frame_cutout(im)
    assert out.mode == "RGB" and out.size == (30, 30)  # twig top 6 to trunk bottom 36
    assert out.getpixel((0, 0)) == (0, 0, 0)
    assert out.getpixel((15, 5)) == (60, 0, 0)  # colour times alpha, like upstream


def test_cutout_frame_ignores_a_faint_haze():
    im = Image.new("RGBA", (40, 40), (255, 255, 255, 10))
    im.paste((0, 255, 0, 255), (10, 10, 20, 30))
    assert frame_cutout(im).size == (20, 20)


def test_opaque_picture_needs_background_removal():
    assert frame_cutout(Image.new("RGB", (8, 8))) is None
    assert frame_cutout(Image.new("RGBA", (8, 8), (1, 2, 3, 255))) is None
