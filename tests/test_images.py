import pytest

Image = pytest.importorskip("PIL.Image")

from trellis2_lowvram.images import BadImage, open_image  # noqa: E402


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
