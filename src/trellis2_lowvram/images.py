"""Opening input pictures the way the phone or camera showed them.

- Phones store portrait photos sideways plus an EXIF "rotate" tag; Pillow ignores the tag, so the
  model would get a picture lying on its side. The web UI is fine (Gradio applies the tag on
  upload); the CLI and make3d drag-and-drop go through here.
- iPhones save HEIC by default, which Pillow cannot read without pillow-heif.

Transparent pictures are framed here too (frame_cutout), not by upstream preprocess_image.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageOps

log = logging.getLogger(__name__)


class BadImage(ValueError):
    """The file exists but is not a picture we can read."""

    def __init__(self, path: Path):
        super().__init__(f"cannot read image {path}")
        self.path = path


def register_heif() -> bool:
    """Teach Pillow (and Gradio, which uses Pillow) to open HEIC/HEIF. Safe to call repeatedly."""
    try:
        import pillow_heif
    except ImportError:  # older installs; HEIC files then fail with BadImage
        return False
    pillow_heif.register_heif_opener()
    return True


def open_image(path: Path) -> Image.Image:
    register_heif()
    try:
        im = Image.open(path)
        im.load()
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        log.info("cannot open %s: %r", path, e)
        raise BadImage(Path(path)) from e
    return ImageOps.exif_transpose(im)


# Alpha (0-255) above which a pixel counts as part of the object when framing a cut-out.
# Upstream frames at alpha > 204, made for the crisp edges of its own background removal; painted
# cut-outs with soft alpha (leaves, twigs, grass tips) then lose everything fainter than that.
CUTOUT_ALPHA = 25


def frame_cutout(im: Image.Image) -> Image.Image | None:
    """A transparent picture cropped to a square around the object and flattened onto black.

    The same output as upstream preprocess_image (square crop around the object, no margin,
    colour times alpha), except that soft edges count as part of the object. None when the
    picture has no transparency and needs background removal instead.
    """
    if im.mode in ("LA", "P", "PA") or "transparency" in im.info:
        im = im.convert("RGBA")
    if im.mode != "RGBA":
        return None
    alpha = im.getchannel("A")
    if alpha.getextrema()[0] == 255:
        return None
    box = alpha.point(lambda v: 255 if v > CUTOUT_ALPHA else 0).getbbox() or alpha.getbbox()
    if box is None:
        return None
    x0, y0, x1, y1 = box
    size = max(x1 - x0, y1 - y0)
    left, top = x0 + (x1 - x0 - size) // 2, y0 + (y1 - y0 - size) // 2
    im = im.crop((left, top, left + size, top + size))  # outside the picture: transparent
    return Image.composite(im.convert("RGB"), Image.new("RGB", im.size), im.getchannel("A"))
