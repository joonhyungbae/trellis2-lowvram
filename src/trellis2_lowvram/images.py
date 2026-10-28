"""Opening input pictures the way the phone or camera showed them.

- Phones store portrait photos sideways plus an EXIF "rotate" tag; Pillow ignores the tag, so the
  model would get a picture lying on its side. The web UI is fine (Gradio applies the tag on
  upload); the CLI and make3d drag-and-drop go through here.
- iPhones save HEIC by default, which Pillow cannot read without pillow-heif.
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
