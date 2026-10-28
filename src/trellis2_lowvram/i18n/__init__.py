"""User-facing strings live in ko.json / en.json. Code only refers to keys."""

from __future__ import annotations

import json
import locale
import os
from functools import cache
from pathlib import Path

LANGS = ("ko", "en")
_DIR = Path(__file__).parent
_lang: str | None = None


@cache
def load(lang: str) -> dict[str, str]:
    return json.loads((_DIR / f"{lang}.json").read_text(encoding="utf-8"))


def detect_lang() -> str:
    """TRELLIS2_LOWVRAM_LANG wins, then the OS locale: Korean -> ko, anything else -> en."""
    env = os.environ.get("TRELLIS2_LOWVRAM_LANG", "").lower()
    if env in LANGS:
        return env
    candidates = [os.environ.get(k, "") for k in ("LC_ALL", "LC_MESSAGES", "LANG")]
    try:
        candidates.append(locale.getlocale()[0] or "")
    except ValueError:
        pass
    if os.name == "nt":
        try:
            import ctypes

            lcid = ctypes.windll.kernel32.GetUserDefaultUILanguage()  # type: ignore[attr-defined]
            candidates.append(locale.windows_locale.get(lcid, ""))
        except Exception:
            pass
    for c in candidates:
        if c.lower().startswith(("ko", "korean")):
            return "ko"
    return "en"


def set_lang(lang: str | None) -> None:
    global _lang
    _lang = lang if lang in LANGS else None


def get_lang() -> str:
    return _lang or detect_lang()


def t(key: str, **params: object) -> str:
    table = load(get_lang())
    text = table.get(key) or load("en").get(key) or key
    return text.format(**params) if params else text
