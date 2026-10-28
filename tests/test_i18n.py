import re
import string
from pathlib import Path

from trellis2_lowvram import i18n

SRC = Path(i18n.__file__).parents[1]


def fields(text):
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


def test_same_keys_in_every_language():
    tables = {lang: i18n.load(lang) for lang in i18n.LANGS}
    keys = set(tables["en"])
    for lang, table in tables.items():
        assert set(table) == keys, f"{lang}: {set(table) ^ keys}"


def test_same_placeholders_in_every_language():
    en, ko = i18n.load("en"), i18n.load("ko")
    for k in en:
        assert fields(en[k]) == fields(ko[k]), k


def test_keys_used_in_code_exist():
    """String literals that look like i18n keys must be defined."""
    en = i18n.load("en")
    pattern = re.compile(r"\"((?:check|fix|doctor|login|download|ext|error|install)\.[a-z_.]+)\"")
    used = set()
    for py in SRC.rglob("*.py"):
        used |= set(pattern.findall(py.read_text(encoding="utf-8")))
    # keys built with f-strings
    dynamic = {k for k in en if k.startswith(("doctor.result.", "check.hf."))}
    missing = used - set(en)
    assert not missing, missing
    assert dynamic


def test_no_em_dash_in_korean():
    for k, v in i18n.load("ko").items():
        assert "\u2014" not in v, k


def test_env_override(monkeypatch):
    monkeypatch.setenv("TRELLIS2_LOWVRAM_LANG", "ko")
    i18n.set_lang(None)
    assert i18n.get_lang() == "ko"
    monkeypatch.setenv("TRELLIS2_LOWVRAM_LANG", "en")
    assert i18n.t("doctor.title", version="x").startswith("trellis2-lowvram x")


def test_install_scripts_have_a_message_for_every_step():
    """install.sh / install.ps1 print install.step<N>.fail for steps 2-6 (step 1 is inline)."""
    root = SRC.parents[1]
    for lang in i18n.LANGS:
        table = i18n.load(lang)
        for n in range(2, 7):
            assert f"install.step{n}.fail" in table
        assert "install.retry" in table
    for script in ("install.sh", "scripts/install.ps1"):
        text = (root / script).read_text(encoding="utf-8-sig")
        assert "install.step" in text and "install.retry" in text
