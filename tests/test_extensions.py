import json

from trellis2_lowvram.extensions import MANIFEST, entries_for


def test_manifest_shape():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for key, entries in m["wheels"].items():
        assert key in ("win_amd64", "linux_x86_64")
        for e in entries:
            assert (
                e["file"].endswith(".whl")
                and len(e["sha256"]) == 64
                and e["url"].startswith("https://")
            )
    assert entries_for(m, "unsupported") == []
