"""Remember the web UI that is running from this install, so a second double-click reuses it.

Each UI process keeps every model in RAM (about 20GB measured on the RTX 3080 PC). Two at once on
a 32GB PC swap so hard that a 5 s stage took 6 minutes. No gradio import here: the CLI checks this
before deciding to start anything.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path

from ..config import get_paths

log = logging.getLogger(__name__)


def _file() -> Path:
    return get_paths().logs / "ui-running.json"


def running_url() -> str | None:
    """URL of a web UI from this install that is up and answering, else None."""
    try:
        info = json.loads(_file().read_text(encoding="utf-8"))
        no_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 127.0.0.1
        with no_proxy.open(info["url"], timeout=3) as r:
            if r.status == 200:
                return info["url"]
    except Exception as e:  # no file, stale file (crashed / closed window), not answering
        log.debug("no running UI: %r", e)
    return None


def remember(url: str) -> None:
    f = _file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"pid": os.getpid(), "url": url}), encoding="utf-8")


def forget() -> None:
    try:
        if json.loads(_file().read_text(encoding="utf-8")).get("pid") == os.getpid():
            _file().unlink()
    except Exception:
        pass
