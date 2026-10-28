import http.server
import threading

import pytest

from trellis2_lowvram.ui import running


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setenv("TRELLIS2_LOWVRAM_LOGS", str(tmp_path))
    return tmp_path


@pytest.fixture
def server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/"
    srv.shutdown()


def test_second_launch_finds_the_running_page(logs, server):
    assert running.running_url() is None
    running.remember(server)
    assert running.running_url() == server
    running.forget()
    assert running.running_url() is None


def test_window_closed_without_cleanup(logs, server):
    running.remember(server.replace(server.split(":")[-1], "1/"))  # nothing listens on port 1
    assert running.running_url() is None
