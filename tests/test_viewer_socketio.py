"""The Socket.IO endpoint that pushes tracker telemetry to the browser."""

import signal

from map_data.viewer import app as viewer_app
from map_data.viewer.app import ACCESS_TOKEN_COOKIE, create_app, socketio


def test_a_client_can_connect(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = socketio.test_client(app)
    assert client.is_connected()
    client.disconnect()


def test_a_client_without_the_access_token_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("MAP_DATA_ACCESS_TOKEN", "sekret")
    app = create_app(data_dir=str(tmp_path))

    assert not socketio.test_client(app).is_connected()
    assert not socketio.test_client(app, query_string="access_token=wrong").is_connected()
    good = socketio.test_client(app, query_string="access_token=sekret")
    assert good.is_connected()
    good.disconnect()
    header = socketio.test_client(app, headers={"X-Access-Token": "sekret"})
    assert header.is_connected()
    header.disconnect()


def test_a_non_ascii_token_is_compared_not_crashed_on(tmp_path, monkeypatch):
    monkeypatch.setenv("MAP_DATA_ACCESS_TOKEN", "heslo-č")
    app = create_app(data_dir=str(tmp_path))
    client = app.test_client()

    assert client.get("/api/files?access_token=%C3%A9").status_code == 401
    assert client.get("/?access_token=heslo-%C4%8D").status_code == 302
    assert client.get("/api/files").status_code == 200  # by the cookie just set
    assert not socketio.test_client(app, query_string="access_token=%C3%A9").is_connected()
    good = socketio.test_client(app, query_string="access_token=heslo-%C4%8D")
    assert good.is_connected()
    good.disconnect()


def test_the_token_is_stripped_from_the_url_after_setting_the_cookie(tmp_path, monkeypatch):
    monkeypatch.setenv("MAP_DATA_ACCESS_TOKEN", "sekret")
    client = create_app(data_dir=str(tmp_path)).test_client()

    resp = client.get("/?access_token=sekret&file=a.mapdata")
    assert resp.status_code == 302
    assert resp.headers["Location"] == "/?file=a.mapdata"
    assert ACCESS_TOKEN_COOKIE in resp.headers["Set-Cookie"]
    assert client.get("/").status_code == 200  # the cookie alone authenticates now
    assert client.get("/?access_token=wrong").status_code == 401


def test_main_exits_cleanly_on_sigint(tmp_path, monkeypatch):
    stopped = []
    monkeypatch.setattr(viewer_app, "stop_ros", lambda: stopped.append(True))
    monkeypatch.setattr(viewer_app, "create_app", lambda **_: object())

    def serve(*_, **__):
        raise KeyboardInterrupt

    monkeypatch.setattr(socketio, "run", serve)
    monkeypatch.setattr("sys.argv", ["map_data_viewer", "--data-dir", str(tmp_path)])
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # as inherited by a background job
        viewer_app.main()
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        assert signal.getsignal(signal.SIGTERM) is signal.default_int_handler
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
    assert stopped == [True]
