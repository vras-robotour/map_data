"""The Socket.IO endpoint that pushes tracker telemetry to the browser."""

import signal

from map_data.viewer import app as viewer_app
from map_data.viewer.app import ACCESS_TOKEN_COOKIE, create_app, socketio


def test_a_client_can_connect(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = socketio.test_client(app)
    assert client.is_connected()
    client.disconnect()


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
