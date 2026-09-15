"""The Socket.IO endpoint that pushes tracker telemetry to the browser."""

from map_data.viewer.app import create_app, socketio


def test_a_client_can_connect(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = socketio.test_client(app)
    assert client.is_connected()
    client.disconnect()
