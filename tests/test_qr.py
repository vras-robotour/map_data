"""Robotour goal QR codes: generation roundtrips through OpenCV's detector and the viewer API."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from map_data.utils.qr import geo_uri, qr_image, qr_png  # noqa: E402
from map_data.viewer.app import create_app  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(cv2, "QRCodeEncoder"), reason="OpenCV build without QRCodeEncoder"
)


def _decode(png: bytes) -> str:
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    return text


def test_geo_uri_format():
    assert geo_uri(50.1103476, 14.4159857) == "geo:50.1103476,14.4159857"
    assert geo_uri(-33.9, 151.2) == "geo:-33.9000000,151.2000000"
    with pytest.raises(ValueError):
        geo_uri(91.0, 0.0)


def test_qr_png_roundtrip():
    text = geo_uri(50.1103476, 14.4159857)
    png = qr_png(text, scale=10)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert _decode(png) == text


def test_qr_image_has_quiet_zone_and_caption():
    img = qr_image("geo:50.1,14.4", scale=8)
    assert img.ndim == 2
    assert img[:8, :].min() == 255 and img[:, :8].min() == 255  # white border
    assert qr_image("geo:50.1,14.4", scale=8, caption="").shape[0] < img.shape[0]


def test_api_qr(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.get("/api/qr?lat=50.1103476&lon=14.4159857")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    assert resp.headers["X-Geo-URI"] == "geo:50.1103476,14.4159857"
    assert _decode(resp.data) == "geo:50.1103476,14.4159857"

    resp = client.get("/api/qr?lat=50.1&lon=14.4&scale=20&download=1")
    assert resp.status_code == 200
    assert resp.headers["Content-Disposition"].endswith('qr_50.1000000_14.4000000.png"')

    assert client.get("/api/qr?lat=91&lon=14").status_code == 400
    assert client.get("/api/qr?lon=14").status_code == 400
    assert client.get("/api/qr?lat=abc&lon=14").status_code == 400
