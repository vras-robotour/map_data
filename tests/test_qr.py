"""Robotour goal QR codes: generation roundtrips through OpenCV's detector and the viewer API."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from map_data.utils.qr import (  # noqa: E402
    MONOSPACE_ADVANCE,
    geo_uri,
    qr_image,
    qr_png,
    qr_svg,
)
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


def test_qr_svg_is_vector_with_real_text():
    text = geo_uri(50.1103476, 14.4159857)
    svg = qr_svg(text, scale=10)
    assert svg.startswith("<?xml") and svg.rstrip().endswith("</svg>")
    # One module is one user unit, so the viewBox - not a pixel size - is what
    # makes it scale; the caption is a <text> element, not baked-in pixels.
    assert 'viewBox="0 0 37 ' in svg
    assert f">{text}</text>" in svg
    assert "<text" not in qr_svg(text, caption="")


def test_qr_svg_escapes_and_fits_the_caption():
    svg = qr_svg("geo:50.1,14.4", caption='a & b <"c">')
    assert "&amp;" in svg and "&lt;" in svg and "<text" in svg
    # A long caption shrinks rather than running off the side of the code.
    wide = qr_svg("geo:50.1,14.4", caption="x" * 200)
    size = float(wide.split('font-size="')[1].split('"')[0])
    assert MONOSPACE_ADVANCE * size * 200 < 33  # 25 modules plus 8 of quiet zone


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

    # The viewer asks for the bare code and draws the caption as real text.
    plain = client.get("/api/qr?lat=50.1&lon=14.4&caption=")
    assert len(plain.data) < len(client.get("/api/qr?lat=50.1&lon=14.4").data)

    assert client.get("/api/qr?lat=91&lon=14").status_code == 400
    assert client.get("/api/qr?lon=14").status_code == 400
    assert client.get("/api/qr?lat=abc&lon=14").status_code == 400


def test_api_qr_svg(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.get("/api/qr.svg?lat=50.1103476&lon=14.4159857")
    assert resp.status_code == 200
    assert resp.mimetype == "image/svg+xml"
    assert resp.headers["X-Geo-URI"] == "geo:50.1103476,14.4159857"
    assert b"<text" in resp.data and b"viewBox" in resp.data

    assert b"<text" not in client.get("/api/qr.svg?lat=50.1&lon=14.4&caption=").data

    resp = client.get("/api/qr.svg?lat=50.1&lon=14.4&download=1")
    assert resp.headers["Content-Disposition"].endswith('qr_50.1000000_14.4000000.svg"')

    assert client.get("/api/qr.svg?lat=91&lon=14").status_code == 400
    assert client.get("/api/qr.svg?lat=abc&lon=14").status_code == 400
