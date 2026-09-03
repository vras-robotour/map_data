"""
QR codes for Robotour goals.

The competition passes a goal as a QR code with a geo URI payload
(``geo:lat,lon``, RFC 5870). :func:`geo_uri` formats the payload the way the
robot's ``qr_goal`` node parses it; :func:`qr_png` renders it with OpenCV's
``QRCodeEncoder`` (no extra dependency) as a PNG that can be shown on a screen
or printed.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Modules of white border around the code (the QR standard asks for 4).
QUIET_ZONE_MODULES = 4
MAX_SCALE = 40


def geo_uri(lat: float, lon: float, decimals: int = 7) -> str:
    """``geo:50.1103476,14.4159857`` - 7 decimals is about 1 cm."""
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError(f"latitude/longitude out of range: {lat}, {lon}")
    return f"geo:{lat:.{decimals}f},{lon:.{decimals}f}"


def qr_matrix(text: str) -> np.ndarray:
    """The raw code as a uint8 image (0 = dark, 255 = light), one pixel per module."""
    if not hasattr(cv2, "QRCodeEncoder"):
        raise RuntimeError("this OpenCV build has no QRCodeEncoder (need >= 4.8)")
    params = cv2.QRCodeEncoder.Params()
    # Medium error correction; the constant moved between OpenCV 4 and 5.
    level = next(
        (
            getattr(cv2, n)
            for n in ("QRCodeEncoder_CORRECT_LEVEL_M", "QRCODE_ENCODER_CORRECT_LEVEL_M")
            if hasattr(cv2, n)
        ),
        None,
    )
    if level is not None:
        params.correction_level = level
    code = cv2.QRCodeEncoder.create(params).encode(text)
    if code is None or code.size == 0:
        raise ValueError(f"cannot encode {text!r} as a QR code")
    return code


def qr_image(text: str, scale: int = 12, caption: str | None = None) -> np.ndarray:
    """
    Grey image of the code: ``scale`` pixels per module, a quiet zone, and the
    ``caption`` (default: the payload itself) printed underneath so a printout
    can also be typed in by hand.
    """
    scale = max(1, min(int(scale), MAX_SCALE))
    code = qr_matrix(text)
    q = QUIET_ZONE_MODULES
    padded = cv2.copyMakeBorder(code, q, q, q, q, cv2.BORDER_CONSTANT, value=255)
    img = cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    caption = text if caption is None else caption
    if caption:
        font, thickness = cv2.FONT_HERSHEY_SIMPLEX, max(1, scale // 8)
        font_scale = max(0.4, min(2.0, img.shape[1] / 600.0))
        (tw, th), base = cv2.getTextSize(caption, font, font_scale, thickness)
        strip = np.full((th + base + 3 * scale, img.shape[1]), 255, dtype=np.uint8)
        x = max(0, (img.shape[1] - tw) // 2)
        cv2.putText(strip, caption, (x, th + scale), font, font_scale, 0, thickness, cv2.LINE_AA)
        img = np.vstack([img, strip])
    return img


def qr_png(text: str, scale: int = 12, caption: str | None = None) -> bytes:
    """PNG bytes of :func:`qr_image`."""
    ok, buf = cv2.imencode(".png", qr_image(text, scale, caption))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
