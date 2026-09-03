"""
QR codes for Robotour goals.

The competition passes a goal as a QR code with a geo URI payload
(``geo:lat,lon``, RFC 5870). :func:`geo_uri` formats the payload the way the
robot's ``qr_goal`` node parses it. :func:`qr_svg` renders it as vector art
(crisp at any size, the right choice for a screen or a printout);
:func:`qr_png` renders it with OpenCV's ``QRCodeEncoder`` (no extra dependency)
as a bitmap for anything that needs pixels.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

import cv2
import numpy as np

#: Modules of white border around the code (the QR standard asks for 4).
QUIET_ZONE_MODULES = 4
MAX_SCALE = 40
#: Caption band in module units: gap above the text, then the text's own line.
CAPTION_GAP_MODULES = 2.0
CAPTION_LINE_MODULES = 3.0
#: Cap on the caption size (module units) and the advance width of one
#: monospace character as a fraction of the font size.
CAPTION_MAX_SIZE_MODULES = 2.4
MONOSPACE_ADVANCE = 0.62
CAPTION_FONT_FAMILY = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


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
    ``caption`` (default: the payload itself, ``""`` for none) printed
    underneath so a printout can also be typed in by hand.

    The caption is baked into the pixels here, so it only looks right at the
    image's own resolution - use :func:`qr_svg` when it will be scaled.
    """
    scale = max(1, min(int(scale), MAX_SCALE))
    code = qr_matrix(text)
    q = QUIET_ZONE_MODULES
    padded = cv2.copyMakeBorder(code, q, q, q, q, cv2.BORDER_CONSTANT, value=255)
    img = cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    caption = text if caption is None else caption
    if caption:
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Size the caption off the code's width and derive the stroke weight
        # from that size, so the caption keeps the same proportions at every
        # scale instead of jumping a pixel between, say, 12 and 20.
        font_scale = max(0.4, img.shape[1] / 600.0)
        (tw, th), base = cv2.getTextSize(caption, font, font_scale, 1)
        if tw > img.shape[1] * 0.96:  # long payload: shrink until it fits
            font_scale *= img.shape[1] * 0.96 / tw
        thickness = max(1, round(font_scale * 1.8))
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


def _module_path(code: np.ndarray, offset: float) -> str:
    """SVG path data for the dark modules, runs of a row merged into one rect."""
    parts = []
    dark = code == 0
    for y, row in enumerate(dark):
        x = 0
        while x < row.size:
            if not row[x]:
                x += 1
                continue
            run = x
            while run < row.size and row[run]:
                run += 1
            parts.append(f"M{x + offset:g} {y + offset:g}h{run - x}v1h-{run - x}z")
            x = run
    return "".join(parts)


def qr_svg(text: str, scale: int = 12, caption: str | None = None) -> str:
    """
    The code as an SVG document: one module is one user unit, so it stays sharp
    at any size, and the ``caption`` (default: the payload, ``""`` for none) is
    a real ``<text>`` element rather than baked-in pixels.

    ``scale`` only sets the default ``width``/``height`` in pixels; the
    ``viewBox`` is what makes it scale.
    """
    scale = max(1, min(int(scale), MAX_SCALE))
    code = qr_matrix(text)
    q = QUIET_ZONE_MODULES
    side = code.shape[0] + 2 * q
    caption = text if caption is None else caption

    height = side
    body = f'<path fill="#000" shape-rendering="crispEdges" d="{_module_path(code, q)}"/>'
    if caption:
        # Shrink the text until it fits the code's width; a monospace glyph
        # advances about MONOSPACE_ADVANCE of the font size.
        size = min(
            CAPTION_MAX_SIZE_MODULES,
            side * 0.96 / (MONOSPACE_ADVANCE * len(caption)),
        )
        baseline = side + CAPTION_GAP_MODULES + size
        height = side + CAPTION_GAP_MODULES + CAPTION_LINE_MODULES
        body += (
            f'<text x="{side / 2:g}" y="{baseline:g}" fill="#000"'
            f' font-family="{CAPTION_FONT_FAMILY}" font-size="{size:g}"'
            f' text-anchor="middle">{escape(caption)}</text>'
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {side:g} {height:g}"'
        f' width="{side * scale:g}" height="{height * scale:g}"'
        f' role="img" aria-label="QR code for {escape(text)}">'
        f'<rect width="100%" height="100%" fill="#fff"/>{body}</svg>'
    )
