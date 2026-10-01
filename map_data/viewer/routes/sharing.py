"""Robot handoff routes: Robotour goal QR codes and magic-wormhole GPX transfers."""

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from flask import (
    Response,
    jsonify,
    request,
)
from flask.typing import ResponseReturnValue

from map_data.utils.qr import geo_uri, qr_png, qr_svg

from .common import (
    bp,
)

logger = logging.getLogger(__name__)


def _qr_request() -> tuple[str, int, str | None]:
    """``(geo URI, scale, caption)`` from the query string; raises on bad input.

    ``caption`` is the text printed under the code - absent means the payload
    itself, an empty ``?caption=`` means none (the viewer draws its own, as
    real text rather than pixels).
    """
    lat, lon = float(request.args["lat"]), float(request.args["lon"])
    return geo_uri(lat, lon), int(request.args.get("scale", 12)), request.args.get("caption")


def _qr_response(body: bytes | str, mimetype: str, text: str, suffix: str) -> Response:
    """Wrap a rendered code, adding the download file name when asked for."""
    resp = Response(body, mimetype=mimetype)
    resp.headers["X-Geo-URI"] = text
    if request.args.get("download") == "1":
        name = text[4:].replace(",", "_") + suffix
        resp.headers["Content-Disposition"] = f'attachment; filename="qr_{name}"'
    return resp


@bp.route("/api/qr")
@bp.route("/api/qr.svg")
def get_qr() -> ResponseReturnValue:
    """
    QR code of a Robotour goal, as PNG or (``.svg``) vector art:
    ``/api/qr[.svg]?lat=50.11&lon=14.41[&scale=12][&download=1][&caption=]``.

    The payload is the geo URI the robot's ``qr_goal`` node parses
    (``geo:lat,lon``); ``scale`` is pixels per module (1-40); ``download=1``
    sets a file name so the browser saves it. 400 on bad coordinates. The
    PNG bakes the caption into the pixels; the SVG is sharp at any size and
    its caption is real text, so prefer ``/api/qr.svg`` for a screen or a
    printer (``scale`` then only sets the default pixel size).
    """
    try:
        text, scale, caption = _qr_request()
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"lat/lon required: {e}"}), 400
    if request.path.endswith(".svg"):
        svg = qr_svg(text, scale=scale, caption=caption)
        return _qr_response(svg, "image/svg+xml", text, ".svg")
    return _qr_response(qr_png(text, scale=scale, caption=caption), "image/png", text, ".png")


class WormholeManager:
    """
    Manage ``magic-wormhole`` subprocesses that send a GPX file to a companion app.

    Each transfer runs an actual ``wormhole send`` subprocess against a
    temp file; a background thread scrapes its output for the generated
    wormhole code, and cleans up the process and temp directory once the
    transfer finishes, fails, or is cancelled. State for all in-flight
    transfers lives in ``active_transfers``, keyed by a generated
    ``transfer_id``.
    """

    def __init__(self) -> None:
        """Initialize with no active transfers, warning once if the ``wormhole`` CLI is missing."""
        self.active_transfers: dict[str, dict[str, Any]] = {}
        if shutil.which("wormhole") is None:
            # We don't want to crash the whole app if wormhole is missing,
            # just log it and the endpoints will fail gracefully.
            logger.warning("'wormhole' command not found. magic-wormhole is required for sharing.")

    def create_transfer(self, gpx_data: str) -> str:
        """
        Start a ``wormhole send`` subprocess for *gpx_data* and return its transfer ID.

        Writes *gpx_data* to a fresh temp directory, spawns ``wormhole
        send`` on it, and starts :meth:`_capture_wormhole_code_thread` in
        the background to scrape the resulting wormhole code from the
        process's output. The temp directory lives until
        :meth:`_cleanup_transfer` runs, once the background thread observes
        the process finish; it is removed immediately if the subprocess
        fails to even start.

        Raises
        ------
        RuntimeError
            If the ``wormhole`` subprocess fails to start.
        """
        transfer_id = str(uuid.uuid4())
        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        file_path = Path(temp_dir.name) / "path.gpx"
        file_path.write_text(gpx_data)

        cmd = ["wormhole", "send", str(file_path)]
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except Exception as e:
            temp_dir.cleanup()
            msg = f"Failed to start wormhole process: {e}"
            raise RuntimeError(msg) from e

        logger.info("Starting wormhole transfer %s (pid %s): %s", transfer_id, process.pid, cmd)
        self.active_transfers[transfer_id] = {
            "process": process,
            "temp_dir": temp_dir,
            "code": None,
            "code_ready": threading.Event(),
        }
        threading.Thread(
            target=self._capture_wormhole_code_thread,
            args=(transfer_id,),
            daemon=True,
        ).start()
        return transfer_id

    def _capture_wormhole_code_thread(self, transfer_id: str) -> None:
        """
        Background-thread target: scrape the wormhole code and await process exit.

        Reads the subprocess's output until the ``"Wormhole code is: ..."``
        line, records it on ``active_transfers[transfer_id]["code"]`` and
        sets its ``"code_ready"`` event (waking up :meth:`get_transfer_code`),
        then waits (up to 60s) for the process to exit. Always calls
        :meth:`_cleanup_transfer` on exit (success, failure, or exception).
        """
        transfer_info = self.active_transfers.get(transfer_id)
        if not transfer_info:
            return
        process = transfer_info["process"]
        try:
            for line in process.stdout:
                match = re.search(r"Wormhole code is: (\S+-\S+-\S+)", line)
                if match:
                    transfer_info["code"] = match.group(1)
                    transfer_info["code_ready"].set()
                    logger.info("Wormhole code for transfer %s: %s", transfer_id, match.group(1))
                    break
                if line.strip():
                    logger.warning("Wormhole output (%s): %s", transfer_id, line.strip())
            process.wait(timeout=60)
        except Exception:
            logger.exception("Error in wormhole thread for %s", transfer_id)
            if process.poll() is None:
                process.kill()
        finally:
            self._cleanup_transfer(transfer_id)

    def get_transfer_code(self, transfer_id: str, timeout: float = 10) -> str | None:
        """
        Block until a transfer's wormhole code is available, or *timeout* elapses.

        Returns the code, or ``None`` if *transfer_id* is unknown or the code
        was not captured within *timeout*.
        """
        transfer_info = self.active_transfers.get(transfer_id)
        if transfer_info and transfer_info["code_ready"].wait(timeout):
            return str(transfer_info["code"])
        return None

    def cancel_transfer(self, transfer_id: str) -> tuple[bool, str]:
        """
        Kill an active transfer's ``wormhole`` subprocess.

        Bookkeeping and the temp directory are still cleaned up by
        :meth:`_capture_wormhole_code_thread` observing the process exit.

        Returns
        -------
        tuple of (bool, str)
            ``(True, "Transfer cancelled")`` on success, or ``(False,
            "Invalid or unknown transfer ID")`` if *transfer_id* isn't active.
        """
        if transfer_id not in self.active_transfers:
            return False, "Invalid or unknown transfer ID"
        logger.info("Cancelling wormhole transfer %s", transfer_id)
        process = self.active_transfers[transfer_id]["process"]
        if process.poll() is None:
            process.kill()
        return True, "Transfer cancelled"

    def _cleanup_transfer(self, transfer_id: str) -> None:
        """Remove a finished transfer's bookkeeping entry and delete its temp directory."""
        transfer = self.active_transfers.pop(transfer_id, None)
        if transfer:
            transfer["temp_dir"].cleanup()


wormhole_manager = WormholeManager()


@bp.route("/api/create_wormhole", methods=["POST"])
def create_wormhole() -> ResponseReturnValue:
    """
    Start sending a GPX path to a companion app via ``magic-wormhole``.

    Blocks up to 15s waiting for the wormhole code to be captured (see
    :meth:`WormholeManager.get_transfer_code`); if it isn't ready in
    time, the transfer is cancelled and an error is returned rather than
    leaving an orphaned transfer running.

    Parameters (JSON body)
    -----------------------
    gpx : str
        Raw GPX file contents to send.

    Returns
    -------
    Response
        JSON ``{"success": true, "code": ..., "transfer_id": ...}`` on
        success (status 200); on failure, ``{"success": false,
        "message": ...}`` with status 400 (missing ``gpx``) or 500
        (wormhole code not captured in time, or an internal error).

    """
    gpx_data = (request.get_json(silent=True) or {}).get("gpx")
    if not gpx_data:
        return jsonify({"success": False, "message": "No GPX data provided"}), 400

    try:
        transfer_id = wormhole_manager.create_transfer(gpx_data)
        code = wormhole_manager.get_transfer_code(transfer_id, timeout=15)
        if code:
            return jsonify({"success": True, "code": code, "transfer_id": transfer_id})
        wormhole_manager.cancel_transfer(transfer_id)
        return jsonify(
            {"success": False, "message": "Failed to capture wormhole code in time"},
        ), 500
    except Exception:
        logger.exception("Error creating wormhole")
        return jsonify({"success": False, "message": "Internal server error"}), 500


@bp.route("/api/cancel_wormhole", methods=["POST"])
def cancel_wormhole() -> ResponseReturnValue:
    """
    Cancel an in-progress ``magic-wormhole`` GPX transfer.

    Returns
    -------
    ResponseReturnValue
        JSON ``{"success": ..., "message": ...}`` per
        :meth:`WormholeManager.cancel_transfer`; ``{"success": false,
        "message": ...}`` with status 400 if the request body is not
        JSON or lacks ``transfer_id``.

    """
    transfer_id = (request.get_json(silent=True) or {}).get("transfer_id")
    if not transfer_id:
        return jsonify({"success": False, "message": "No transfer_id provided"}), 400
    success, message = wormhole_manager.cancel_transfer(transfer_id)
    return jsonify({"success": success, "message": message})
