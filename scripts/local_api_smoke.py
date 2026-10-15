"""Check the local FastAPI -> checkpoint -> prediction response, without MVTec data.

Run after starting uvicorn on the Mac:
    python -m scripts.local_api_smoke --url http://127.0.0.1:8001
    python -m scripts.local_api_smoke --url http://127.0.0.1:8001 --image /path/to/bottle.png

The default sample is a synthetic solid-color image. Its anomaly score has no
scientific/quality-control meaning; it only tests the service integration.
"""
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
import uuid

from PIL import Image


def make_png(image_path: Path | None = None) -> bytes:
    if image_path is not None:
        with Image.open(image_path) as img:
            output = BytesIO()
            img.convert("RGB").save(output, format="PNG")
            return output.getvalue()
    output = BytesIO()
    Image.new("RGB", (256, 256), (180, 180, 180)).save(output, format="PNG")
    return output.getvalue()


def multipart_png(payload: bytes) -> tuple[bytes, str]:
    boundary = "ihsanvision-" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="smoke.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + payload + f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def smoke(base_url: str, payload: bytes, timeout: float = 180) -> dict:
    """Call health, UI and prediction endpoints; never print the base64 overlay."""
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Smoke test intentionally accepts localhost HTTP origin URLs only")
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid localhost port") from None
    if port is None:
        raise ValueError("Specify the local server port, for example :8001")
    base = f"http://{parsed.hostname}:{port}"
    with urllib.request.urlopen(base + "/health", timeout=10) as response:
        health = json.load(response)
    if not health.get("model_configured"):
        raise RuntimeError("Server is running but IHSANVISION_CHECKPOINT is not configured")
    with urllib.request.urlopen(base + "/", timeout=10) as response:
        if response.status != 200 or b"IhsanVision" not in response.read():
            raise RuntimeError("Dashboard did not return expected HTML")
    body, content_type = multipart_png(payload)
    request = urllib.request.Request(
        base + "/predict", data=body, method="POST",
        headers={"Content-Type": content_type},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)

    if result.get("model") not in ("padim", "patchcore"):
        raise RuntimeError("Prediction JSON has no known model")
    if not isinstance(result.get("anomaly_score"), (int, float)):
        raise RuntimeError("Prediction JSON has no numeric anomaly_score")
    overlay = result.get("heatmap_png_base64")
    if overlay:
        raw = base64.b64decode(overlay, validate=True)
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("Overlay is not PNG")
    return {
        "api": base,
        "model": result["model"],
        "anomaly_score": result["anomaly_score"],
        "predicted_anomaly": result.get("predicted_anomaly"),
        "latency_ms_including_load": result.get("latency_ms"),
        "model_reused": result.get("model_reused"),
        "overlay_png": bool(overlay),
        "status": "api-smoke-passed",
        "warning": "Synthetic input checks connectivity, NOT model accuracy or warm-serving latency.",
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)
    try:
        result = smoke(args.url, make_png(args.image), timeout=args.timeout)
    except (urllib.error.URLError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Local API smoke failed: {exc}") from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
