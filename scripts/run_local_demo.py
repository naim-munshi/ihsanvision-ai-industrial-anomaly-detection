"""Start the local dual-model IhsanVision dashboard with one command.

Use only checkpoints that you personally trained or independently trust. Never
expose this prototype to the public internet.

    python -m scripts.run_local_demo
    python -m scripts.run_local_demo --artifacts-dir /path/to/private/runs --port 8004
    python -m scripts.run_local_demo --no-browser

This command does not train models, run benchmarks, modify checkpoints or
create output artifacts.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


MODEL_NAMES = ("padim", "patchcore")
DEFAULT_ARTIFACTS = (
    Path.home() / "ihsanvision-artifacts" / "benchmark-bottle-seed42-2026-10-15"
)


def trusted_checkpoints(artifacts_dir: Path) -> dict[str, Path]:
    """Locate both trusted files without deserializing their contents."""
    folder = artifacts_dir.expanduser().resolve()
    paths = {
        name: folder / f"{name}-bottle-seed42" / "model.ckpt"
        for name in MODEL_NAMES
    }
    missing = [
        f"{name}: {path}"
        for name, path in paths.items()
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise ValueError(
            "Both trusted model checkpoints are required. Missing or empty:\n"
            + "\n".join(missing)
            + "\nPass --artifacts-dir to the folder containing the model subfolders."
        )
    return paths


def ensure_local_port_available(port: int) -> None:
    if not 1024 <= port <= 65535:
        raise ValueError("Choose an unprivileged localhost port from 1024 to 65535")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as check:
        try:
            check.bind(("127.0.0.1", port))
        except OSError:
            raise ValueError(
                f"Port {port} is already occupied. Use --port 8004, "
                "or stop only the server you started yourself."
            ) from None


def configure_local_models(paths: dict[str, Path]) -> None:
    """Select the two explicitly trusted Bottle checkpoints for this demo.

    The multi-category registry takes precedence over per-model environment
    variables for the same category. Fail closed instead of silently serving
    a different checkpoint when a previous shell session left a registry set.
    Full multi-category operators should launch the API with their explicitly
    reviewed registry, not the fixed two-model Bottle convenience launcher.
    """
    registry = os.environ.get("IHSANVISION_MODEL_REGISTRY", "").strip()
    if registry:
        raise ValueError(
            "IHSANVISION_MODEL_REGISTRY is set and can override Bottle weights. "
            "Unset it for the fixed two-checkpoint local demo, or launch the "
            "API directly with your reviewed registry configuration."
        )
    os.environ.pop("IHSANVISION_CHECKPOINT", None)
    for name in MODEL_NAMES:
        os.environ[f"IHSANVISION_{name.upper()}_CHECKPOINT"] = str(paths[name])
    os.environ["IHSANVISION_MODEL"] = "padim"


def _open_browser_when_ready(port: int) -> None:
    """Open only after our localhost API has finished starting."""
    target = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urlopen(target + "/health", timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(target)
                    return
        except (OSError, URLError, ValueError):
            time.sleep(0.25)
    print(f"Browser did not open automatically; visit {target}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS,
        help="Private folder containing padim-bottle-seed42/ and patchcore-bottle-seed42/",
    )
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Start the server without opening a browser tab",
    )
    args = parser.parse_args(argv)

    try:
        paths = trusted_checkpoints(args.artifacts_dir)
        ensure_local_port_available(args.port)
        configure_local_models(paths)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    try:
        import uvicorn
    except ImportError:
        parser.error("Uvicorn is missing. Install the project's runtime requirements.")

    base = f"http://127.0.0.1:{args.port}"
    print(f"IhsanVision.AI local dashboard: {base}", flush=True)
    print("Models: PaDiM and PatchCore (trusted local checkpoints only)", flush=True)
    print("Press Ctrl+C to stop the server. No training is performed.", flush=True)
    if os.getenv("IHSANVISION_API_KEY"):
        print("API-key protection is enabled through your existing environment.", flush=True)

    if not args.no_browser:
        threading.Thread(
            target=_open_browser_when_ready, args=(args.port,), daemon=True
        ).start()

    # Keep the HTTP application bound to loopback. Import and model
    # construction are deferred until after the environment is configured.
    uvicorn.run(
        "ihsanvision.api:app",
        host="127.0.0.1",
        port=args.port,
        workers=1,
        reload=False,
    )


if __name__ == "__main__":
    main()
