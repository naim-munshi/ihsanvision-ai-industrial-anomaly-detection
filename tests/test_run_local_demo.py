"""Offline tests for the local one-command dashboard launcher."""
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

from scripts import run_local_demo as launcher


@pytest.fixture
def model_folder(tmp_path):
    for name in launcher.MODEL_NAMES:
        model = tmp_path / f"{name}-bottle-seed42" / "model.ckpt"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"private trusted checkpoint stub")
    return tmp_path


def test_both_checkpoints_are_detected_without_loading(model_folder):
    result = launcher.trusted_checkpoints(model_folder)
    assert set(result) == {"padim", "patchcore"}
    assert all(path.is_file() for path in result.values())


def test_missing_checkpoint_is_reported_without_fallback(model_folder):
    (model_folder / "patchcore-bottle-seed42" / "model.ckpt").unlink()
    with pytest.raises(ValueError, match="Patchcore|patchcore"):
        launcher.trusted_checkpoints(model_folder)


def test_empty_checkpoint_is_rejected(model_folder):
    (model_folder / "padim-bottle-seed42" / "model.ckpt").write_bytes(b"")
    with pytest.raises(ValueError, match="padim"):
        launcher.trusted_checkpoints(model_folder)


def test_prohibits_public_or_privileged_port():
    with pytest.raises(ValueError, match="1024"):
        launcher.ensure_local_port_available(80)


def test_reports_existing_port_instead_of_killing_process():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        port = occupied.getsockname()[1]
        with pytest.raises(ValueError, match="already occupied"):
            launcher.ensure_local_port_available(port)


def test_main_sets_both_paths_before_starting_loopback_uvicorn(
    model_folder, monkeypatch
):
    launched = {}

    def fake_run(application, **settings):
        launched["application"] = application
        launched.update(settings)
        assert os.environ["IHSANVISION_PADIM_CHECKPOINT"] == str(
            model_folder / "padim-bottle-seed42" / "model.ckpt"
        )
        assert os.environ["IHSANVISION_PATCHCORE_CHECKPOINT"] == str(
            model_folder / "patchcore-bottle-seed42" / "model.ckpt"
        )
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))
    for variable in ("IHSANVISION_PADIM_CHECKPOINT", "IHSANVISION_PATCHCORE_CHECKPOINT",
                     "IHSANVISION_MODEL"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("IHSANVISION_CHECKPOINT", "/old/legacy.ckpt")
    launcher.main([
        "--artifacts-dir", str(model_folder),
        "--port", "8777", "--no-browser",
    ])

    assert "IHSANVISION_CHECKPOINT" not in os.environ
    assert launched == {
        "application": "ihsanvision.api:app",
        "host": "127.0.0.1",
        "port": 8777,
        "workers": 1,
        "reload": False,
    }


def test_invalid_port_exits_before_server_launch(model_folder):
    with pytest.raises(SystemExit) as exc:
        launcher.main([
            "--artifacts-dir", str(model_folder),
            "--port", "80", "--no-browser",
        ])
    assert exc.value.code == 2


def test_browser_opens_after_healthy_local_server(monkeypatch):
    opened = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def probe(url, timeout):
        assert url == "http://127.0.0.1:8003/health"
        assert timeout == 1
        return FakeResponse()

    monkeypatch.setattr(launcher, "urlopen", probe)
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    launcher._open_browser_when_ready(8003)
    assert opened == ["http://127.0.0.1:8003"]
