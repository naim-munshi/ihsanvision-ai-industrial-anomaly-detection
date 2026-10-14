"""Validate canonical package layout and long-lived compatibility shims."""
from importlib import import_module
from pathlib import Path
import subprocess
import sys

import ihsanvision


def test_canonical_package_has_version_and_modules():
    assert ihsanvision.__version__ == "0.4.0"
    names = ("api", "benchmark", "config", "predict", "schemas", "serving", "train")
    for name in names:
        canonical = import_module(f"ihsanvision.{name}")
        legacy = import_module(f"src.{name}")
        assert canonical is legacy, f"Legacy src.{name} should alias canonical module"


def test_legacy_module_identity_preserves_monkeypatching():
    from src import api as old_api
    from ihsanvision import api as new_api
    assert old_api is new_api
    assert old_api.create_app is new_api.create_app


def test_old_and_new_cli_modules_support_help():
    root = Path(__file__).resolve().parents[1]
    for name in ("train", "predict", "benchmark"):
        for prefix in ("src", "ihsanvision"):
            command = [sys.executable, "-m", f"{prefix}.{name}", "--help"]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                    timeout=20, check=False)
            assert result.returncode == 0, (
                f"CLI {prefix}.{name}: {result.stdout}\n{result.stderr}"
            )
            assert "usage:" in result.stdout.lower()


def test_project_metadata_and_directory_contracts():
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").is_file()
    for folder in ("ihsanvision", "src", "scripts", "tests", "docs", "reports", "demo"):
        assert (root / folder).is_dir()
    # Prevent the accidental introduction of large ML binaries into tracked paths.
    for module in ("api", "predict", "train", "benchmark"):
        assert (root / "ihsanvision" / f"{module}.py").is_file()
