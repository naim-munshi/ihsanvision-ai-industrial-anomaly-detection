"""Static container configuration checks; not a runtime Docker build."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_uses_nonroot_and_temp_workdir():
    contents = (ROOT / "Dockerfile").read_text()
    assert "FROM python:3.12-slim-bookworm" in contents
    assert "USER 10001:10001" in contents
    assert "WORKDIR /tmp" in contents
    assert "HEALTHCHECK" in contents
    assert "COPY . /app" in contents
    assert "ihsanvision.api:app" in contents
    assert '"--workers","1"' in contents
    assert "model.ckpt" not in contents


def test_compose_is_local_only_with_external_readonly_artifacts():
    contents = (ROOT / "compose.yaml").read_text()
    assert '"127.0.0.1:8080:8080"' in contents
    assert "IHSANVISION_MODELS_DIR" in contents
    assert "IHSANVISION_API_KEY" in contents
    assert "target: /models" in contents
    assert "read_only: true" in contents
    assert 'user: "10001:10001"' in contents
    assert "no-new-privileges:true" in contents
    assert "cap_drop:" in contents and "ALL" in contents
    assert 'restart: "no"' in contents


def test_build_context_excludes_private_data_and_secrets():
    patterns = set((ROOT / ".dockerignore").read_text().splitlines())
    required = {".env", ".env.*", ".venv", "data", "datasets",
                "runs", "results", "*.ckpt", "*.pt", "*.bin", "*.zip", ".git"}
    assert required.issubset(patterns)
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in ignored and ".env.*" in ignored
    assert "!.env.docker.example" in ignored


def test_cloud_cpu_container_checks_real_ml_imports_without_weights():
    workflow = (ROOT / ".github/workflows/docker-cpu-smoke.yml").read_text()
    assert "Import the actual ML stack and initialize both model families" in workflow
    assert "import torch" in workflow
    assert "import torchvision" in workflow
    assert "import anomalib" in workflow
    assert "make_model(family, pre_trained=False)" in workflow
    assert 'for family in ("padim", "patchcore")' in workflow
    assert "--network none" in workflow
    assert "NO checkpoint inference." in workflow


def test_cpu_image_dependency_check_is_inside_docker_and_versions_are_saved():
    workflow = (ROOT / ".github/workflows/docker-cpu-smoke.yml").read_text()
    assert "ihsanvision-cpu-smoke:ci -m pip check" in workflow
    assert "ihsanvision-cpu-smoke:ci -m pip freeze" in workflow
    assert "uses: actions/upload-artifact@v4" in workflow
    assert "name: ihsanvision-cpu-dependencies" in workflow
    assert "retention-days: 14" in workflow
