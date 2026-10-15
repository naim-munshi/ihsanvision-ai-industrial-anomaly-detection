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
