"""Approval manifest and fail-closed checkpoint pinning tests; no ML weights loaded."""
from hashlib import sha256
from io import BytesIO
import json

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from ihsanvision.api import create_app
from ihsanvision.approvals import configured_approvals


OPERATOR_KEY = "test-private-operator-secret"
HEADERS = {"X-IhsanVision-Key": OPERATOR_KEY}


def png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (12, 12), "white").save(output, "PNG")
    return output.getvalue()


def fixture(monkeypatch, tmp_path, *, approval=True, history=False, second=False):
    for name in (
        "IHSANVISION_APPROVED_MODELS", "IHSANVISION_MODEL_REGISTRY",
        "IHSANVISION_CHECKPOINT", "IHSANVISION_PADIM_CHECKPOINT",
        "IHSANVISION_PATCHCORE_CHECKPOINT", "IHSANVISION_HISTORY_DB",
        "IHSANVISION_API_KEY", "IHSANVISION_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    a = tmp_path / "bottle-padim.ckpt"
    a.write_bytes(b"trusted-local-padim-fixture")
    b = tmp_path / "bottle-patchcore.ckpt"
    monkeypatch.setenv("IHSANVISION_PADIM_CHECKPOINT", str(a))
    if second:
        b.write_bytes(b"trusted-local-patchcore-fixture")
        monkeypatch.setenv("IHSANVISION_PATCHCORE_CHECKPOINT", str(b))
    if history:
        monkeypatch.setenv("IHSANVISION_HISTORY_DB", str(tmp_path / "private.sqlite3"))
    if approval or history:
        monkeypatch.setenv("IHSANVISION_API_KEY", OPERATOR_KEY)
    manifest = tmp_path / "approved.json"
    if approval:
        manifest.write_text(json.dumps({
            "version": 1,
            "approvals": [{
                "category": "bottle", "model": "padim",
                "checkpoint_sha256": sha256(a.read_bytes()).hexdigest(),
            }],
        }))
        monkeypatch.setenv("IHSANVISION_APPROVED_MODELS", str(manifest))
    return a, b, manifest


def post(client, model="padim", headers=None):
    return client.post(
        "/predict?model=" + model,
        headers=headers or HEADERS,
        files={"file": ("private-image.png", png(), "image/png")},
    )


def mock_predict(calls):
    def predict(_image, checkpoint, model):
        calls.append((checkpoint, model))
        return {
            "model": model, "anomaly_score": 0.61,
            "predicted_anomaly": False, "latency_ms": 4.2,
        }
    return predict


def test_valid_approval_only_runs_exact_pinned_bytes(monkeypatch, tmp_path):
    path, second, _ = fixture(monkeypatch, tmp_path, second=True)
    calls = []
    client = TestClient(create_app(predict_fn=mock_predict(calls)))
    models = client.get("/models").json()
    assert models["available_models"] == ["padim"]
    assert models["models_by_category"] == {"bottle": ["padim"]}
    assert client.get("/health").json()["model_configured"] is True

    response = post(client)
    assert response.status_code == 200
    assert response.json()["checkpoint_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert response.json()["inspection_id"] is None
    assert len(calls) == 1

    # Configured but not pinned models are not advertised and cannot execute.
    assert post(client, "patchcore").status_code == 503
    assert len(calls) == 1
    assert client.get("/models").json()["available_models"] == ["padim"]
    assert str(tmp_path) not in response.text


def test_replaced_bytes_fail_before_inference_and_recover_on_restore(monkeypatch, tmp_path):
    path, _, _ = fixture(monkeypatch, tmp_path)
    calls = []
    client = TestClient(create_app(predict_fn=mock_predict(calls)))
    assert post(client).status_code == 200
    path.write_bytes(b"unapproved-new-checkpoint")
    rejected = post(client)
    assert rejected.status_code == 503
    assert rejected.json()["detail"] == "Selected checkpoint version is not operator-approved"
    assert len(calls) == 1
    assert "unapproved-new-checkpoint" not in rejected.text
    path.write_bytes(b"trusted-local-padim-fixture")
    assert post(client).status_code == 200
    assert len(calls) == 2


def test_selected_checkpoint_switch_same_path_is_denied(monkeypatch, tmp_path):
    path, _, _ = fixture(monkeypatch, tmp_path)
    client = TestClient(create_app(predict_fn=mock_predict([])))
    replacement = tmp_path / "new.ckpt"
    replacement.write_bytes(b"new-unapproved-content")
    replacement.replace(path)
    assert post(client).status_code == 503


def test_reject_missing_key_and_incorrect_key(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    monkeypatch.delenv("IHSANVISION_API_KEY")
    with pytest.raises(ValueError, match="IHSANVISION_API_KEY"):
        create_app(predict_fn=mock_predict([]))
    monkeypatch.setenv("IHSANVISION_API_KEY", "short")
    with pytest.raises(ValueError, match="at least 16"):
        create_app(predict_fn=mock_predict([]))
    monkeypatch.setenv("IHSANVISION_API_KEY", OPERATOR_KEY)
    calls = []
    client = TestClient(create_app(predict_fn=mock_predict(calls)))
    assert client.post(
        "/predict",
        files={"file": ("a.png", png(), "image/png")},
    ).status_code == 401
    assert client.post(
        "/predict", headers={"X-IhsanVision-Key": "invalid"},
        files={"file": ("a.png", png(), "image/png")},
    ).status_code == 401
    assert calls == []


@pytest.mark.parametrize("replace", [
    {"version": 2, "approvals": []},
    {"version": True, "approvals": []},
    {"version": 1, "approvals": []},
    {"version": 1, "approvals": [{"category": "bottle", "model": "padim",
                                "checkpoint_sha256": "ab"}]},
    {"version": 1, "approvals": [{"category": "bottle", "model": "padim",
                                "checkpoint_sha256": "F" * 64}]},
    {"version": 1, "approvals": [{"category": "not_real", "model": "padim",
                                "checkpoint_sha256": "a" * 64}]},
    {"version": 1, "approvals": [{"category": "bottle", "model": "padim",
                                "checkpoint_sha256": "a" * 64},
                               {"category": "bottle", "model": "padim",
                                "checkpoint_sha256": "b" * 64}]},
    {"version": 1, "approvals": [{"category": "bottle", "model": "padim",
                                "checkpoint_sha256": "a" * 64, "checkpoint": "/private"}]},
])
def test_malformed_approvals_fail_closed(monkeypatch, tmp_path, replace):
    _, _, manifest = fixture(monkeypatch, tmp_path)
    manifest.write_text(json.dumps(replace))
    with pytest.raises(ValueError):
        configured_approvals()


def test_missing_symlink_relative_and_excessively_large_manifest_rejected(
    monkeypatch, tmp_path
):
    _, _, manifest = fixture(monkeypatch, tmp_path)
    manifest.unlink()
    with pytest.raises(ValueError, match="non-symlink"):
        configured_approvals()

    target = tmp_path / "manifest_target.json"
    target.write_text('{"version":1,"approvals":[]}')
    manifest.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        configured_approvals()
    manifest.unlink()
    monkeypatch.setenv("IHSANVISION_APPROVED_MODELS", "relative.json")
    with pytest.raises(ValueError, match="absolute"):
        configured_approvals()
    monkeypatch.setenv("IHSANVISION_APPROVED_MODELS", str(manifest))
    manifest.write_text("x" * 64001)
    with pytest.raises(ValueError, match="size"):
        configured_approvals()


def test_approvals_and_private_history_work_together(monkeypatch, tmp_path):
    path, _, _ = fixture(monkeypatch, tmp_path, history=True)
    calls = []
    client = TestClient(create_app(predict_fn=mock_predict(calls)))
    response = post(client)
    assert response.status_code == 200
    digest = sha256(path.read_bytes()).hexdigest()
    assert response.json()["checkpoint_sha256"] == digest
    saved = client.get("/inspections", headers=HEADERS).json()["records"]
    assert len(saved) == 1
    assert saved[0]["checkpoint_sha256"] == digest

    path.write_bytes(b"unapproved")
    assert post(client).status_code == 503
    assert client.get("/inspections", headers=HEADERS).json()["total"] == 1
    assert len(calls) == 1


def test_opt_in_off_preserves_legacy_unrestricted_demo(monkeypatch, tmp_path):
    path, second, _ = fixture(monkeypatch, tmp_path, approval=False, second=True)
    calls = []
    client = TestClient(create_app(predict_fn=mock_predict(calls)))
    assert set(client.get("/models").json()["available_models"]) == {"padim", "patchcore"}
    r = post(client)
    assert r.status_code == 200
    assert r.json()["checkpoint_sha256"] is None
    assert post(client, "patchcore").status_code == 200
    assert len(calls) == 2


def test_approvals_scope_model_and_category_not_only_family(monkeypatch, tmp_path):
    bottle, _, manifest = fixture(monkeypatch, tmp_path)
    cable = tmp_path / "cable-padim.ckpt"
    cable.write_bytes(b"independent-cable-model-fixture")
    registry = tmp_path / "model-registry.json"
    registry.write_text(json.dumps({
        "version": 1,
        "models": [{
            "category": "cable", "model": "padim",
            "checkpoint": str(cable),
        }],
    }))
    monkeypatch.setenv("IHSANVISION_MODEL_REGISTRY", str(registry))
    recorded = []
    first = TestClient(create_app(predict_fn=mock_predict(recorded)))
    assert first.get("/models").json()["available_categories"] == ["bottle"]
    assert first.post(
        "/predict?model=padim&category=cable", headers=HEADERS,
        files={"file": ("image.png", png(), "image/png")},
    ).status_code == 503
    assert recorded == []

    manifest.write_text(json.dumps({
        "version": 1,
        "approvals": [
            {"category": "bottle", "model": "padim",
             "checkpoint_sha256": sha256(bottle.read_bytes()).hexdigest()},
            {"category": "cable", "model": "padim",
             "checkpoint_sha256": sha256(cable.read_bytes()).hexdigest()},
        ],
    }))
    second = TestClient(create_app(predict_fn=mock_predict(recorded)))
    assert second.get("/models").json()["available_categories"] == ["bottle", "cable"]
    result = second.post(
        "/predict?model=padim&category=cable", headers=HEADERS,
        files={"file": ("image.png", png(), "image/png")},
    )
    assert result.status_code == 200
    assert result.json()["category"] == "cable"
    assert result.json()["checkpoint_sha256"] == sha256(cable.read_bytes()).hexdigest()
    assert recorded == [(cable.resolve(), "padim")]


def test_explicitly_empty_approval_env_never_silently_disables_pin_guard(
    monkeypatch, tmp_path
):
    fixture(monkeypatch, tmp_path)
    monkeypatch.setenv("IHSANVISION_APPROVED_MODELS", "   ")
    with pytest.raises(ValueError, match="set but empty"):
        create_app(predict_fn=mock_predict([]))
