"""Real SQLite/session HTTP tests; prediction itself is explicitly stubbed."""
from contextlib import closing
import sqlite3
import time

from fastapi.testclient import TestClient
import pytest

from ihsanvision.api import create_app
from ihsanvision.auth import Accounts, COOKIE
from tests.test_inspection_history import configure, image_bytes, predictor

PASSWORD = "test-only-operator-password"
ORIGIN = "http://127.0.0.1:8004"


@pytest.fixture
def setup(monkeypatch, tmp_path):
    history = configure(monkeypatch, tmp_path)
    path = tmp_path / "accounts.sqlite3"
    monkeypatch.setenv("IHSANVISION_AUTH_DB", str(path))
    monkeypatch.setenv("IHSANVISION_ORIGIN", ORIGIN)
    accounts = Accounts(path)
    accounts.add_user("operator-one", PASSWORD, "operator")
    accounts.add_user("admin-one", PASSWORD, "admin")
    client = TestClient(create_app(predict_fn=predictor), base_url=ORIGIN)
    return client, accounts, history


def login(client, username="operator-one", password=PASSWORD):
    return client.post("/auth/login", json={"username": username, "password": password},
                       headers={"X-IhsanVision-Login": "1", "Origin": ORIGIN})


def csrf(client):
    return {"X-IhsanVision-CSRF": client.get("/auth/me").json()["user"]["csrf"]}


def inspect(client):
    return client.post("/predict", headers=csrf(client), files={"file": ("image.png", image_bytes(), "image/png")})


def test_cookie_auth_prediction_review_audit_and_roles(setup):
    client, accounts, history = setup
    assert client.get("/inspections").status_code == 401
    response = login(client)
    assert response.status_code == 200
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert "max-age=28800" in cookie
    result = inspect(client)
    assert result.status_code == 200, result.text
    record_id = result.json()["inspection_id"]
    assert client.patch(f"/inspections/{record_id}/review", json={"verdict": "confirmed_normal"}, headers=csrf(client)).status_code == 200
    with closing(sqlite3.connect(history)) as db:
        assert db.execute("SELECT actor,verdict FROM inspection_reviews").fetchone() == ("operator-one", "confirmed_normal")
    for path in ("/inspections", "/inspection-lots", "/inspection-model-versions", "/inspections/export.csv"):
        assert client.get(path).status_code == 200
    assert client.delete(f"/inspections/{record_id}", headers=csrf(client)).status_code == 403
    assert login(client, "admin-one").status_code == 200
    assert client.delete(f"/inspections/{record_id}", headers=csrf(client)).status_code == 200


def test_csrf_login_guard_and_host(setup):
    client, _, _ = setup
    assert client.post("/auth/login", json={}).status_code == 403
    assert client.post("/auth/login", json={}, headers={"X-IhsanVision-Login":"1", "Origin":"https://evil.example"}).status_code == 403
    assert client.get("/auth/me", headers={"Host":"evil.example"}).status_code == 400
    assert login(client).status_code == 200
    assert client.post("/predict", content=b"not multipart").status_code == 403
    assert client.post("/auth/logout").status_code == 403
    assert client.post("/auth/logout", headers={**csrf(client), "Origin":"https://evil.example"}).status_code == 403
    assert client.get("/inspections").status_code == 200


def test_logout_revokes_copied_cookie_and_restart_preserves_session(setup):
    client, _, _ = setup
    login(client)
    token = client.cookies.get(COOKIE)
    restarted = TestClient(create_app(predict_fn=predictor), base_url=ORIGIN)
    restarted.cookies.set(COOKIE, token)
    assert restarted.get("/inspections").status_code == 200
    assert client.post("/auth/logout", headers=csrf(client)).status_code == 200
    assert restarted.get("/inspections").status_code == 401


def test_expired_and_disabled_sessions(setup):
    client, accounts, _ = setup
    login(client)
    with closing(accounts.connect()) as db, db:
        db.execute("UPDATE sessions SET expires=?", (time.time()-1,))
    assert client.get("/inspections").status_code == 401
    login(client)
    accounts.disable("operator-one")
    assert client.get("/inspections").status_code == 401
    assert login(client).status_code == 401


def test_wrong_password_uniform_error_budget_persists(setup):
    client, accounts, _ = setup
    missing = login(client, "missing-user")
    wrong = login(client, password="incorrect")
    assert missing.status_code == wrong.status_code == 401
    assert missing.json() == wrong.json()
    with closing(accounts.connect()) as db, db:
        db.execute("UPDATE login_budget SET attempts=10")
    restarted = TestClient(create_app(predict_fn=predictor), base_url=ORIGIN)
    assert login(restarted).status_code == 429


def test_private_storage_hashes_and_duplicate_user_preserves_original(setup):
    client, accounts, _ = setup
    login(client)
    assert accounts.path.stat().st_mode & 0o077 == 0
    with closing(accounts.connect()) as db:
        salt, digest = db.execute("SELECT salt,digest FROM users WHERE username='operator-one'").fetchone()
        stored = db.execute("SELECT digest FROM sessions").fetchone()[0]
    assert len(salt) == 16 and len(digest) == 32
    assert PASSWORD.encode() not in accounts.path.read_bytes()
    assert stored != client.cookies.get(COOKIE)
    with pytest.raises(sqlite3.IntegrityError):
        accounts.add_user("operator-one", "other-test-password", "admin")
    assert login(client).status_code == 200
    assert client.get("/auth/me").json()["user"]["role"] == "operator"


def test_key_scripts_still_work_and_invalid_key_cannot_fall_back(setup):
    client, _, _ = setup
    key = {"X-IhsanVision-Key": "private-test-operator-secret"}
    assert client.get("/inspections", headers=key).status_code == 200
    assert client.post("/predict", headers=key, files={"file": ("image.png", image_bytes(), "image/png")}).status_code == 200
    login(client)
    assert client.get("/inspections", headers={"X-IhsanVision-Key":"wrong"}).status_code == 401
    assert client.get("/inspections", headers=[("X-IhsanVision-Key","wrong"),("X-IhsanVision-Key","wrong")]).status_code == 400


def test_login_body_bounds_and_no_store(setup):
    client, _, _ = setup
    assert client.post("/auth/login", content=b"x"*4097, headers={"X-IhsanVision-Login":"1"}).status_code == 413
    assert client.post("/auth/login", content=b"not json", headers={"X-IhsanVision-Login":"1"}).status_code == 400
    login(client)
    assert "no-store" in client.get("/auth/me").headers["cache-control"]
    assert client.get("/login").status_code == 200
    assert client.get("/auth-ui.js").status_code == 200


def test_tls_cookie_and_invalid_remote_http_origin(setup, monkeypatch):
    _, _, _ = setup
    monkeypatch.setenv("IHSANVISION_ORIGIN", "https://vision.example")
    client = TestClient(create_app(predict_fn=predictor), base_url="https://vision.example")
    response = client.post("/auth/login", json={"username":"operator-one", "password":PASSWORD}, headers={"X-IhsanVision-Login":"1"})
    assert response.status_code == 200
    assert "Secure" in response.headers["set-cookie"]
    monkeypatch.setenv("IHSANVISION_ORIGIN", "http://vision.example")
    with pytest.raises(ValueError, match="HTTPS"):
        create_app(predict_fn=predictor)


def test_real_server_restart_keeps_session_and_history(setup):
    """Start/stop only disposable cloud subprocesses; no trained inference claim."""
    import httpx
    import os
    import socket
    import subprocess
    import sys
    from ihsanvision.history import InspectionHistory

    _, accounts, history_path = setup
    history = InspectionHistory(history_path)
    record_id = history.record("bottle", "padim", {
        "anomaly_score": 0.5, "predicted_anomaly": True,
        "checkpoint_sha256": "a" * 64,
    }, "AUTH-RESTART-TEST")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    env = {**os.environ, "IHSANVISION_ORIGIN": origin}

    def start():
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "ihsanvision.api:app", "--host", "127.0.0.1", "--port", str(port), "--no-access-log"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise AssertionError("Disposable server stopped before readiness")
                try:
                    if httpx.get(origin + "/health", timeout=0.2, trust_env=False).status_code == 200:
                        return process
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
            raise AssertionError("Disposable server did not become ready")
        except BaseException:
            process.terminate()
            process.wait(timeout=10)
            raise

    process = start()
    try:
        with httpx.Client(base_url=origin, trust_env=False) as client:
            response = client.post("/auth/login", json={"username":"operator-one", "password":PASSWORD}, headers={"X-IhsanVision-Login":"1"})
            assert response.status_code == 200
            process.terminate()
            process.wait(timeout=10)
            process = start()
            records = client.get("/inspections").json()
            assert any(record["id"] == record_id for record in records["records"])
            assert client.post("/auth/logout", headers=csrf(client)).status_code == 200
            assert client.get("/inspections").status_code == 401
    finally:
        process.terminate()
        process.wait(timeout=10)
