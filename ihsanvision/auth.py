"""Opt-in local accounts and revocable, opaque browser sessions.

Single-site shared inspection ledger, not tenant isolation or an identity provider.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import time
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from starlette.requests import Request as ASGIRequest

COOKIE = "ihsanvision_session"
TTL = 8 * 60 * 60
USERNAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}\Z")


def password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=32768, r=8, p=3,
                          maxmem=128 * 1024 * 1024, dklen=32)


class Accounts:
    def __init__(self, path: Path):
        path = Path(path).expanduser()
        if not path.is_absolute() or not path.parent.is_dir() or path.parent.is_symlink():
            raise ValueError("Account DB needs an absolute path in an existing private directory")
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("Account DB must be a regular file")
        if path.exists():
            if stat.S_IMODE(path.stat().st_mode) & 0o077:
                raise ValueError("Account DB must have mode 600")
        else:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
            os.close(fd)
        self.path = path
        with closing(self.connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY, salt BLOB NOT NULL, digest BLOB NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin','operator')), enabled INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (
                    digest TEXT PRIMARY KEY, username TEXT NOT NULL,
                    csrf TEXT NOT NULL, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS login_budget (
                    id INTEGER PRIMARY KEY CHECK(id=1), start REAL NOT NULL, attempts INTEGER NOT NULL);
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def add_user(self, username: str, password: str, role: str):
        if username == "api-key" or not USERNAME.fullmatch(username) or role not in {"admin", "operator"}:
            raise ValueError("Invalid username or role")
        if not 12 <= len(password) <= 128:
            raise ValueError("Password must contain 12-128 characters")
        salt = secrets.token_bytes(16)
        digest = password_hash(password, salt)
        with closing(self.connect()) as db, db:
            # INSERT deliberately rejects duplicates; never overwrite an existing account.
            db.execute("INSERT INTO users VALUES (?,?,?,?,1)", (username, salt, digest, role))

    def disable(self, username: str):
        with closing(self.connect()) as db, db:
            db.execute("UPDATE users SET enabled=0 WHERE username=?", (username,))
            db.execute("DELETE FROM sessions WHERE username=?", (username,))

    def login(self, username: str, password: str):
        now = time.time()
        with closing(self.connect()) as db, db:
            # Persisted global budget bounds hashing and guessing across workers/restarts.
            db.execute("BEGIN IMMEDIATE")
            budget = db.execute("SELECT start,attempts FROM login_budget WHERE id=1").fetchone()
            if budget and now - budget[0] < 60 and budget[1] >= 10:
                raise HTTPException(429, "Login attempt limit reached", headers={"Retry-After": "60"})
            start, attempts = (budget[0], budget[1] + 1) if budget and now - budget[0] < 60 else (now, 1)
            db.execute("INSERT OR REPLACE INTO login_budget VALUES (1,?,?)", (start, attempts))
            user = db.execute("SELECT salt,digest,role,enabled FROM users WHERE username=?", (username,)).fetchone()
        salt = user[0] if user else b"\0" * 16
        digest = password_hash(password, salt)
        if not user or not hmac.compare_digest(digest, user[1]) or not user[3]:
            raise HTTPException(401, "Invalid username or password")
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        with closing(self.connect()) as db, db:
            db.execute("DELETE FROM sessions WHERE expires <= ?", (now,))
            # Bound retained sessions per account, and invalidate earlier browser logins.
            db.execute("DELETE FROM sessions WHERE username=?", (username,))
            db.execute("INSERT INTO sessions VALUES (?,?,?,?)", (self.token_hash(token), username, csrf, now + TTL))
        return token

    @staticmethod
    def token_hash(token: str):
        return hashlib.sha256(token.encode()).hexdigest()

    def session(self, token: str):
        if not token or len(token) > 128:
            return None
        with closing(self.connect()) as db:
            row = db.execute("""SELECT u.username,u.role,s.csrf FROM sessions s JOIN users u
                ON s.username=u.username WHERE s.digest=? AND s.expires>? AND u.enabled=1""",
                (self.token_hash(token), time.time())).fetchone()
        return dict(zip(("username", "role", "csrf"), row)) if row else None

    def logout(self, token: str):
        with closing(self.connect()) as db, db:
            db.execute("DELETE FROM sessions WHERE digest=?", (self.token_hash(token),))


class SessionGuard:
    def __init__(self, app, accounts: Accounts, origin: str):
        self.app, self.accounts, self.origin = app, accounts, origin
        self.host = urlsplit(origin).netloc

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = ASGIRequest(scope)
        path, method = scope["path"], scope["method"]
        async def reject(code, message):
            await JSONResponse({"detail": message}, status_code=code,
                               headers={"Cache-Control": "no-store"})(scope, receive, send)
        if request.headers.get("host", "").lower() != self.host.lower():
            return await reject(400, "Unexpected host")
        protected = path in {"/predict", "/operator-metrics", "/inspection-lots", "/inspection-model-versions"} or path.startswith("/inspections")
        modifying = method not in {"GET", "HEAD", "OPTIONS"}
        keys = request.headers.getlist("x-ihsanvision-key")
        if protected and len(keys) > 1:
            return await reject(400, "Duplicate authentication headers")
        key = keys[0] if keys else ""
        actor = None
        if protected and key:
            configured = os.getenv("IHSANVISION_API_KEY", "")
            if not configured or not hmac.compare_digest(key.encode(), configured.encode()):
                return await reject(401, "Valid API key required")
            actor = {"username": "api-key", "role": "admin"}
        else:
            actor = await run_in_threadpool(self.accounts.session, request.cookies.get(COOKIE, ""))
            if protected and not actor:
                return await reject(401, "Login required")
            if modifying and (protected or path in {"/auth/login", "/auth/logout"}):
                if request.headers.get("origin") not in {None, self.origin}:
                    return await reject(403, "Same-origin request required")
                if path == "/auth/login":
                    if request.headers.get("x-ihsanvision-login") != "1":
                        return await reject(403, "Login request header required")
                elif not actor or not hmac.compare_digest(
                    request.headers.get("x-ihsanvision-csrf", "").encode(), actor["csrf"].encode()
                ):
                    return await reject(403, "Valid CSRF token required")
        if protected and method == "DELETE" and actor["role"] != "admin":
            return await reject(403, "Administrator permission required")
        scope.setdefault("state", {})["principal"] = actor
        # Login JSON is bounded before FastAPI parses it, including chunked requests.
        if path == "/auth/login" and method == "POST":
            parts, total = [], 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                part = message.get("body", b"")
                total += len(part)
                if total > 4096:
                    return await reject(413, "Login request too large")
                parts.append(part)
                if not message.get("more_body", False):
                    break
            delivered = False
            original_receive = receive
            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": b"".join(parts), "more_body": False}
                return await original_receive()
            receive = replay
        async def private_send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend([(b"cache-control", b"no-store"), (b"x-frame-options", b"DENY"),
                                (b"x-content-type-options", b"nosniff"), (b"referrer-policy", b"same-origin")])
                message = {**message, "headers": headers}
            await send(message)
        await self.app(scope, receive, private_send)


def install_auth(app):
    path = os.getenv("IHSANVISION_AUTH_DB", "").strip()
    accounts = Accounts(Path(path)) if path else None
    origin = os.getenv("IHSANVISION_ORIGIN", "").rstrip("/")
    if accounts:
        parsed = urlsplit(origin)
        if (parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path
                or parsed.query or parsed.fragment or parsed.username or parsed.password
                or (parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"})):
            raise ValueError("IHSANVISION_ORIGIN must be an exact HTTPS origin or HTTP loopback origin")
        app.add_middleware(SessionGuard, accounts=accounts, origin=origin)
    router = APIRouter()

    @router.get("/auth/me")
    def me(request: Request):
        actor = getattr(request.state, "principal", None)
        return {"enabled": accounts is not None, "user": actor}

    @router.get("/login")
    def login_page():
        return FileResponse(Path(__file__).resolve().parents[1] / "demo" / "login.html")

    @router.get("/auth-ui.js")
    def auth_script():
        return FileResponse(Path(__file__).resolve().parents[1] / "demo" / "auth.js", media_type="application/javascript")

    @router.post("/auth/login")
    async def login(request: Request):
        if not accounts:
            raise HTTPException(503, "Local accounts are disabled")
        try:
            data = await request.json()
        except ValueError:
            raise HTTPException(400, "Invalid login request") from None
        if not isinstance(data, dict):
            raise HTTPException(400, "Invalid login request")
        username, password = data.get("username"), data.get("password")
        if (not isinstance(username, str) or not USERNAME.fullmatch(username)
                or not isinstance(password, str) or not 1 <= len(password) <= 128):
            raise HTTPException(401, "Invalid username or password")
        token = await run_in_threadpool(accounts.login, username, password)
        response = JSONResponse({"authenticated": True})
        response.set_cookie(COOKIE, token, max_age=TTL, httponly=True,
                            secure=origin.startswith("https:"), samesite="strict", path="/")
        return response

    @router.post("/auth/logout")
    async def logout(request: Request):
        if not accounts:
            raise HTTPException(503, "Local accounts are disabled")
        await run_in_threadpool(accounts.logout, request.cookies.get(COOKIE, ""))
        response = JSONResponse({"authenticated": False})
        response.delete_cookie(COOKIE, path="/", secure=origin.startswith("https:"), httponly=True, samesite="strict")
        return response

    app.include_router(router)
