import re
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy import select, update

from app.auth.security import hash_token
from app.config import get_settings
from app.database import get_db
from app.main import app
from app.models import AuthSession, User

PASSWORD = "correct-horse-battery"
PUBLIC_ROUTES = {("GET", "/health"), ("POST", "/auth/register"), ("POST", "/auth/login")}


@pytest.fixture
def anon(client):
    """A client with no credentials (the `client` fixture installs the test overrides)."""
    return TestClient(app)


def register(who, email="ann@example.com", password=PASSWORD):
    return who.post("/auth/register", json={"email": email, "password": password})


def login(who, email="ann@example.com", password=PASSWORD):
    return who.post("/auth/login", json={"email": email, "password": password})


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_register_returns_a_token_and_a_profile(anon):
    response = register(anon)
    body = response.json()

    assert response.status_code == 201
    assert body["token_type"] == "bearer" and len(body["access_token"]) >= 43
    assert body["user"]["email"] == "ann@example.com"
    me = anon.get("/auth/me", headers=bearer(body["access_token"]))
    assert me.status_code == 200
    assert set(me.json()) == {"id", "email", "is_admin"}  # no password data


def test_emails_are_trimmed_and_case_insensitive(anon):
    assert register(anon, "  Ann@Example.COM ").json()["user"]["email"] == "ann@example.com"
    assert login(anon, "ANN@example.com").status_code == 200


def test_duplicate_email_is_rejected_regardless_of_case(anon):
    register(anon)

    assert register(anon, "ANN@example.com").status_code == 409


def test_short_password_is_rejected(anon):
    assert register(anon, password="short").status_code == 422


def test_invalid_email_is_rejected(anon):
    assert register(anon, "not-an-email").status_code == 422
    assert register(anon, "a b@example.com").status_code == 422


def test_unknown_email_and_wrong_password_look_identical(anon):
    register(anon)
    wrong = login(anon, password="a-wrong-password")
    unknown = login(anon, "nobody@example.com", "a-wrong-password")

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_logout_ends_only_that_session(anon):
    register(anon)
    first = login(anon).json()["access_token"]
    second = login(anon).json()["access_token"]
    assert first != second

    assert anon.post("/auth/logout", headers=bearer(first)).status_code == 204
    assert anon.get("/auth/me", headers=bearer(first)).status_code == 401
    assert anon.get("/auth/me", headers=bearer(second)).status_code == 200


def test_an_expired_session_is_rejected(anon):
    token = register(anon).json()["access_token"]
    db = next(app.dependency_overrides[get_db]())
    db.execute(update(AuthSession).values(expires_at=datetime.now(timezone.utc) - timedelta(minutes=1)))
    db.commit()

    assert anon.get("/auth/me", headers=bearer(token)).status_code == 401


def test_bad_credentials_are_rejected(anon):
    assert anon.get("/auth/me").status_code == 401
    assert anon.get("/auth/me", headers=bearer("nonsense")).status_code == 401
    assert anon.get("/auth/me", headers={"Authorization": "Basic YWJjOmRlZg=="}).status_code == 401
    assert anon.get("/auth/me", headers={"Authorization": "Bearer "}).status_code == 401


def test_passwords_and_tokens_are_stored_hashed(anon):
    token = register(anon).json()["access_token"]
    db = next(app.dependency_overrides[get_db]())
    user = db.scalar(select(User).where(User.email == "ann@example.com"))

    assert user.password_hash.startswith("$argon2id$") and PASSWORD not in user.password_hash
    assert db.scalar(select(AuthSession).where(AuthSession.token_hash == hash_token(token))) is not None
    assert db.scalar(select(AuthSession).where(AuthSession.token_hash == token)) is None


def test_registration_can_be_switched_off(anon):
    base = app.dependency_overrides[get_settings]()
    app.dependency_overrides[get_settings] = lambda: base.model_copy(update={"allow_registration": False})

    assert register(anon, "late@example.com").status_code == 403


def test_only_configured_emails_are_admins(client, anon):
    assert client.get("/auth/me").json()["is_admin"] is True  # admin@example.com
    token = register(anon).json()["access_token"]

    assert anon.get("/auth/me", headers=bearer(token)).json()["is_admin"] is False


def test_cors_preflight_allows_the_authorization_header(anon):
    response = anon.options(
        "/documents",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )

    assert response.status_code == 200
    assert "authorization" in response.headers["access-control-allow-headers"].lower()


def test_every_route_requires_authentication_except_the_public_ones(anon):
    checked = 0
    for path, operations in app.openapi()["paths"].items():
        for method in operations:
            method = method.upper()
            if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                continue
            if (method, path) in PUBLIC_ROUTES:
                continue
            response = anon.request(method, re.sub(r"\{[^}]+\}", "1", path))
            assert response.status_code == 401, (
                f"{method} {path} answered {response.status_code} without credentials"
            )
            checked += 1
    assert checked >= 15  # a new route that forgets authentication fails above