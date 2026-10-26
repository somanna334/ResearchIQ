import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.observability import (
    ACCESS_LOGGER,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)


def make_app(hsts: bool = False) -> TestClient:
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    app.add_middleware(SecurityHeadersMiddleware, enable_hsts=hsts)
    app.add_middleware(RequestContextMiddleware)
    return TestClient(app)


def test_generates_request_id():
    r = make_app().get("/ping")
    assert len(r.headers["X-Request-ID"]) == 32


def test_echoes_safe_request_id():
    r = make_app().get("/ping", headers={"X-Request-ID": "abc-12345-xyz"})
    assert r.headers["X-Request-ID"] == "abc-12345-xyz"


def test_replaces_unsafe_request_id():
    r = make_app().get("/ping", headers={"X-Request-ID": "bad id with spaces!"})
    assert r.headers["X-Request-ID"] != "bad id with spaces!"
    assert len(r.headers["X-Request-ID"]) == 32


def test_security_headers_present():
    r = make_app().get("/ping")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Referrer-Policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]


def test_hsts_only_when_enabled():
    assert "Strict-Transport-Security" not in make_app(hsts=False).get("/ping").headers
    assert "Strict-Transport-Security" in make_app(hsts=True).get("/ping").headers


def test_access_log_has_path_but_no_query(caplog):
    with caplog.at_level(logging.INFO, logger=ACCESS_LOGGER):
        make_app().get("/ping?token=SUPERSECRET")
    records = [r for r in caplog.records if r.name == ACCESS_LOGGER]
    assert records
    fields = records[-1].fields
    assert fields["path"] == "/ping"
    assert fields["status"] == 200
    assert "SUPERSECRET" not in str(fields)
