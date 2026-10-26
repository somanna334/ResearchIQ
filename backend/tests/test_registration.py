from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.main import app

PASSWORD = "correct-horse-battery"


def register(email: str, **extra):
    return TestClient(app).post(
        "/auth/register", json={"email": email, "password": PASSWORD, **extra}
    )


def test_registration_can_be_closed(client, tune):
    tune(allow_registration=False)

    assert register("new@example.com").status_code == 403


def test_an_invite_code_is_required_when_one_is_configured(client, tune):
    tune(invite_code=SecretStr("open-sesame-42"))

    missing = register("new@example.com")
    wrong = register("new@example.com", invite_code="open-sesame-43")
    right = register("new@example.com", invite_code="open-sesame-42")

    assert missing.status_code == wrong.status_code == 403
    assert missing.json() == wrong.json()  # no hint about what was wrong
    assert right.status_code == 201


def test_a_wrong_invite_code_does_not_reveal_which_emails_exist(client, tune):
    tune(invite_code=SecretStr("open-sesame-42"))

    response = register("admin@example.com", invite_code="nope")  # already registered

    assert response.status_code == 403  # not 409


def test_existing_users_sign_in_without_any_code(client, tune):
    tune(invite_code=SecretStr("open-sesame-42"))

    response = TestClient(app).post(
        "/auth/login", json={"email": "admin@example.com", "password": PASSWORD}
    )

    assert response.status_code == 200
