from app.auth.security import (
    hash_password,
    hash_token,
    needs_rehash,
    new_session_token,
    verify_password,
)
from app.config import Settings

CHEAP = Settings(
    argon2_time_cost=1, argon2_memory_kib=1024, argon2_parallelism=1, _env_file=None
)


def test_hash_and_verify_roundtrip():
    stored = hash_password(CHEAP, "a long passphrase")

    assert stored.startswith("$argon2id$")
    assert "a long passphrase" not in stored
    assert verify_password(CHEAP, stored, "a long passphrase") is True
    assert verify_password(CHEAP, stored, "a different passphrase") is False


def test_malformed_stored_hashes_never_verify():
    assert verify_password(CHEAP, "not-a-hash", "anything") is False
    assert verify_password(CHEAP, "", "anything") is False


def test_each_hash_uses_a_fresh_salt():
    assert hash_password(CHEAP, "same password") != hash_password(CHEAP, "same password")


def test_stronger_settings_trigger_a_rehash():
    stored = hash_password(CHEAP, "pw-for-rehash")
    stronger = CHEAP.model_copy(update={"argon2_time_cost": 2})

    assert needs_rehash(CHEAP, stored) is False
    assert needs_rehash(stronger, stored) is True
    assert verify_password(stronger, stored, "pw-for-rehash") is True  # parameters live in the hash


def test_session_tokens_are_long_random_and_stored_only_as_hashes():
    token, stored = new_session_token()
    other_token, _ = new_session_token()

    assert len(token) >= 43 and token != other_token
    assert stored == hash_token(token)
    assert len(stored) == 64 and stored != token