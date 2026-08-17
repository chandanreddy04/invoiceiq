"""
Tests for the stdlib-only auth module. Password hashing and session
tokens are the kind of code where a subtle bug (e.g. comparing hashes
with `==` instead of a constant-time compare) is a real vulnerability,
not just a wrong answer - worth testing deliberately.
"""

import time

from app.security.auth import (
    hash_password, verify_password, create_session_token, verify_session_token,
)


def test_hash_password_is_not_plaintext():
    h = hash_password("correct horse battery staple")
    assert "correct horse battery staple" not in h


def test_verify_password_accepts_correct_password():
    h = hash_password("my-secret-password")
    assert verify_password("my-secret-password", h) is True


def test_verify_password_rejects_wrong_password():
    h = hash_password("my-secret-password")
    assert verify_password("wrong-password", h) is False


def test_same_password_hashes_differently_each_time():
    # different random salt each call - protects against rainbow tables
    # and reveals nothing if two users happen to pick the same password
    h1 = hash_password("password123")
    h2 = hash_password("password123")
    assert h1 != h2
    assert verify_password("password123", h1)
    assert verify_password("password123", h2)


def test_verify_password_rejects_malformed_hash():
    assert verify_password("anything", "not-a-valid-hash-format") is False


def test_session_token_round_trips_correct_user_id():
    token = create_session_token(user_id=42)
    assert verify_session_token(token) == 42


def test_session_token_rejects_tampered_user_id():
    token = create_session_token(user_id=1)
    user_part, expiry_part, sig_part = token.split(".")
    tampered = f"999.{expiry_part}.{sig_part}"  # try to impersonate user 999
    assert verify_session_token(tampered) is None


def test_session_token_rejects_tampered_expiry():
    token = create_session_token(user_id=1)
    user_part, expiry_part, sig_part = token.split(".")
    far_future = int(expiry_part) + 10_000_000
    tampered = f"{user_part}.{far_future}.{sig_part}"  # try to extend own session
    assert verify_session_token(tampered) is None


def test_session_token_rejects_garbage():
    assert verify_session_token("not.a.valid.token.at.all") is None
    assert verify_session_token("") is None


def test_expired_session_token_is_rejected(monkeypatch):
    import app.security.auth as auth_module
    monkeypatch.setattr(auth_module, "SESSION_MAX_AGE_SECONDS", -1)  # already expired the moment it's created
    token = create_session_token(user_id=1)
    assert verify_session_token(token) is None
