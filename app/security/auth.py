"""
Authentication - deliberately built on Python's standard library only
(hashlib, hmac, secrets), not a third-party auth framework. For a
single-organization local demo this is enough to demonstrate the real
concepts (Section 11: password hashing, session tokens, RBAC) without
adding a dependency chain just to log a couple of people in.

Password hashing: hashlib.scrypt (memory-hard, stdlib since Python
3.6, recommended by the Python docs over anything home-rolled from
plain SHA-256). Never store or compare plain-text passwords.

Sessions: a signed cookie, not a database-backed session table -
simpler for this scale, and the signature (HMAC-SHA256) makes it
tamper-evident: a client can't forge "user_id=1" without knowing
SECRET_KEY, and can't extend their own expiry either.
"""

import hashlib
import hmac
import secrets
import time

from app.core.config import SECRET_KEY as _SECRET_KEY_STR

SECRET_KEY = _SECRET_KEY_STR.encode()
SESSION_MAX_AGE_SECONDS = 60 * 60 * 8  # 8 hours

_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 16384, 8, 1


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"{salt.hex()}${derived.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        salt_hex, derived_hex = password_hash.split("$")
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(derived_hex)
    actual = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return hmac.compare_digest(actual, expected)  # constant-time, avoids timing attacks


def _sign(payload: str) -> str:
    return hmac.new(SECRET_KEY, payload.encode(), hashlib.sha256).hexdigest()


def create_session_token(user_id: int) -> str:
    expiry = int(time.time()) + SESSION_MAX_AGE_SECONDS
    payload = f"{user_id}.{expiry}"
    return f"{payload}.{_sign(payload)}"


def verify_session_token(token: str) -> int | None:
    """Returns the user_id if the token is validly signed and not
    expired, else None. Never trusts the token's claimed user_id
    without checking the signature first."""
    try:
        user_id_str, expiry_str, signature = token.split(".")
    except ValueError:
        return None

    payload = f"{user_id_str}.{expiry_str}"
    if not hmac.compare_digest(_sign(payload), signature):
        return None
    if int(expiry_str) < time.time():
        return None
    return int(user_id_str)
