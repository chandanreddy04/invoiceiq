"""
datetime.utcnow() is deprecated since Python 3.12 in favor of
timezone-aware datetimes, but every DateTime column in this app is
naive (SQLite has no native timezone type). Mixing a timezone-aware
"now" with naive stored values would break comparisons like
`utcnow_naive() - vendor.created_at`, so this returns an aware UTC
time and immediately strips the tzinfo - same naive-UTC value
utcnow() used to give, non-deprecated API underneath.
"""

from datetime import datetime, timezone


def utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
