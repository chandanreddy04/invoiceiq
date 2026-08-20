"""
Tests for is_available()'s caching layer - the fix for a real gap
found live: Render polls /health roughly every 5 seconds, and each
uncached call burned a real request against Groq's daily quota (no
free "is it up" endpoint exists), exhausting it purely from health
polling, independent of any real app usage.
"""

from app.services import llm_client


def test_is_available_only_checks_once_within_ttl(monkeypatch):
    llm_client._availability_cache["result"] = None
    llm_client._availability_cache["checked_at"] = 0.0

    call_count = {"n": 0}

    def fake_check():
        call_count["n"] += 1
        return True

    monkeypatch.setattr(llm_client, "_check_availability_now", fake_check)

    for _ in range(5):
        assert llm_client.is_available() is True

    assert call_count["n"] == 1  # cached for all 5 calls, not re-checked every time


def test_is_available_rechecks_after_ttl_expires(monkeypatch):
    llm_client._availability_cache["result"] = None
    llm_client._availability_cache["checked_at"] = 0.0

    call_count = {"n": 0}
    monkeypatch.setattr(llm_client, "_check_availability_now", lambda: (call_count.__setitem__("n", call_count["n"] + 1), True)[1])

    fake_now = [1000.0]
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: fake_now[0])

    llm_client.is_available()
    assert call_count["n"] == 1

    fake_now[0] += llm_client._AVAILABILITY_CACHE_TTL_SECONDS + 1
    llm_client.is_available()
    assert call_count["n"] == 2  # TTL expired, real check ran again


def test_is_available_caches_false_result_too(monkeypatch):
    """A cached "unavailable" must stay cached the same as "available" -
    this isn't just an optimization for the happy path."""
    llm_client._availability_cache["result"] = None
    llm_client._availability_cache["checked_at"] = 0.0

    call_count = {"n": 0}

    def fake_check():
        call_count["n"] += 1
        return False

    monkeypatch.setattr(llm_client, "_check_availability_now", fake_check)

    assert llm_client.is_available() is False
    assert llm_client.is_available() is False
    assert call_count["n"] == 1
