"""
Single seam between every agent and whichever LLM backend is actually
running. Locally that's Ollama serving phi3.5 on localhost - the
project's "local-first, free by default" starting point. Render's free
tier has no CPU/RAM budget to run even that small a model, so the cloud
deployment needs a different backend - not different agent code.

When GROQ_API_KEY is set (only on the cloud deployment; unset locally),
every call here goes to Groq's free-tier hosted API instead, an
OpenAI-compatible endpoint. Agents never import ollama or call Groq's
API directly - they only ever call chat() and is_available() here, and
keep validating the returned JSON against their own Pydantic schema and
raising LLMUnavailableError on failure exactly as before. Swapping
backends changes where the words come from, not the contract agents
already rely on (same as Section 38's original failure-handling design).
"""

import json
import logging
import time
from typing import Iterator

from app.core.config import GROQ_API_KEY, GROQ_MODEL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

MODEL_NAME = GROQ_MODEL if GROQ_API_KEY else OLLAMA_MODEL


class LLMUnavailableError(Exception):
    """Raised when the active LLM backend can't be reached or returns unusable output."""


def chat(messages: list[dict], schema: dict | None = None, temperature: float = 0.0) -> str:
    """Returns the model's raw reply text. `schema` is a Pydantic model's
    model_json_schema() - Ollama constrains decoding to it directly;
    Groq's JSON mode only guarantees valid JSON, not a specific shape, so
    the schema is also given to it as instructions. Either way, the
    caller is still responsible for validating the result against the
    real Pydantic model afterward, same as before this abstraction."""
    if GROQ_API_KEY:
        return _chat_groq(messages, schema, temperature)
    return _chat_ollama(messages, schema, temperature)


def chat_stream(messages: list[dict], temperature: float = 0.0) -> Iterator[str]:
    """Yields the model's reply incrementally, token-chunk by token-chunk,
    instead of waiting for the full response. Only meaningful for
    free-text prose a human reads live (Fraud/Risk's explanation, Payment/AP's
    narration) - never used for structured-output calls, since a partial
    JSON object isn't useful to show mid-generation and every structured
    caller already validates the complete result against a Pydantic model
    afterward. No `schema` parameter for that reason.

    Ollama streams genuinely incrementally. Groq's REST API also supports
    SSE streaming, but implementing and testing that path requires a real
    GROQ_API_KEY this project doesn't have during development - so the
    Groq branch yields the complete response as a single chunk (correct
    output, no incremental UX) rather than shipping an SSE parser that
    was never actually run against a live Groq response."""
    if GROQ_API_KEY:
        yield _chat_groq(messages, schema=None, temperature=temperature)
        return
    yield from _chat_ollama_stream(messages, temperature)


def _chat_ollama_stream(messages: list[dict], temperature: float) -> Iterator[str]:
    import ollama

    try:
        for chunk in ollama.chat(model=OLLAMA_MODEL, messages=messages, options={"temperature": temperature}, stream=True):
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
    except Exception as e:
        raise LLMUnavailableError(str(e)) from e


def _chat_ollama(messages: list[dict], schema: dict | None, temperature: float) -> str:
    import ollama

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            format=schema,
            options={"temperature": temperature},
        )
        return response["message"]["content"]
    except Exception as e:
        raise LLMUnavailableError(str(e)) from e


def _chat_groq(messages: list[dict], schema: dict | None, temperature: float) -> str:
    import httpx

    messages = [dict(m) for m in messages]  # copy - never mutate the caller's list
    extra: dict = {}
    if schema is not None:
        instruction = f"Respond with ONLY a JSON object matching this schema: {json.dumps(schema)}"
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += "\n\n" + instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})
        extra["response_format"] = {"type": "json_object"}

    try:
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": GROQ_MODEL, "messages": messages, "temperature": temperature, **extra},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        # Logged, not just wrapped - a bad model ID and a bad API key both
        # collapse into the same generic "LLM unavailable" from the
        # caller's side, and the /health endpoint deliberately doesn't
        # expose failure detail (it's a public, unauthenticated route).
        # This is the one place that detail survives anywhere - previously
        # it was silently discarded, which is exactly why a deprecated
        # model ID was indistinguishable from an invalid key without
        # checking Groq's model list directly.
        body = getattr(getattr(e, "response", None), "text", "")
        logger.warning("Groq call failed (model=%s): %s %s", GROQ_MODEL, e, body[:300])
        raise LLMUnavailableError(str(e)) from e


_AVAILABILITY_CACHE_TTL_SECONDS = 300  # 5 minutes
_availability_cache = {"result": None, "checked_at": 0.0}


def is_available() -> bool:
    """Used by /health. Cached for _AVAILABILITY_CACHE_TTL_SECONDS - real
    gap found live: Render polls /health roughly every 5 seconds as its
    own ongoing platform health check, and Groq has no free "is it up"
    endpoint short of a real chat request (see _check_availability_now
    below), so an uncached check was burning a real request against
    Groq's daily quota on every single probe - over 17,000 requests/day
    from health polling ALONE, independent of any actual app usage,
    which is what was actually exhausting the free-tier daily limit.
    Caching keeps /health cheap while still reflecting a real Groq
    outage within a few minutes, not instantly - an acceptable tradeoff
    for a field that was already documented as "reported, not fatal"."""
    now = time.monotonic()
    if now - _availability_cache["checked_at"] < _AVAILABILITY_CACHE_TTL_SECONDS:
        return _availability_cache["result"]

    result = _check_availability_now()
    _availability_cache["result"] = result
    _availability_cache["checked_at"] = now
    return result


def _check_availability_now() -> bool:
    """The real, uncached check - kept separate from is_available()'s
    caching wrapper so each stays simple and independently testable."""
    if GROQ_API_KEY:
        try:
            _chat_groq([{"role": "user", "content": "ping"}], schema=None, temperature=0)
            return True
        except LLMUnavailableError:
            return False
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False
