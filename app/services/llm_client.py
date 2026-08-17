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
        raise LLMUnavailableError(str(e)) from e


def is_available() -> bool:
    """Used by /health. Cheap existence check, not a full round-trip
    chat call for the Ollama case; Groq has no equivalent "is it up"
    endpoint short of a real request, so it gets a minimal one."""
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
