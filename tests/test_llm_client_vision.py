"""
Test for chat_with_image()'s Ollama branch - the same scope the rest of
this project's llm_client coverage sticks to (no existing test exercises
chat()'s Groq branch either; that's plain httpx mechanics, not agent
behavior worth pinning down with a mock).
"""

from app.services import llm_client


def test_chat_with_image_passes_image_bytes_to_ollama(monkeypatch):
    captured = {}

    def fake_chat(model, messages, format=None, options=None):
        captured["model"] = model
        captured["messages"] = messages
        return {"message": {"content": "ok"}}

    import ollama
    monkeypatch.setattr(ollama, "chat", fake_chat)
    monkeypatch.setattr(llm_client, "GROQ_API_KEY", "")  # force the Ollama branch

    result = llm_client.chat_with_image("describe this invoice", b"fake-png-bytes")

    assert result == "ok"
    assert captured["model"] == llm_client.OLLAMA_VISION_MODEL
    assert captured["messages"][0]["images"] == [b"fake-png-bytes"]


def test_chat_with_image_raises_llm_unavailable_on_ollama_error(monkeypatch):
    import ollama

    def fake_chat(*a, **kw):
        raise ConnectionError("ollama not running")
    monkeypatch.setattr(ollama, "chat", fake_chat)
    monkeypatch.setattr(llm_client, "GROQ_API_KEY", "")

    try:
        llm_client.chat_with_image("describe this invoice", b"fake-png-bytes")
        assert False, "expected LLMUnavailableError"
    except llm_client.LLMUnavailableError:
        pass
