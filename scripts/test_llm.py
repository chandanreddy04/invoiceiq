"""
Phase 0 sanity check: proves the local Ollama LLM server is running
and reachable, and that we can get a text response back. No agent
logic, no tools, no structured output yet - that starts in Phase 3.

Requires:
  1. Ollama installed and running (it runs as a background service
     after install - see README instructions).
  2. The model pulled once: `ollama pull phi3.5`
"""

import ollama

MODEL_NAME = "phi3.5"


def main() -> None:
    print(f"Sending a test prompt to local model '{MODEL_NAME}'...\n")

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": "In one sentence, what is an invoice?",
            }
        ],
    )

    reply = response["message"]["content"]
    print("Model replied:")
    print(reply)
    print("\nLLM test PASSED.")


if __name__ == "__main__":
    main()
