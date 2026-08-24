"""
Day 1 practice: what does a "reasoning model" actually do differently
from the chat models this project already uses?

Every other LLM call in this codebase (phi3.5 for extraction/narration,
Groq's gpt-oss for the cloud deployment) is a one-shot chat completion -
ask, get an answer, done. A reasoning model (deepseek-r1 here) is
trained to emit an extended chain-of-thought BEFORE its final answer,
and modern serving stacks (Ollama's `think` request field, OpenAI's
`reasoning_effort`/o-series, DeepSeek's own API) expose that
deliberation as a separate, inspectable field rather than making you
regex it out of the prose.

This script runs the SAME real, borderline fraud-signal set (the exact
"two moderate signals summing to just under threshold" case
PROJECT_REPORT.md documents as this project's one genuine measured
failure mode) through:

  1. phi3.5 via chat()   - narration only, what explain_risk_with_llm()
                            already does in production
  2. deepseek-r1:8b via reason() - genuine step-by-step deliberation,
                            what deliberate_on_borderline_case() now
                            does for borderline scores only

...and prints them side by side, so the difference is something you can
actually see, not just read about. Run it after `ollama pull deepseek-r1:8b`
(see .env.example) and with `ollama serve` running.

    python scripts/reasoning_model_practice.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.fraud_risk_agent import (
    _risk_explanation_prompt, _borderline_review_prompt, HIGH_RISK_THRESHOLD, BORDERLINE_BAND,
)
from app.services.llm_client import chat, reason, LLMUnavailableError

# The real borderline case: two moderate signals (amount 1.5-3x typical,
# AND a moderately-similar invoice number) that individually wouldn't
# cross HIGH_RISK_THRESHOLD, but together land right in the ambiguous
# band this project's own evaluation flagged as its weakest spot.
RISK_SCORE = 0.60
REASONS = [
    "Amount is 2.1x this vendor's average invoice (moderately high).",
    "Invoice number closely resembles a previous invoice ('INV-2024-0087', 87% similar).",
]


def main():
    print(f"Scenario: risk={RISK_SCORE:.0%} (threshold={HIGH_RISK_THRESHOLD:.0%}, "
          f"borderline band={BORDERLINE_BAND[0]:.0%}-{BORDERLINE_BAND[1]:.0%})")
    for r in REASONS:
        print(f"  - {r}")
    print()

    print("=" * 70)
    print("1. phi3.5 via chat() - the narration-only call this project already")
    print("   makes for EVERY flagged invoice (explain_risk_with_llm).")
    print("=" * 70)
    t0 = time.monotonic()
    try:
        narration = chat(messages=[{"role": "user", "content": _risk_explanation_prompt(RISK_SCORE, REASONS)}])
        print(narration.strip())
    except LLMUnavailableError as e:
        print(f"[unavailable: {e}]")
    print(f"({time.monotonic() - t0:.1f}s, no separate reasoning trace - it went straight to prose)")

    print()
    print("=" * 70)
    print("2. deepseek-r1:8b via reason() - what deliberate_on_borderline_case()")
    print("   now calls ONLY for scores inside BORDERLINE_BAND like this one.")
    print("=" * 70)
    t0 = time.monotonic()
    try:
        result = reason(messages=[{"role": "user", "content": _borderline_review_prompt(RISK_SCORE, REASONS)}])
        print("--- thinking (message.thinking - the actual chain-of-thought) ---")
        print(result["thinking"])
        print("\n--- content (message.content - the final answer) ---")
        print(result["content"])
    except LLMUnavailableError as e:
        print(f"[unavailable: {e}]")
    print(f"({time.monotonic() - t0:.1f}s)")

    print()
    print("Notice: phi3.5 above only ever produces the equivalent of 'content' -")
    print("it has no separate deliberation step to inspect, by design (it's not")
    print("a reasoning model). That's the actual API-level difference this")
    print("exercise is about, not just 'the second one sounds smarter.'")


if __name__ == "__main__":
    main()
