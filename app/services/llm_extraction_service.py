"""
Real LLM-based field extraction - what Phase 2's regex parser was
always meant to be replaced by. Same job (turn invoice text into
structured fields), completely different mechanism: language
understanding instead of pattern matching, which is why it can find
line items in an arbitrary layout and the regex parser never could.

We ask Ollama for structured output by passing the Pydantic schema's
JSON Schema as the `format` argument - the model is constrained to
produce valid JSON matching that shape, so we never have to parse
free-text prose out of the reply or hope the model "remembered" to
use JSON.

This function does ONE thing: text in, structured guess out. It is
deliberately not an agent (no memory, no tools, no multi-step
reasoning, no decision-making) - see Section 1/36 of the project
design. It becomes one component INSIDE the real Extraction Agent
once that agent is built in Phase 4.
"""

import json
import logging

import ollama

from app.core.config import OLLAMA_MODEL
from app.schemas.extraction import LLMExtractedInvoice

logger = logging.getLogger(__name__)

MODEL_NAME = OLLAMA_MODEL  # re-exported: every agent imports MODEL_NAME from this module

SYSTEM_PROMPT = (
    "You extract structured data from invoice text. Read the invoice text "
    "and fill in the fields as accurately as possible. If a field is not "
    "present in the text, leave it as null/default. Dates must be in "
    "YYYY-MM-DD format. Only include line items that actually appear as "
    "billed goods or services - do not invent items."
)


class LLMUnavailableError(Exception):
    """Raised when Ollama can't be reached or the model isn't pulled yet."""


def extract_invoice_with_llm(raw_text: str) -> LLMExtractedInvoice:
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Invoice text:\n\n{raw_text}"},
            ],
            format=LLMExtractedInvoice.model_json_schema(),
            options={"temperature": 0},  # deterministic-ish extraction, not creative writing
        )
    except Exception as e:
        logger.warning("LLM extraction failed: %s", e)
        raise LLMUnavailableError(str(e)) from e

    content = response["message"]["content"]
    try:
        data = json.loads(content)
        return LLMExtractedInvoice.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("LLM returned invalid structured output: %s", e)
        raise LLMUnavailableError(f"Model output did not match schema: {e}") from e
