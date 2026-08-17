"""
Expense Classification Agent - the smallest of our agents, but it
still has the same shape as the others:

  INPUT      -> line item descriptions on a freshly created invoice
  CONTEXT    -> the fixed category taxonomy (this business's chart of
                accounts, simplified - a small business doesn't need a
                huge one)
  LLM LAYER  -> the only way to do this well: matching free-text like
                "industrial mixer belt replacement" to a category
                requires actual language understanding, not keyword
                rules (see Section 7 - this is a real LLM-shaped task,
                not a database query wearing an AI costume)
  REASONING  -> confidence check: if the model isn't confident, we
                still save its best guess (better than nothing for a
                demo) but log it distinctly so a human could review it
  ACTION     -> writes InvoiceItem.category for each item
  FEEDBACK   -> confirms every item got matched back to a category;
                anything the model didn't return cleanly falls back to
                "Other" rather than being silently skipped
"""

import json
import logging

from sqlalchemy.orm import Session

from app.models.models import Invoice
from app.schemas.classification import ClassificationResult
from app.services.llm_extraction_service import MODEL_NAME, LLMUnavailableError

logger = logging.getLogger(__name__)

CATEGORIES = [
    "Raw Ingredients",
    "Packaging & Supplies",
    "Equipment & Repairs",
    "Shipping & Freight",
    "Utilities",
    "Professional Services",
    "Other",
]

LOW_CONFIDENCE_THRESHOLD = 0.6


def classify_descriptions(descriptions: list[str]) -> ClassificationResult:
    import ollama

    prompt = (
        "Classify each of these invoice line items into exactly one of these categories: "
        f"{', '.join(CATEGORIES)}.\n\n"
        "Line items:\n" + "\n".join(f"- {d}" for d in descriptions) + "\n\n"
        "Return the exact original description text for each item alongside its category "
        "and your confidence (0 to 1)."
    )
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            format=ClassificationResult.model_json_schema(),
            options={"temperature": 0},
        )
        return ClassificationResult.model_validate(json.loads(response["message"]["content"]))
    except Exception as e:
        logger.warning("Classification LLM call failed: %s", e)
        raise LLMUnavailableError(str(e)) from e


def run_classification(db: Session, invoice: Invoice) -> dict:
    """Returns a summary dict for the Orchestrator's log, e.g.
    {"classified": 2, "low_confidence": 0, "fallback": 0}."""
    if not invoice.items:
        return {"classified": 0, "low_confidence": 0, "fallback": 0}

    descriptions = [item.description for item in invoice.items]
    summary = {"classified": 0, "low_confidence": 0, "fallback": 0}

    try:
        result = classify_descriptions(descriptions)
        by_description = {c.description.strip().lower(): c for c in result.items}
    except LLMUnavailableError:
        by_description = {}

    for item in invoice.items:
        match = by_description.get(item.description.strip().lower())
        if match:
            item.category = match.category
            summary["classified"] += 1
            if match.confidence < LOW_CONFIDENCE_THRESHOLD:
                summary["low_confidence"] += 1
        else:
            item.category = "Other"
            summary["fallback"] += 1

    db.commit()
    return summary
