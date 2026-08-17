"""
Phase 2 extraction: pull raw text out of an uploaded PDF, then make a
best-effort guess at a few header fields using regex.

This regex parser is intentionally naive and is NOT the real Extraction
Agent from the project design (Section 4/14) - it cannot reliably find
line items in an arbitrary vendor layout, and it has no real language
understanding. It exists so the full pipeline (upload -> extract ->
human review -> save) works end-to-end right now. Phase 3 replaces
just this function's job - guessing fields from text - with an LLM
call, without changing anything else in the pipeline. That swap is
the clearest demonstration in the whole project of "the LLM is one
replaceable component, not the whole system."

No OCR here on purpose (see project simplification decision): this
only handles PDFs that already have a text layer, which covers most
real-world invoices. Scanned image support is a documented future
extension, not required for the core pipeline to work.
"""

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

import fitz  # PyMuPDF


def extract_text_from_pdf(file_bytes: bytes) -> str:
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


DATE_PATTERNS = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"]


def _parse_date(text: str) -> date | None:
    for fmt in DATE_PATTERNS:
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _first_match(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else None


class ExtractedFields:
    def __init__(self):
        self.invoice_number: str | None = None
        self.invoice_date: date | None = None
        self.due_date: date | None = None
        self.total: Decimal | None = None
        self.confidence: float = 0.0


def naive_parse_invoice_fields(text: str) -> ExtractedFields:
    result = ExtractedFields()
    found_count = 0

    invoice_number = _first_match(r"invoice\s*(?:#|number|no\.?)\s*[:\-]?\s*([A-Za-z0-9\-]+)", text)
    if invoice_number:
        result.invoice_number = invoice_number
        found_count += 1

    invoice_date_str = _first_match(r"invoice\s*date\s*[:\-]?\s*([A-Za-z0-9/,\- ]+?)(?:\n|$)", text)
    if invoice_date_str and (parsed := _parse_date(invoice_date_str)):
        result.invoice_date = parsed
        found_count += 1

    due_date_str = _first_match(r"due\s*date\s*[:\-]?\s*([A-Za-z0-9/,\- ]+?)(?:\n|$)", text)
    if due_date_str and (parsed := _parse_date(due_date_str)):
        result.due_date = parsed
        found_count += 1

    # negative lookbehind for "sub" so "Subtotal: 425.00" doesn't get
    # mistaken for the actual total - a real bug this parser hit during
    # testing, and a good example of why naive text matching is fragile.
    total_str = _first_match(r"(?<!sub)total\s*(?:due|amount)?\s*[:\-]?\s*\$?\s*([\d,]+\.\d{2})", text)
    if total_str:
        try:
            result.total = Decimal(total_str.replace(",", ""))
            found_count += 1
        except InvalidOperation:
            pass

    # confidence is just "how many of the 4 fields did we find" - a crude
    # but honest stand-in until Phase 3's LLM extraction reports real
    # per-field confidence.
    result.confidence = round(found_count / 4, 2)
    return result
