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

No OCR here (see project simplification decision) - this module still
only ever pulls PyMuPDF's own text layer, nothing more. A scanned PDF
or a photographed invoice has no text layer at all, so has_extractable_
text() below exists to detect exactly that case and hand off to a
vision-capable model instead (see llm_extraction_service.
extract_invoice_from_image()) rather than the pipeline just giving up,
which is what happened before this module grew that second path.
"""

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

import fitz  # PyMuPDF


def extract_text_from_pdf(file_bytes: bytes) -> str:
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


# A scanned/image-only PDF page has no text layer at all, so PyMuPDF
# returns an empty (or whitespace/junk) string for it - there's nothing
# to OCR here, just what's embedded as real text objects. A real
# text-layer invoice, even a short one, always produces far more than
# this many characters, so the threshold is a deliberately generous,
# honest way to tell "no text layer" apart from "very short document"
# without hardcoding a page count or fabricating an OCR confidence
# score this app doesn't actually have.
MIN_TEXT_LAYER_CHARS = 20


def has_extractable_text(text: str) -> bool:
    return len(text.strip()) >= MIN_TEXT_LAYER_CHARS


def render_pdf_page_to_image(file_bytes: bytes, page_number: int = 0) -> bytes:
    """The fallback path when has_extractable_text() says no: render the
    page as a PNG so a vision-capable model can read it directly (see
    llm_extraction_service.extract_invoice_from_image()) instead of the
    pipeline simply giving up, which is all it could do before this.
    150 DPI is enough resolution for a model to read normal printed
    invoice text without producing an unnecessarily large image."""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        pixmap = doc[page_number].get_pixmap(dpi=150)
        return pixmap.tobytes("png")


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
        self.vendor_name: str | None = None
        self.vendor_address: str | None = None
        self.vendor_tax_id: str | None = None
        self.invoice_number: str | None = None
        self.invoice_date: date | None = None
        self.due_date: date | None = None
        self.total: Decimal | None = None
        self.confidence: float = 0.0


_HEADER_SKIP_LINES = ("invoice", "bill", "receipt")
_LABELED_FIELD = re.compile(r"^(invoice|bill\s*to|ship\s*to|due|date|number|no\.?|tax|ein|vat|license)\b", re.IGNORECASE)


def _guess_vendor_header(text: str) -> tuple[str | None, str | None]:
    """No LLM here to actually understand the layout, so this is a
    blunt but honest heuristic: on the overwhelming majority of real
    invoices, the vendor's own name and address are the first two
    non-blank lines, printed together near a logo/letterhead before
    any label like "Invoice" or "Bill To" appears. Returns
    (name, address) - address is only returned if a plausible second
    line immediately follows the name (not a labeled field like
    "Invoice Number: ..."), never guessed from elsewhere in the
    document where it could just as easily be the buyer's."""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]  # drop blanks, keep order
    name, address = None, None
    for i, line in enumerate(lines):
        if line.lower() in _HEADER_SKIP_LINES:
            continue
        name = line
        if i + 1 < len(lines) and not _LABELED_FIELD.match(lines[i + 1]):
            address = lines[i + 1]
        break
    return name, address


def naive_parse_invoice_fields(text: str) -> ExtractedFields:
    result = ExtractedFields()
    found_count = 0

    vendor_name, vendor_address = _guess_vendor_header(text)
    if vendor_name:
        result.vendor_name = vendor_name
        found_count += 1
    if vendor_address:
        result.vendor_address = vendor_address

    # Not counted toward found_count/confidence below - unlike the other
    # fields, a tax ID is genuinely absent from most real invoices, so
    # counting it would make ordinary, well-extracted invoices look
    # artificially low-confidence just for not printing one.
    tax_id = _first_match(r"(?:tax\s*id|ein|vat\s*(?:no\.?|number)|(?:business\s*)?license\s*#?)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-]{3,20})", text)
    if tax_id:
        result.vendor_tax_id = tax_id

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

    # confidence is just "how many of the 5 fields did we find" - a crude
    # but honest stand-in until Phase 3's LLM extraction reports real
    # per-field confidence.
    result.confidence = round(found_count / 5, 2)
    return result
