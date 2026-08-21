"""
Unit tests for the naive regex parser (Phase 2 fallback). No LLM, no
database - just text in, fields out.
"""

from decimal import Decimal

from app.services.extraction_service import naive_parse_invoice_fields, _guess_vendor_header


SAMPLE_TEXT = """Sunrise Packaging Co.
21 Industrial Way

INVOICE

Invoice Number: SPC-2201
Invoice Date: 2026-07-15
Due Date: 2026-08-14

Subtotal: 425.00
Tax: 10.00
Total Due: $435.00
"""


def test_finds_invoice_number():
    result = naive_parse_invoice_fields(SAMPLE_TEXT)
    assert result.invoice_number == "SPC-2201"


def test_finds_invoice_and_due_dates():
    result = naive_parse_invoice_fields(SAMPLE_TEXT)
    assert result.invoice_date.isoformat() == "2026-07-15"
    assert result.due_date.isoformat() == "2026-08-14"


def test_finds_total_not_subtotal():
    """Regression test: this regex originally matched 'Subtotal: 425.00'
    because 'otal' is a substring of 'Subtotal' - caught by hand during
    Phase 2 testing, fixed with a negative lookbehind. This test pins
    that fix down so it can't silently regress."""
    result = naive_parse_invoice_fields(SAMPLE_TEXT)
    assert result.total == Decimal("435.00")


def test_finds_vendor_name_from_first_line():
    result = naive_parse_invoice_fields(SAMPLE_TEXT)
    assert result.vendor_name == "Sunrise Packaging Co."


def test_finds_vendor_address_from_second_line():
    result = naive_parse_invoice_fields(SAMPLE_TEXT)
    assert result.vendor_address == "21 Industrial Way"


def test_vendor_address_not_guessed_when_second_line_is_a_label():
    """A document with no real address line right after the vendor name
    (e.g. it goes straight into "Invoice Number: ...") must not
    mistake that label for the address."""
    text = "Acme Corp\nInvoice Number: AC-1\n"
    result = naive_parse_invoice_fields(text)
    assert result.vendor_name == "Acme Corp"
    assert result.vendor_address is None


def test_finds_vendor_tax_id_when_present():
    text = SAMPLE_TEXT + "\nTax ID: 91-2345678\n"
    result = naive_parse_invoice_fields(text)
    assert result.vendor_tax_id == "91-2345678"


def test_vendor_tax_id_absent_does_not_affect_confidence():
    """Most real invoices never print a tax ID - that must not make an
    otherwise fully-extracted invoice look low-confidence."""
    result = naive_parse_invoice_fields(SAMPLE_TEXT)
    assert result.vendor_tax_id is None
    assert result.confidence == 1.0


def test_guess_vendor_header_skips_invoice_title_line():
    text = "INVOICE\nGolden Grain Milling\n500 Mill Rd\n"
    name, address = _guess_vendor_header(text)
    assert name == "Golden Grain Milling"
    assert address == "500 Mill Rd"


def test_guess_vendor_header_returns_none_on_empty_text():
    assert _guess_vendor_header("") == (None, None)


def test_confidence_is_1_when_all_five_fields_found():
    result = naive_parse_invoice_fields(SAMPLE_TEXT)
    assert result.confidence == 1.0


def test_confidence_is_0_on_empty_text():
    result = naive_parse_invoice_fields("")
    assert result.confidence == 0.0
    assert result.invoice_number is None
    assert result.total is None


def test_partial_match_gives_partial_confidence():
    text = "Invoice Number: XYZ-1\nSome other unrelated text with no other fields."
    result = naive_parse_invoice_fields(text)
    assert result.invoice_number == "XYZ-1"
    assert result.total is None
    assert 0 < result.confidence < 1
