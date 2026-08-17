"""
The slow tier: real calls to the actual local Ollama model, no
mocking. Run these with `pytest -m llm` on their own, or they run
along with everything else by default since Ollama is available in
this dev environment. Skipped automatically if Ollama isn't reachable
(Section 38 - the suite shouldn't hard-fail just because the model
service happens to be down).

These only assert on cases already empirically confirmed reliable in
Phase 3/6 manual testing (see conversation history) - NOT the
compound multi-field query case that phi3.5 was shown to handle
unreliably. That reliability gap is measured properly in
scripts/evaluate_agents.py instead, where a probabilistic result is
reported as a number, not asserted as pass/fail.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.llm


def _ollama_available() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


skip_if_no_ollama = pytest.mark.skipif(not _ollama_available(), reason="Ollama is not reachable")


@skip_if_no_ollama
def test_real_llm_responds():
    import ollama
    from app.services.llm_extraction_service import MODEL_NAME
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": "Reply with the word OK."}])
    assert len(response["message"]["content"]) > 0


@skip_if_no_ollama
def test_real_extraction_from_sample_pdf():
    from app.services.extraction_service import extract_text_from_pdf
    from app.services.llm_extraction_service import extract_invoice_with_llm

    pdf_path = Path(__file__).resolve().parent.parent / "data" / "invoices" / "test_invoice_sunrise.pdf"
    if not pdf_path.exists():
        pytest.skip("Sample PDF not generated - run scripts/make_test_invoice_pdf.py first")

    text = extract_text_from_pdf(pdf_path.read_bytes())
    result = extract_invoice_with_llm(text)

    assert result.invoice_number == "SPC-2201"
    assert result.invoice_date == "2026-07-15"
    assert result.due_date == "2026-08-14"
    assert len(result.line_items) == 2


@skip_if_no_ollama
def test_real_fraud_explanation_produces_nonempty_text():
    from app.agents.fraud_risk_agent import explain_risk_with_llm
    from app.models.models import Invoice, InvoiceDirection, InvoiceStatus, PaymentStatus
    from datetime import date
    from decimal import Decimal

    invoice = Invoice(
        id=1, organization_id=1, direction=InvoiceDirection.incoming, invoice_number="TEST-1",
        invoice_date=date.today(), due_date=date.today(), subtotal=Decimal("100"), tax=Decimal("0"),
        discount=Decimal("0"), total=Decimal("5000"), payment_status=PaymentStatus.unpaid,
        invoice_status=InvoiceStatus.pending_review,
    )
    explanation = explain_risk_with_llm(invoice, 0.85, ["Amount is 10.0x this vendor's average invoice.", "Vendor was added only 1 day(s) ago."])
    assert isinstance(explanation, str)
    assert len(explanation) > 20


@skip_if_no_ollama
def test_real_intent_parsing_summary_case():
    """The one query type Phase 6 confirmed phi3.5 handles reliably."""
    from app.agents.financial_analysis_agent import parse_intent
    intent = parse_intent("What do I owe?")
    assert intent.wants_summary is True
