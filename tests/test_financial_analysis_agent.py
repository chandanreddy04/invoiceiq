"""
Tests for the AI Assistant's deterministic safety net -
_correct_ranking_intent(). Real gap found live on the cloud deployment:
Groq's free-tier model (openai/gpt-oss-20b) consistently answered
"which vendor do I spend the least with" with a generic summary instead
of the ranked list - reproduced even with that exact question spelled
out as a worked example in the prompt, so it's a genuine model-accuracy
gap, not a prompt-wording problem. "most" vs "least" is a closed,
fixed vocabulary, so it's corrected deterministically here rather than
by trying to out-word the model.
"""

from datetime import date, timedelta
from decimal import Decimal

from app.agents.financial_analysis_agent import _correct_ranking_intent, _format_with_conversion, answer_question
from app.services import fx_rate_service
from app.models.models import Invoice, InvoiceDirection, InvoiceStatus, PaymentStatus, Vendor
from app.schemas.query_intent import QueryIntent


def test_correct_ranking_intent_fixes_a_missed_least_question():
    """Reproduces the exact real-world failure: the model returned
    wants_summary=true for a "least" question instead of recognizing it
    as a ranking request at all."""
    bad_intent = QueryIntent(wants_summary=True)
    fixed = _correct_ranking_intent("Which vendor do I spend the least with?", bad_intent)

    assert fixed.wants_summary is False
    assert fixed.wants_aggregate is True
    assert fixed.aggregate_by == "vendor"
    assert fixed.aggregate_metric == "total"
    assert fixed.aggregate_order == "lowest"


def test_correct_ranking_intent_fixes_a_missed_most_question():
    bad_intent = QueryIntent(wants_summary=True)
    fixed = _correct_ranking_intent("Which vendor is the biggest expense?", bad_intent)

    assert fixed.wants_aggregate is True
    assert fixed.aggregate_order == "highest"


def test_correct_ranking_intent_detects_customer_and_category():
    fixed = _correct_ranking_intent("Which customer do I bill the least?", QueryIntent(wants_summary=True))
    assert fixed.aggregate_by == "customer"

    fixed = _correct_ranking_intent("Which expense category is the smallest?", QueryIntent(wants_summary=True))
    assert fixed.aggregate_by == "category"


def test_correct_ranking_intent_never_overrides_a_correct_answer():
    """Only a safety net for a missed case - must never touch an intent
    the model already parsed correctly (a real risk: overriding a
    correct aggregate_by like "category" back to the "vendor" default
    would make things worse, not better)."""
    already_correct = QueryIntent(
        wants_aggregate=True, aggregate_by="category", aggregate_metric="average", aggregate_order="lowest",
    )
    result = _correct_ranking_intent("Which category has the least average spend?", already_correct)
    assert result == already_correct


def test_correct_ranking_intent_leaves_unrelated_questions_alone():
    intent = QueryIntent(wants_summary=True)
    result = _correct_ranking_intent("What do I owe?", intent)
    assert result == intent


def test_answer_question_recovers_when_llm_misses_a_least_question(db_session, org, monkeypatch):
    """Full integration path: even if the LLM call itself returns the
    exact wrong (but validly-parsed) answer seen live, the final answer
    users see should still be the correct ranked breakdown."""
    v1 = Vendor(organization_id=org.id, name="Big Spend Vendor", email="big@test.example")
    v2 = Vendor(organization_id=org.id, name="Small Spend Vendor", email="small@test.example")
    db_session.add_all([v1, v2])
    db_session.commit()

    db_session.add(Invoice(
        organization_id=org.id, direction=InvoiceDirection.incoming, invoice_number="BIG-1", vendor_id=v1.id,
        invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("1000"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("1000"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    ))
    db_session.add(Invoice(
        organization_id=org.id, direction=InvoiceDirection.incoming, invoice_number="SMALL-1", vendor_id=v2.id,
        invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("50"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("50"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    ))
    db_session.commit()

    def _fake_chat(model, messages, format=None, options=None, stream=False):
        import json
        # Reproduces the real observed failure: valid JSON, wants_summary
        # true, none of the aggregate fields set - exactly what
        # gpt-oss-20b returned for this exact question live.
        return {"message": {"content": json.dumps({
            "wants_summary": True, "direction": None, "payment_status": None, "min_total": None,
            "max_total": None, "overdue_only": False, "party_name": None, "invoice_number": None,
            "category": None, "risky_only": False, "wants_aggregate": False, "aggregate_by": None,
            "aggregate_metric": None, "aggregate_order": None,
        })}}

    import ollama
    monkeypatch.setattr(ollama, "chat", _fake_chat)

    result = answer_question(db_session, org.id, "Which vendor do I spend the least with?")

    assert "Small Spend Vendor" in result["answer"]
    assert "is lowest at" in result["answer"]
    assert result["intent"]["wants_aggregate"] is True


def test_format_with_conversion_skips_fx_lookup_for_a_single_currency(monkeypatch):
    """The common case - no reason to ever call out to the FX API when
    there's nothing to convert."""
    def _fail(*a, **kw):
        raise AssertionError("should never fetch a rate for a single currency")
    monkeypatch.setattr(fx_rate_service, "convert_totals_to_single_currency", _fail)

    result = _format_with_conversion({"USD": Decimal("300.00")})
    assert result == "300.00 USD"


def test_format_with_conversion_appends_combined_total_for_multiple_currencies(monkeypatch):
    monkeypatch.setattr(fx_rate_service, "convert_totals_to_single_currency", lambda totals, target=None: Decimal("354.23"))

    result = _format_with_conversion({"EUR": Decimal("50.00"), "USD": Decimal("300.00")})
    assert result == "50.00 EUR + 300.00 USD (≈ 354.23 USD at today's rate)"


def test_format_with_conversion_falls_back_cleanly_when_fx_lookup_fails(monkeypatch):
    """The rate service is offline or the currency isn't supported -
    the per-currency breakdown must still be shown, not an error."""
    monkeypatch.setattr(fx_rate_service, "convert_totals_to_single_currency", lambda totals, target=None: None)

    result = _format_with_conversion({"EUR": Decimal("50.00"), "USD": Decimal("300.00")})
    assert result == "50.00 EUR + 300.00 USD"
    assert "≈" not in result
