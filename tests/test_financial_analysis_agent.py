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

from app.agents.financial_analysis_agent import (
    _correct_ranking_intent, _format_with_conversion, answer_question,
    is_compound_question, parse_intent_decomposed,
)
from app.services import fx_rate_service
from app.services.llm_client import LLMUnavailableError
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


# ---------------------------------------------------------------------------
# is_compound_question() - the deterministic gate deciding which extraction
# path a question takes. Every case here is grounded in the project's own
# real evaluation questions (scripts/evaluate_agents.py's INTENT_CASES).
# ---------------------------------------------------------------------------

def test_single_constraint_questions_are_not_compound():
    assert is_compound_question("What do I owe?") is False
    assert is_compound_question("How much am I owed?") is False
    assert is_compound_question("Show overdue invoices") is False
    assert is_compound_question("List all Golden Grain invoices") is False


def test_amount_alone_is_compound_too():
    """A real gap found while measuring the fix: even a single amount
    filter with nothing else reliably dropped the value via the big
    schema (three live runs, 0/3) - the dedicated amount-only call got
    it right 3/3, so amount questions route through it even alone."""
    assert is_compound_question("List invoices under 100 dollars") is True


def test_status_plus_amount_is_compound():
    """The exact real failure case: measured live at 78.7s against a
    reasoning model, which still dropped the amount filter anyway."""
    assert is_compound_question("Show unpaid invoices over 500 dollars") is True


def test_status_plus_direction_is_compound():
    """The other real failure case: measured live at 303.5s against a
    reasoning model, which never produced an answer at all."""
    assert is_compound_question("Show paid outgoing invoices") is True


def test_aggregate_plus_category_is_compound():
    assert is_compound_question("Which vendor do I spend the most with in the utilities category?") is True


# ---------------------------------------------------------------------------
# parse_intent_decomposed() - mocked chat() calls, one per dimension.
# ---------------------------------------------------------------------------

def test_parse_intent_decomposed_merges_all_four_calls(monkeypatch):
    """The actual point of decomposition: a status word and a number each
    land in their own call, with their own tiny schema, so neither call
    has to choose which field to keep - both survive into the merge."""
    import json as json_module
    import app.agents.financial_analysis_agent as faa

    responses = [
        {"wants_summary": False, "wants_aggregate": False, "aggregate_by": None, "aggregate_metric": None, "aggregate_order": None},
        {"payment_status": "unpaid", "direction": None, "overdue_only": False},
        {"min_total": 500, "max_total": None},
        {"party_name": None, "invoice_number": None, "category": None, "risky_only": False},
    ]
    call_order = iter(responses)

    def fake_chat(messages, schema=None):
        return json_module.dumps(next(call_order))
    monkeypatch.setattr(faa, "chat", fake_chat)

    intent = parse_intent_decomposed("Show unpaid invoices over 500 dollars")

    assert intent.payment_status == "unpaid"
    assert intent.min_total == 500
    assert intent.wants_summary is False


def test_parse_intent_decomposed_propagates_llm_unavailable(monkeypatch):
    """Same fallback contract as parse_intent() - a decomposed parse that
    can't complete must fail the same clean way, not return a partially-
    filled intent that looks more confident than it actually is."""
    import app.agents.financial_analysis_agent as faa

    def _raise(*a, **kw):
        raise LLMUnavailableError("no model")
    monkeypatch.setattr(faa, "chat", _raise)

    try:
        parse_intent_decomposed("Show unpaid invoices over 500 dollars")
        assert False, "expected LLMUnavailableError"
    except LLMUnavailableError:
        pass


def test_answer_question_routes_compound_questions_to_decomposed_parse(db_session, org, monkeypatch):
    """The gate itself, exercised end to end - a compound question must
    call the decomposed parser, not the single fast one."""
    import app.agents.financial_analysis_agent as faa

    calls = {"decomposed": 0, "fast": 0}
    monkeypatch.setattr(faa, "parse_intent_decomposed", lambda q: (calls.__setitem__("decomposed", calls["decomposed"] + 1), QueryIntent(payment_status="unpaid", min_total=500))[1])
    monkeypatch.setattr(faa, "parse_intent", lambda q: (calls.__setitem__("fast", calls["fast"] + 1), QueryIntent(wants_summary=True))[1])

    answer_question(db_session, org.id, "Show unpaid invoices over 500 dollars")
    assert calls == {"decomposed": 1, "fast": 0}

    answer_question(db_session, org.id, "What do I owe?")
    assert calls == {"decomposed": 1, "fast": 1}
