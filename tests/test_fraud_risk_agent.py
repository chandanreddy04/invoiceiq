"""
Tests for the Fraud/Risk Agent's REASONING layer specifically (not
the LLM layer - see the agent's own docstring on why that split
exists). score_risk() is pure and gets tested directly with synthetic
signal dicts; compute_risk_signals() needs a database, so those tests
use the db_session/vendor/customer fixtures from conftest.

Vendor and Customer are structurally identical for this agent's
purposes (name, email, address, created_at), so most tests exercise the
vendor path (less setup) and a smaller, targeted set proves the
customer path works identically - not a full duplicate suite for each,
since that would just be testing the same logic twice.
"""

from datetime import date, timedelta
from decimal import Decimal

from app.agents import fraud_risk_agent
from app.agents.fraud_risk_agent import (
    score_risk, compute_risk_signals, get_party, find_parties_sharing_contact_info, HIGH_RISK_THRESHOLD,
    BORDERLINE_BAND, deliberate_on_borderline_case, run_fraud_check,
)
from app.models.models import Invoice, InvoiceDirection, InvoiceStatus, PaymentStatus, Vendor, Customer, FraudFlag
from app.services.llm_client import LLMUnavailableError
from app.utils.time import utcnow_naive


def make_signals(**overrides):
    base = {
        "party_label": "vendor",
        "is_new_party": False,
        "party_age_days": 400,
        "amount_ratio": None,
        "amount_ratio_basis": None,
        "max_invoice_number_similarity": 0.0,
        "similar_invoice_number": None,
        "shared_contact_parties": [],
    }
    base.update(overrides)
    return base


def test_no_signals_gives_zero_risk():
    risk, reasons = score_risk(make_signals())
    assert risk == 0.0
    assert "No anomalies" in reasons[0]


def test_high_amount_ratio_crosses_high_risk_threshold_alone_is_not_enough():
    # 0.45 alone (amount>=3x) should NOT cross the 0.7 threshold by itself -
    # pins down that a single strong signal isn't automatically "high risk"
    risk, reasons = score_risk(make_signals(amount_ratio=5.0, amount_ratio_basis="party"))
    assert risk == 0.45
    assert risk < HIGH_RISK_THRESHOLD
    assert "5.0x" in reasons[0]


def test_new_party_plus_high_amount_crosses_threshold():
    risk, _ = score_risk(make_signals(amount_ratio=5.0, amount_ratio_basis="party", is_new_party=True, party_age_days=1))
    assert risk >= HIGH_RISK_THRESHOLD


def test_moderate_amount_ratio_scores_less_than_high_ratio():
    low_risk, _ = score_risk(make_signals(amount_ratio=2.0, amount_ratio_basis="party"))
    high_risk, _ = score_risk(make_signals(amount_ratio=5.0, amount_ratio_basis="party"))
    assert low_risk < high_risk


def test_invoice_number_similarity_signal():
    risk, reasons = score_risk(make_signals(max_invoice_number_similarity=0.9, similar_invoice_number="ABC-100"))
    assert risk == 0.30
    assert "ABC-100" in reasons[0]


def test_similarity_below_threshold_gives_no_signal():
    risk, reasons = score_risk(make_signals(max_invoice_number_similarity=0.5))
    assert risk == 0.0


def test_risk_score_never_exceeds_one():
    risk, _ = score_risk(make_signals(
        amount_ratio=50.0, amount_ratio_basis="party", is_new_party=True, party_age_days=0,
        max_invoice_number_similarity=1.0, similar_invoice_number="X",
    ))
    assert risk == 1.0


def test_reason_text_says_customer_not_vendor_for_outgoing_invoices():
    """A real gap this whole rewrite closes: reasons used to hardcode
    the word "vendor" no matter which party the invoice actually had -
    confusing/wrong on an outgoing invoice about a customer."""
    risk, reasons = score_risk(make_signals(party_label="customer", is_new_party=True, party_age_days=2))
    assert "Customer was added only 2 day(s) ago." in reasons[0]
    assert "vendor" not in reasons[0].lower()


def test_shared_contact_parties_adds_risk_and_names_the_other_party():
    risk, reasons = score_risk(make_signals(shared_contact_parties=["Golden Grain Milling"]))
    assert risk == 0.35
    assert "Golden Grain Milling" in reasons[0]


def test_shared_contact_parties_stacks_with_other_signals():
    alone, _ = score_risk(make_signals(shared_contact_parties=["Other Co."]))
    stacked, _ = score_risk(make_signals(shared_contact_parties=["Other Co."], is_new_party=True, party_age_days=1))
    assert stacked > alone


def test_org_wide_fallback_flags_new_vendor_first_large_invoice(db_session, org, vendor):
    """Regression test for the real bug found during Phase 9 synthetic
    data testing: a brand-new vendor's FIRST invoice has no vendor
    history to compute a ratio against, so without the org-wide
    fallback it only ever gets the flat new-vendor signal - even if
    the amount is wildly out of line with everything else this
    business normally sees."""
    established = Vendor(organization_id=org.id, name="Established Vendor", email="e@test.example",
                          created_at=utcnow_naive() - timedelta(days=400))
    db_session.add(established)
    db_session.commit()

    for i in range(3):
        db_session.add(Invoice(
            organization_id=org.id, direction=InvoiceDirection.incoming, invoice_number=f"EST-{i}",
            vendor_id=established.id, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
            subtotal=Decimal("100.00"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100.00"),
            payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
        ))
    db_session.commit()

    new_vendor = Vendor(organization_id=org.id, name="Suspicious New Co.", email="s@test.example",
                         created_at=utcnow_naive())
    db_session.add(new_vendor)
    db_session.commit()

    big_invoice = Invoice(
        organization_id=org.id, direction=InvoiceDirection.incoming, invoice_number="SUS-1",
        vendor_id=new_vendor.id, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("2000.00"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("2000.00"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    db_session.add(big_invoice)
    db_session.commit()
    db_session.refresh(big_invoice)

    signals = compute_risk_signals(db_session, big_invoice, new_vendor)
    assert signals["amount_ratio_basis"] == "org"
    assert signals["amount_ratio"] == 20.0  # 2000 / 100 (the established baseline)

    risk, reasons = score_risk(signals)
    assert risk >= HIGH_RISK_THRESHOLD
    assert any("typical invoice size" in r for r in reasons)


def test_org_wide_baseline_excludes_already_flagged_invoices(db_session, org, vendor):
    """Regression test for the second half of the same fix: including
    already-flagged invoices in the 'normal' baseline is circular -
    a suspected-fraud invoice would inflate the average and dilute the
    signal meant to catch the NEXT suspicious one."""
    established = Vendor(organization_id=org.id, name="Established Vendor", email="e@test.example",
                          created_at=utcnow_naive() - timedelta(days=400))
    db_session.add(established)
    db_session.commit()

    db_session.add(Invoice(
        organization_id=org.id, direction=InvoiceDirection.incoming, invoice_number="EST-NORMAL",
        vendor_id=established.id, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("100.00"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100.00"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    ))
    db_session.add(Invoice(
        organization_id=org.id, direction=InvoiceDirection.incoming, invoice_number="EST-FLAGGED",
        vendor_id=established.id, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("10000.00"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("10000.00"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.pending_review,
        risk_score=Decimal("0.9"),
    ))
    db_session.commit()

    new_vendor = Vendor(organization_id=org.id, name="New Co.", email="n@test.example", created_at=utcnow_naive())
    db_session.add(new_vendor)
    db_session.commit()

    test_invoice = Invoice(
        organization_id=org.id, direction=InvoiceDirection.incoming, invoice_number="NEW-1",
        vendor_id=new_vendor.id, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("500.00"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("500.00"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    db_session.add(test_invoice)
    db_session.commit()
    db_session.refresh(test_invoice)

    signals = compute_risk_signals(db_session, test_invoice, new_vendor)
    # Baseline should be just the $100 normal invoice, not (100+10000)/2 -
    # so ratio should be 500/100 = 5.0, not 500/5050 ≈ 0.1
    assert signals["amount_ratio"] == 5.0


def test_find_parties_sharing_contact_info_matches_on_email(db_session, org):
    v1 = Vendor(organization_id=org.id, name="Acme Supply", email="billing@acme.example", address="1 Main St")
    v2 = Vendor(organization_id=org.id, name="Acme Supply LLC", email="billing@acme.example", address="2 Other St")
    db_session.add_all([v1, v2])
    db_session.commit()

    matches = find_parties_sharing_contact_info(db_session, org.id, v1)
    assert [m.name for m in matches] == ["Acme Supply LLC"]


def test_find_parties_sharing_contact_info_address_match_is_case_and_whitespace_insensitive(db_session, org):
    v1 = Vendor(organization_id=org.id, name="Vendor A", email="a@test.example", address="500 Warehouse Rd, Suite 3")
    v2 = Vendor(organization_id=org.id, name="Vendor B", email="b@test.example", address="  500 WAREHOUSE RD, SUITE 3  ")
    db_session.add_all([v1, v2])
    db_session.commit()

    matches = find_parties_sharing_contact_info(db_session, org.id, v1)
    assert [m.name for m in matches] == ["Vendor B"]


def test_find_parties_sharing_contact_info_ignores_blank_fields(db_session, org):
    """Two vendors both missing an address must never "match" on that
    blank field - that would flag nearly every vendor with no address
    on file as suspicious, which is worse than not checking at all."""
    v1 = Vendor(organization_id=org.id, name="Vendor A", email="a@test.example", address=None)
    v2 = Vendor(organization_id=org.id, name="Vendor B", email="b@test.example", address=None)
    db_session.add_all([v1, v2])
    db_session.commit()

    assert find_parties_sharing_contact_info(db_session, org.id, v1) == []


def test_find_parties_sharing_contact_info_finds_none_for_distinct_vendors(db_session, org, vendor):
    other = Vendor(organization_id=org.id, name="Totally Different Co.", email="x@test.example", address="9 Nowhere Ave")
    db_session.add(other)
    db_session.commit()

    assert find_parties_sharing_contact_info(db_session, org.id, vendor) == []


def test_find_parties_sharing_contact_info_never_matches_across_vendor_and_customer(db_session, org, vendor, customer):
    """A vendor and a customer happening to share an email/address is
    not the pattern being checked for here (they're not "the same kind
    of party" secretly duplicated) - only ever compares like with like."""
    vendor.email = "shared@test.example"
    customer.email = "shared@test.example"
    db_session.commit()

    assert find_parties_sharing_contact_info(db_session, org.id, vendor) == []
    assert find_parties_sharing_contact_info(db_session, org.id, customer) == []


# --- Customer-side coverage: proves the same logic genuinely works for
# outgoing invoices too, not just that it doesn't crash. -----------------

def test_get_party_returns_customer_for_outgoing_invoice(db_session, org, customer):
    invoice = Invoice(
        organization_id=org.id, direction=InvoiceDirection.outgoing, invoice_number="OUT-1",
        customer_id=customer.id, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("100"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    party = get_party(invoice)
    assert party is customer


def test_compute_risk_signals_flags_new_customer_with_large_invoice(db_session, org):
    """The actual real-world case this whole extension exists for: a
    brand-new customer being extended a large amount of credit (an
    unpaid outgoing invoice IS credit extended) is exactly as worth
    flagging as a brand-new vendor billing an unusually large amount."""
    new_customer = Customer(organization_id=org.id, name="Suspicious New Client", email="s@test.example",
                             created_at=utcnow_naive())
    db_session.add(new_customer)
    db_session.commit()

    invoice = Invoice(
        organization_id=org.id, direction=InvoiceDirection.outgoing, invoice_number="OUT-BIG-1",
        customer_id=new_customer.id, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("5000"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("5000"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    signals = compute_risk_signals(db_session, invoice, new_customer)
    assert signals["party_label"] == "customer"
    assert signals["is_new_party"] is True

    risk, reasons = score_risk(signals)
    assert risk > 0
    assert any("Customer" in r for r in reasons)


def test_compute_risk_signals_org_baseline_never_mixes_incoming_and_outgoing(db_session, org, vendor):
    """A real correctness concern in the generalized baseline: incoming
    and outgoing invoices are typically very different sizes (what you
    pay for supplies vs. what you bill for finished goods) - the
    fallback baseline for a new customer must never accidentally
    average in unrelated incoming/vendor invoices, and vice versa."""
    # A large incoming/vendor invoice that should NOT count toward an
    # outgoing/customer baseline.
    db_session.add(Invoice(
        organization_id=org.id, direction=InvoiceDirection.incoming, invoice_number="VENDOR-HUGE",
        vendor_id=vendor.id, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("50000"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("50000"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    ))
    # A small established outgoing/customer baseline.
    established_customer = Customer(organization_id=org.id, name="Regular Client", email="r@test.example")
    db_session.add(established_customer)
    db_session.commit()
    db_session.add(Invoice(
        organization_id=org.id, direction=InvoiceDirection.outgoing, invoice_number="OUT-NORMAL",
        customer_id=established_customer.id, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("100"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    ))
    db_session.commit()

    new_customer = Customer(organization_id=org.id, name="New Client", email="n@test.example", created_at=utcnow_naive())
    db_session.add(new_customer)
    db_session.commit()

    test_invoice = Invoice(
        organization_id=org.id, direction=InvoiceDirection.outgoing, invoice_number="OUT-NEW-1",
        customer_id=new_customer.id, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("500"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("500"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    db_session.add(test_invoice)
    db_session.commit()
    db_session.refresh(test_invoice)

    signals = compute_risk_signals(db_session, test_invoice, new_customer)
    # If the $50,000 vendor invoice leaked into the baseline, this ratio
    # would be tiny (500 / ~25050) instead of 500/100 = 5.0.
    assert signals["amount_ratio"] == 5.0


# --- Borderline-case reasoning-model deliberation: advisory only, never
# a decision. See deliberate_on_borderline_case()'s own docstring for why
# this split exists and BORDERLINE_BAND for why these particular scores. --

def test_deliberate_skips_the_reasoning_call_entirely_outside_the_band(monkeypatch):
    """Reasoning models are slow - the whole point of gating on
    BORDERLINE_BAND is to never pay that cost on a clear-cut score.
    Asserts the LLM is never even called, not just that its result is
    discarded."""
    def _must_not_be_called(*a, **kw):
        raise AssertionError("reason() should not be called for a non-borderline score")
    monkeypatch.setattr(fraud_risk_agent, "reason", _must_not_be_called)

    assert deliberate_on_borderline_case(0.1, ["No anomalies detected."]) is None
    assert deliberate_on_borderline_case(0.95, ["Everything is wrong."]) is None


def test_deliberate_calls_the_reasoning_model_inside_the_band(monkeypatch):
    captured = {}

    def _fake_reason(messages, temperature=0.0):
        captured["prompt"] = messages[0]["content"]
        return {"thinking": "Weighing signal A against signal B...", "content": "RECOMMENDATION: closer look warranted"}

    monkeypatch.setattr(fraud_risk_agent, "reason", _fake_reason)

    mid_band_score = sum(BORDERLINE_BAND) / 2
    result = deliberate_on_borderline_case(mid_band_score, ["Amount is 3.2x this vendor's average invoice."])
    assert result["thinking"] == "Weighing signal A against signal B..."
    assert "3.2x" in captured["prompt"]


def test_deliberate_falls_back_to_none_when_llm_unavailable(monkeypatch):
    def _raise(*a, **kw):
        raise LLMUnavailableError("model not pulled")
    monkeypatch.setattr(fraud_risk_agent, "reason", _raise)

    assert deliberate_on_borderline_case(BORDERLINE_BAND[0] + 0.01, ["some reason"]) is None


def test_deliberate_stream_yields_nothing_outside_the_band(monkeypatch):
    """Same gating as the non-streaming version, for the function the web
    route actually calls."""
    def _must_not_be_called(*a, **kw):
        raise AssertionError("reason_stream() should not be called for a non-borderline score")
    monkeypatch.setattr(fraud_risk_agent, "reason_stream", _must_not_be_called)

    assert list(fraud_risk_agent.deliberate_on_borderline_case_stream(0.1, ["No anomalies detected."])) == []
    assert list(fraud_risk_agent.deliberate_on_borderline_case_stream(0.95, ["Everything is wrong."])) == []


def test_run_fraud_check_never_calls_the_reasoning_model_or_stores_a_trace(db_session, org, vendor, monkeypatch):
    """The real design decision this project's own measurements forced:
    a live run against deepseek-r1:8b took 3-6+ minutes, so
    run_fraud_check() (on the invoice-creation path) must never call the
    reasoning model at all, regardless of score - that review only ever
    happens on-demand via deliberate_on_borderline_case_stream(), from a
    button a human clicks. Pins that down for every score band, not just
    the borderline one, since a regression here would silently reintroduce
    a multi-minute block on saving an invoice."""
    monkeypatch.setattr(fraud_risk_agent, "explain_risk_with_llm", lambda *a, **kw: "mocked explanation")

    def _must_not_be_called(*a, **kw):
        raise AssertionError("run_fraud_check() must never call the reasoning model directly")
    monkeypatch.setattr(fraud_risk_agent, "reason", _must_not_be_called)
    monkeypatch.setattr(fraud_risk_agent, "reason_stream", _must_not_be_called)

    def make_invoice(number):
        invoice = Invoice(
            organization_id=org.id, direction=InvoiceDirection.incoming, invoice_number=number,
            vendor_id=vendor.id, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
            subtotal=Decimal("100"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100"),
            payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
        )
        db_session.add(invoice)
        db_session.commit()
        db_session.refresh(invoice)
        return invoice

    for score in (0.1, sum(BORDERLINE_BAND) / 2, 0.95):
        monkeypatch.setattr(fraud_risk_agent, "score_risk", lambda signals, s=score: (s, ["some reason"]))
        flag = run_fraud_check(db_session, make_invoice(f"RT-{score}"))
        assert flag.reasoning_trace is None
        assert float(flag.risk_score) == score
