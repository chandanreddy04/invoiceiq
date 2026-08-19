"""
Tests for the Fraud/Risk Agent's REASONING layer specifically (not
the LLM layer - see the agent's own docstring on why that split
exists). score_risk() is pure and gets tested directly with synthetic
signal dicts; compute_risk_signals() needs a database, so those tests
use the db_session/vendor fixtures from conftest.
"""

from datetime import date, timedelta
from decimal import Decimal

from app.agents.fraud_risk_agent import (
    score_risk, compute_risk_signals, find_vendors_sharing_contact_info, HIGH_RISK_THRESHOLD,
)
from app.models.models import Invoice, InvoiceDirection, InvoiceStatus, PaymentStatus, Vendor
from app.utils.time import utcnow_naive


def make_signals(**overrides):
    base = {
        "is_new_vendor": False,
        "vendor_age_days": 400,
        "amount_ratio": None,
        "amount_ratio_basis": None,
        "max_invoice_number_similarity": 0.0,
        "similar_invoice_number": None,
        "shared_contact_vendors": [],
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
    risk, reasons = score_risk(make_signals(amount_ratio=5.0, amount_ratio_basis="vendor"))
    assert risk == 0.45
    assert risk < HIGH_RISK_THRESHOLD
    assert "5.0x" in reasons[0]


def test_new_vendor_plus_high_amount_crosses_threshold():
    risk, _ = score_risk(make_signals(amount_ratio=5.0, amount_ratio_basis="vendor", is_new_vendor=True, vendor_age_days=1))
    assert risk >= HIGH_RISK_THRESHOLD


def test_moderate_amount_ratio_scores_less_than_high_ratio():
    low_risk, _ = score_risk(make_signals(amount_ratio=2.0, amount_ratio_basis="vendor"))
    high_risk, _ = score_risk(make_signals(amount_ratio=5.0, amount_ratio_basis="vendor"))
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
        amount_ratio=50.0, amount_ratio_basis="vendor", is_new_vendor=True, vendor_age_days=0,
        max_invoice_number_similarity=1.0, similar_invoice_number="X",
    ))
    assert risk == 1.0


def test_org_wide_fallback_flags_new_vendor_first_large_invoice(db_session, org, vendor):
    """Regression test for the real bug found during Phase 9 synthetic
    data testing: a brand-new vendor's FIRST invoice has no vendor
    history to compute a ratio against, so without the org-wide
    fallback it only ever gets the flat new-vendor signal - even if
    the amount is wildly out of line with everything else this
    business normally sees."""
    from app.models.models import Vendor

    # Establish a "normal" baseline: a handful of small invoices from a
    # long-established vendor.
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

    # A brand-new vendor's very first invoice, 20x the established baseline.
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
    from app.models.models import Vendor

    established = Vendor(organization_id=org.id, name="Established Vendor", email="e@test.example",
                          created_at=utcnow_naive() - timedelta(days=400))
    db_session.add(established)
    db_session.commit()

    # A normal invoice and one already flagged as high-risk.
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


def test_shared_contact_vendors_adds_risk_and_names_the_other_vendor():
    risk, reasons = score_risk(make_signals(shared_contact_vendors=["Golden Grain Milling"]))
    assert risk == 0.35
    assert "Golden Grain Milling" in reasons[0]


def test_shared_contact_vendors_stacks_with_other_signals():
    alone, _ = score_risk(make_signals(shared_contact_vendors=["Other Co."]))
    stacked, _ = score_risk(make_signals(shared_contact_vendors=["Other Co."], is_new_vendor=True, vendor_age_days=1))
    assert stacked > alone


def test_find_vendors_sharing_contact_info_matches_on_email(db_session, org):
    v1 = Vendor(organization_id=org.id, name="Acme Supply", email="billing@acme.example", address="1 Main St")
    v2 = Vendor(organization_id=org.id, name="Acme Supply LLC", email="billing@acme.example", address="2 Other St")
    db_session.add_all([v1, v2])
    db_session.commit()

    matches = find_vendors_sharing_contact_info(db_session, org.id, v1)
    assert [m.name for m in matches] == ["Acme Supply LLC"]


def test_find_vendors_sharing_contact_info_address_match_is_case_and_whitespace_insensitive(db_session, org):
    v1 = Vendor(organization_id=org.id, name="Vendor A", email="a@test.example", address="500 Warehouse Rd, Suite 3")
    v2 = Vendor(organization_id=org.id, name="Vendor B", email="b@test.example", address="  500 WAREHOUSE RD, SUITE 3  ")
    db_session.add_all([v1, v2])
    db_session.commit()

    matches = find_vendors_sharing_contact_info(db_session, org.id, v1)
    assert [m.name for m in matches] == ["Vendor B"]


def test_find_vendors_sharing_contact_info_ignores_blank_fields(db_session, org):
    """Two vendors both missing an address must never "match" on that
    blank field - that would flag nearly every vendor with no address
    on file as suspicious, which is worse than not checking at all."""
    v1 = Vendor(organization_id=org.id, name="Vendor A", email="a@test.example", address=None)
    v2 = Vendor(organization_id=org.id, name="Vendor B", email="b@test.example", address=None)
    db_session.add_all([v1, v2])
    db_session.commit()

    assert find_vendors_sharing_contact_info(db_session, org.id, v1) == []


def test_find_vendors_sharing_contact_info_finds_none_for_distinct_vendors(db_session, org, vendor):
    other = Vendor(organization_id=org.id, name="Totally Different Co.", email="x@test.example", address="9 Nowhere Ave")
    db_session.add(other)
    db_session.commit()

    assert find_vendors_sharing_contact_info(db_session, org.id, vendor) == []
