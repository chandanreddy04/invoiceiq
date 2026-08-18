"""
Tests for the fixed tool functions agents are allowed to call
(Section 18). All plain SQLAlchemy queries, no LLM.
"""

from datetime import date, timedelta
from decimal import Decimal

from app.models.models import Invoice, InvoiceDirection, InvoiceStatus, PaymentStatus
from app.tools.invoice_tools import (
    search_invoices, get_overdue_invoices, check_duplicate_invoice, generate_financial_summary,
)


def make_invoice(org_id, vendor_id, customer_id, direction, invoice_number, total, due_date,
                  payment_status=PaymentStatus.unpaid, direction_days_ago=0):
    return Invoice(
        organization_id=org_id, direction=direction, invoice_number=invoice_number,
        vendor_id=vendor_id, customer_id=customer_id,
        invoice_date=date.today() - timedelta(days=direction_days_ago), due_date=due_date,
        subtotal=Decimal(str(total)), tax=Decimal("0"), discount=Decimal("0"), total=Decimal(str(total)),
        payment_status=payment_status, invoice_status=InvoiceStatus.validated,
    )


def test_search_invoices_filters_by_direction(db_session, org, vendor, customer):
    db_session.add(make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "IN-1", 100, date.today() + timedelta(days=10)))
    db_session.add(make_invoice(org.id, None, customer.id, InvoiceDirection.outgoing, "OUT-1", 200, date.today() + timedelta(days=10)))
    db_session.commit()

    incoming = search_invoices(db_session, org.id, direction="incoming")
    assert len(incoming) == 1
    assert incoming[0].invoice_number == "IN-1"


def test_search_invoices_filters_by_payment_status(db_session, org, vendor):
    db_session.add(make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "PAID-1", 100, date.today(), payment_status=PaymentStatus.paid))
    db_session.add(make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "UNPAID-1", 100, date.today(), payment_status=PaymentStatus.unpaid))
    db_session.commit()

    paid = search_invoices(db_session, org.id, payment_status="paid")
    assert len(paid) == 1
    assert paid[0].invoice_number == "PAID-1"


def test_search_invoices_min_max_total(db_session, org, vendor):
    db_session.add(make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "CHEAP", 50, date.today()))
    db_session.add(make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "MID", 500, date.today()))
    db_session.add(make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "EXPENSIVE", 5000, date.today()))
    db_session.commit()

    result = search_invoices(db_session, org.id, min_total=100, max_total=1000)
    assert [i.invoice_number for i in result] == ["MID"]


def test_get_overdue_invoices_excludes_paid(db_session, org, vendor):
    db_session.add(make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "OVERDUE-UNPAID", 100,
                                 date.today() - timedelta(days=5), payment_status=PaymentStatus.unpaid))
    db_session.add(make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "OVERDUE-BUT-PAID", 100,
                                 date.today() - timedelta(days=5), payment_status=PaymentStatus.paid))
    db_session.add(make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "NOT-OVERDUE", 100,
                                 date.today() + timedelta(days=5), payment_status=PaymentStatus.unpaid))
    db_session.commit()

    result = get_overdue_invoices(db_session, org.id)
    assert [i.invoice_number for i in result] == ["OVERDUE-UNPAID"]


def test_check_duplicate_invoice_detects_same_vendor_same_number(db_session, org, vendor):
    existing = make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "DUP-1", 100, date.today())
    db_session.add(existing)
    db_session.commit()

    dup = check_duplicate_invoice(db_session, org.id, vendor.id, "DUP-1")
    assert dup is not None
    assert dup.invoice_number == "DUP-1"


def test_check_duplicate_invoice_allows_same_number_different_vendor(db_session, org, vendor):
    from app.models.models import Vendor
    other_vendor = Vendor(organization_id=org.id, name="Other Vendor", email="o@test.example")
    db_session.add(other_vendor)
    db_session.commit()

    db_session.add(make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "SHARED-NUM", 100, date.today()))
    db_session.commit()

    dup = check_duplicate_invoice(db_session, org.id, other_vendor.id, "SHARED-NUM")
    assert dup is None


def test_check_duplicate_invoice_excludes_self_on_update(db_session, org, vendor):
    existing = make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "SELF-1", 100, date.today())
    db_session.add(existing)
    db_session.commit()
    db_session.refresh(existing)

    dup = check_duplicate_invoice(db_session, org.id, vendor.id, "SELF-1", exclude_invoice_id=existing.id)
    assert dup is None


def test_generate_financial_summary_math(db_session, org, vendor, customer):
    db_session.add(make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "PAY-1", 100, date.today() + timedelta(days=10)))
    db_session.add(make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "PAY-2", 200, date.today() - timedelta(days=5)))  # overdue
    db_session.add(make_invoice(org.id, None, customer.id, InvoiceDirection.outgoing, "REC-1", 50, date.today() + timedelta(days=10)))
    db_session.commit()

    summary = generate_financial_summary(db_session, org.id)
    assert summary["total_payable_outstanding_by_currency"] == {"USD": Decimal("300")}
    assert summary["total_receivable_outstanding_by_currency"] == {"USD": Decimal("50")}
    assert summary["count_overdue"] == 1
    assert summary["overdue_total_by_currency"] == {"USD": Decimal("200")}
    assert summary["count_invoices"] == 3


def test_generate_financial_summary_keeps_currencies_separate(db_session, org, vendor):
    """Regression test for a real bug: totals used to sum every invoice's
    .total regardless of currency, so a $100 USD invoice and a EUR 100
    invoice would have silently added up to a meaningless "200"."""
    usd_invoice = make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "USD-1", 100, date.today() + timedelta(days=10))
    eur_invoice = make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "EUR-1", 100, date.today() + timedelta(days=10))
    eur_invoice.currency = "EUR"
    db_session.add_all([usd_invoice, eur_invoice])
    db_session.commit()

    summary = generate_financial_summary(db_session, org.id)
    assert summary["total_payable_outstanding_by_currency"] == {"USD": Decimal("100"), "EUR": Decimal("100")}


def test_search_invoices_filters_by_party_name_vendor(db_session, org, vendor, customer):
    """Regression test for the real gap found live: the AI Assistant
    had no way to filter by vendor/customer name at all until this."""
    from app.models.models import Vendor
    other_vendor = Vendor(organization_id=org.id, name="Sunrise Packaging Co.", email="s@test.example")
    db_session.add(other_vendor)
    db_session.commit()

    db_session.add(make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "V1-1", 100, date.today()))
    db_session.add(make_invoice(org.id, other_vendor.id, None, InvoiceDirection.incoming, "V2-1", 200, date.today()))
    db_session.commit()

    # `vendor` fixture is named "Test Vendor" - partial, case-insensitive match
    result = search_invoices(db_session, org.id, party_name="test vendor")
    assert [i.invoice_number for i in result] == ["V1-1"]

    result2 = search_invoices(db_session, org.id, party_name="sunrise")
    assert [i.invoice_number for i in result2] == ["V2-1"]


def test_search_invoices_filters_by_party_name_customer(db_session, org, vendor, customer):
    db_session.add(make_invoice(org.id, None, customer.id, InvoiceDirection.outgoing, "C1-1", 100, date.today()))
    db_session.commit()

    result = search_invoices(db_session, org.id, party_name="test customer")
    assert [i.invoice_number for i in result] == ["C1-1"]

    result_none = search_invoices(db_session, org.id, party_name="nonexistent company")
    assert result_none == []


def test_search_invoices_filters_by_invoice_number_partial(db_session, org, vendor):
    db_session.add(make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "GGM-0847", 100, date.today()))
    db_session.add(make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "SPC-1102", 100, date.today()))
    db_session.commit()

    result = search_invoices(db_session, org.id, invoice_number="ggm")
    assert [i.invoice_number for i in result] == ["GGM-0847"]


def test_search_invoices_filters_by_category(db_session, org, vendor):
    from app.models.models import InvoiceItem

    inv1 = make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "CAT-1", 100, date.today())
    inv1.items = [InvoiceItem(description="Flour", quantity=Decimal("1"), unit_price=Decimal("100"),
                               line_total=Decimal("100"), category="Raw Ingredients")]
    inv2 = make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "CAT-2", 50, date.today())
    inv2.items = [InvoiceItem(description="Electricity", quantity=Decimal("1"), unit_price=Decimal("50"),
                               line_total=Decimal("50"), category="Utilities")]
    db_session.add_all([inv1, inv2])
    db_session.commit()

    result = search_invoices(db_session, org.id, category="utilities")
    assert [i.invoice_number for i in result] == ["CAT-2"]


def test_search_invoices_filters_by_risky_only(db_session, org, vendor):
    normal = make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "RISK-LOW", 100, date.today())
    normal.risk_score = Decimal("0.3")
    risky = make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "RISK-HIGH", 100, date.today())
    risky.risk_score = Decimal("0.75")
    unassessed = make_invoice(org.id, vendor.id, None, InvoiceDirection.incoming, "RISK-NONE", 100, date.today())
    db_session.add_all([normal, risky, unassessed])
    db_session.commit()

    result = search_invoices(db_session, org.id, risky_only=True)
    assert [i.invoice_number for i in result] == ["RISK-HIGH"]
