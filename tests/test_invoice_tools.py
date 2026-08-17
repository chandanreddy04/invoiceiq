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
    assert summary["total_payable_outstanding"] == Decimal("300")
    assert summary["total_receivable_outstanding"] == Decimal("50")
    assert summary["count_overdue"] == 1
    assert summary["overdue_total"] == Decimal("200")
    assert summary["count_invoices"] == 3
