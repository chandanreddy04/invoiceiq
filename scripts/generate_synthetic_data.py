"""
Synthetic dataset generator (Section 31). Every invoice here goes
through the real invoice_service.create_invoice() path - the same
code the UI and API use - so the Fraud/Risk and Classification Agents
genuinely run on each one. Nothing here is pre-labeled or faked.

Deliberately a modest count (not hundreds): each incoming invoice
costs ~2 real LLM calls (~30-40s combined on this CPU-only local
model), so this favors deliberate variety over raw volume - normal,
overdue, oversized, new-vendor, duplicate-rejected, and multi-category
invoices, covering every case Section 31 asks for without a 20+ minute
run. Safe to re-run: duplicate invoice_numbers are skipped, not
recreated.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.session import SessionLocal, init_db
from app.models.models import Organization, Vendor, Customer, PaymentStatus
from app.schemas.invoice import InvoiceCreate, InvoiceItemCreate, InvoiceUpdate
from app.services import invoice_service
from app.services.validation_service import InvoiceValidationError

ORG_ID = 1
TODAY = date.today()


def get_or_create_vendor(db, org_id, name, email, address):
    v = db.query(Vendor).filter(Vendor.organization_id == org_id, Vendor.name == name).first()
    if v:
        return v
    v = Vendor(organization_id=org_id, name=name, email=email, address=address)
    db.add(v)
    db.commit()
    db.refresh(v)
    return v


def get_or_create_customer(db, org_id, name, email, address):
    c = db.query(Customer).filter(Customer.organization_id == org_id, Customer.name == name).first()
    if c:
        return c
    c = Customer(organization_id=org_id, name=name, email=email, address=address)
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


def create(db, label, direction, vendor_id, customer_id, invoice_number, invoice_date, due_date, items, tax=0, mark_paid=False):
    payload = InvoiceCreate(
        direction=direction, invoice_number=invoice_number, vendor_id=vendor_id, customer_id=customer_id,
        invoice_date=invoice_date, due_date=due_date, tax=tax, discount=0, currency="USD",
        items=[InvoiceItemCreate(**it) for it in items],
    )
    try:
        invoice = invoice_service.create_invoice(db, ORG_ID, payload)
    except InvoiceValidationError as e:
        print(f"  [{label}] REJECTED as expected: {e}")
        return None

    if mark_paid:
        invoice_service.update_invoice(db, invoice, InvoiceUpdate(payment_status=PaymentStatus.paid))

    print(f"  [{label}] invoice #{invoice.id} '{invoice_number}' -> risk={invoice.risk_score}, status={invoice.invoice_status.value}")
    return invoice


def main():
    init_db()
    db = SessionLocal()

    org = db.get(Organization, ORG_ID)
    if org is None:
        print("Run scripts/seed_demo_data.py first.")
        return

    golden_grain = get_or_create_vendor(db, ORG_ID, "Golden Grain Milling", "sales@goldengrain.example", "500 Mill Rd")
    sunrise = get_or_create_vendor(db, ORG_ID, "Sunrise Packaging Co.", "orders@sunrisepack.example", "21 Industrial Way")
    city_electric = get_or_create_vendor(db, ORG_ID, "City Electric Utility", "billing@cityelectric.example", "1 Power Plant Rd")
    new_vendor = get_or_create_vendor(db, ORG_ID, "QuickCash Wholesale LLC", "info@quickcashwholesale.example", "Unknown")

    riverside = get_or_create_customer(db, ORG_ID, "Riverside Cafe", "ap@riversidecafe.example", "12 River St")
    downtown = get_or_create_customer(db, ORG_ID, "Downtown Catering Co.", "billing@downtowncatering.example", "77 Main St")

    print("Generating synthetic invoices...\n")

    # 1. Normal, already paid
    create(db, "normal-paid", "incoming", golden_grain.id, None, "GGM-1101",
           TODAY - timedelta(days=20), TODAY + timedelta(days=10),
           [{"description": "Flour - bulk 50lb", "quantity": 25, "unit_price": 41.00}],
           tax=10, mark_paid=True)

    # 2. Normal, upcoming due date, unpaid
    create(db, "normal-upcoming", "incoming", sunrise.id, None, "SPC-1102",
           TODAY - timedelta(days=2), TODAY + timedelta(days=28),
           [{"description": "Packing tape (roll)", "quantity": 30, "unit_price": 2.75},
            {"description": "Cardboard boxes (case)", "quantity": 15, "unit_price": 4.00}],
           tax=8)

    # 3. Overdue
    create(db, "overdue", "incoming", golden_grain.id, None, "GGM-1103",
           TODAY - timedelta(days=45), TODAY - timedelta(days=15),
           [{"description": "Sugar - bulk 25lb", "quantity": 20, "unit_price": 13.50}],
           tax=6)

    # 4. Different expense category: utilities
    create(db, "utilities", "incoming", city_electric.id, None, "CE-2026-08",
           TODAY - timedelta(days=5), TODAY + timedelta(days=25),
           [{"description": "Monthly electricity usage - bakery facility", "quantity": 1, "unit_price": 640.00}])

    # 5. Unusually large amount from an established vendor - should trigger fraud flag
    create(db, "oversized-established-vendor", "incoming", golden_grain.id, None, "GGM-1105",
           TODAY, TODAY + timedelta(days=30),
           [{"description": "Emergency flour restock - large order", "quantity": 1, "unit_price": 15000.00}])

    # 6. Brand-new vendor + large amount - should trigger fraud flag (new-vendor signal)
    create(db, "new-vendor-large", "incoming", new_vendor.id, None, "QC-0001",
           TODAY, TODAY + timedelta(days=15),
           [{"description": "Bulk dry goods pallet", "quantity": 1, "unit_price": 7800.00}])

    # 7. Duplicate of invoice #1 from Phase 1 testing (GGM-0847) - should be REJECTED
    create(db, "duplicate-attempt", "incoming", golden_grain.id, None, "GGM-0847",
           TODAY, TODAY + timedelta(days=30),
           [{"description": "This should never be saved", "quantity": 1, "unit_price": 999.00}])

    # 8. Outgoing, paid
    create(db, "outgoing-paid", "outgoing", None, riverside.id, "MSB-2201",
           TODAY - timedelta(days=25), TODAY - timedelta(days=5),
           [{"description": "Wholesale pastry order", "quantity": 1, "unit_price": 340.00}],
           mark_paid=True)

    # 9. Outgoing, overdue, new customer
    create(db, "outgoing-overdue", "outgoing", None, downtown.id, "MSB-2202",
           TODAY - timedelta(days=50), TODAY - timedelta(days=20),
           [{"description": "Catering order - bread and pastries", "quantity": 1, "unit_price": 610.00}])

    # 10. Professional services category
    create(db, "professional-services", "incoming", sunrise.id, None, "SPC-1110",
           TODAY - timedelta(days=3), TODAY + timedelta(days=27),
           [{"description": "Annual accounting and bookkeeping consultation", "quantity": 1, "unit_price": 350.00}])

    db.close()
    print("\nDone. Visit /web/dashboard to see the results.")


if __name__ == "__main__":
    main()
