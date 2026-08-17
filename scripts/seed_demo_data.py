"""
Creates the Maple Street Bakery Supply Co. organization, demo
customers/vendors, and two demo login users (owner + bookkeeper -
Section 12 RBAC). Deliberately small and manual - the full synthetic
invoice generator lives in scripts/generate_synthetic_data.py.

Safe to re-run: each section checks for its own existing data before
creating anything, so it won't create duplicates - this matters
because Phase 11 added users to a database that, in most setups,
already has the organization/customers/vendors from earlier runs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.session import SessionLocal, init_db
from app.models.models import Organization, Customer, Vendor, User, UserRole
from app.security.auth import hash_password

DEMO_OWNER_PASSWORD = "owner-demo-pass123"
DEMO_BOOKKEEPER_PASSWORD = "bookkeeper-demo-pass123"


def seed_org_and_parties(db) -> Organization:
    org = db.get(Organization, 1)
    if org is not None:
        print("Organization already seeded (id=1). Skipping customers/vendors.")
        return org

    org = Organization(name="Maple Street Bakery Supply Co.")
    db.add(org)
    db.flush()  # assigns org.id without committing yet

    customers = [
        Customer(organization_id=org.id, name="Riverside Cafe", email="ap@riversidecafe.example", address="12 River St"),
        Customer(organization_id=org.id, name="The Corner Bistro", email="billing@cornerbistro.example", address="88 Corner Ave"),
    ]
    vendors = [
        Vendor(organization_id=org.id, name="Golden Grain Milling", email="sales@goldengrain.example", address="500 Mill Rd"),
        Vendor(organization_id=org.id, name="Sunrise Packaging Co.", email="orders@sunrisepack.example", address="21 Industrial Way"),
    ]
    db.add_all(customers + vendors)
    db.commit()

    print(f"Seeded organization '{org.name}' (id={org.id})")
    print(f"  Customers: {[c.name for c in customers]}")
    print(f"  Vendors:   {[v.name for v in vendors]}")
    return org


def seed_users(db, org: Organization) -> None:
    if db.query(User).count() > 0:
        print("Users already seeded. Skipping.")
        return

    owner = User(
        organization_id=org.id, email="owner@maplestreet.example",
        password_hash=hash_password(DEMO_OWNER_PASSWORD), role=UserRole.owner,
    )
    bookkeeper = User(
        organization_id=org.id, email="bookkeeper@maplestreet.example",
        password_hash=hash_password(DEMO_BOOKKEEPER_PASSWORD), role=UserRole.bookkeeper,
    )
    db.add_all([owner, bookkeeper])
    db.commit()

    print("\nSeeded demo login users (CHANGE THESE before any real deployment):")
    print(f"  Owner:      {owner.email} / {DEMO_OWNER_PASSWORD}  (can approve/reject)")
    print(f"  Bookkeeper: {bookkeeper.email} / {DEMO_BOOKKEEPER_PASSWORD}  (cannot approve/reject)")


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        org = seed_org_and_parties(db)
        seed_users(db, org)
    finally:
        db.close()


if __name__ == "__main__":
    main()
