from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.models import Customer, Organization
from app.schemas.party import CustomerCreate, CustomerRead

router = APIRouter(prefix="/customers", tags=["customers"])

DEFAULT_ORG_ID = 1  # single-organization demo for Phase 1 - see seed_demo_data.py


@router.post("", response_model=CustomerRead)
def create_customer(payload: CustomerCreate, db: Session = Depends(get_db)):
    org = db.get(Organization, DEFAULT_ORG_ID)
    if org is None:
        raise HTTPException(status_code=400, detail="Default organization not seeded. Run scripts/seed_demo_data.py first.")
    customer = Customer(organization_id=DEFAULT_ORG_ID, **payload.model_dump())
    db.add(customer)
    db.commit()
    db.refresh(customer)
    return customer


@router.get("", response_model=list[CustomerRead])
def list_customers(db: Session = Depends(get_db)):
    return db.query(Customer).filter(Customer.organization_id == DEFAULT_ORG_ID).all()


@router.get("/{customer_id}", response_model=CustomerRead)
def get_customer(customer_id: int, db: Session = Depends(get_db)):
    customer = db.get(Customer, customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer
