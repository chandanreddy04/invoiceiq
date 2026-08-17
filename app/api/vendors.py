from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.models import Vendor, Organization
from app.schemas.party import VendorCreate, VendorRead

router = APIRouter(prefix="/vendors", tags=["vendors"])

DEFAULT_ORG_ID = 1


@router.post("", response_model=VendorRead)
def create_vendor(payload: VendorCreate, db: Session = Depends(get_db)):
    org = db.get(Organization, DEFAULT_ORG_ID)
    if org is None:
        raise HTTPException(status_code=400, detail="Default organization not seeded. Run scripts/seed_demo_data.py first.")
    vendor = Vendor(organization_id=DEFAULT_ORG_ID, **payload.model_dump())
    db.add(vendor)
    db.commit()
    db.refresh(vendor)
    return vendor


@router.get("", response_model=list[VendorRead])
def list_vendors(db: Session = Depends(get_db)):
    return db.query(Vendor).filter(Vendor.organization_id == DEFAULT_ORG_ID).all()


@router.get("/{vendor_id}", response_model=VendorRead)
def get_vendor(vendor_id: int, db: Session = Depends(get_db)):
    vendor = db.get(Vendor, vendor_id)
    if vendor is None:
        raise HTTPException(status_code=404, detail="Vendor not found")
    return vendor
