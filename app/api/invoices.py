"""
Invoice CRUD JSON API. All the actual logic lives in
app/services/invoice_service.py so it's shared with the HTML pages
in app/web/routes.py - this file is just the HTTP layer on top.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.models import InvoiceDirection, PaymentStatus
from app.schemas.invoice import InvoiceCreate, InvoiceRead, InvoiceUpdate
from app.services import invoice_service
from app.services.validation_service import InvoiceValidationError

router = APIRouter(prefix="/invoices", tags=["invoices"])

DEFAULT_ORG_ID = 1


@router.post("", response_model=InvoiceRead, status_code=201)
def create_invoice(payload: InvoiceCreate, db: Session = Depends(get_db)):
    try:
        return invoice_service.create_invoice(db, DEFAULT_ORG_ID, payload)
    except InvoiceValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("", response_model=list[InvoiceRead])
def list_invoices(
    direction: InvoiceDirection | None = None,
    payment_status: PaymentStatus | None = None,
    db: Session = Depends(get_db),
):
    return invoice_service.list_invoices(db, DEFAULT_ORG_ID, direction, payment_status)


@router.get("/{invoice_id}", response_model=InvoiceRead)
def get_invoice(invoice_id: int, db: Session = Depends(get_db)):
    invoice = invoice_service.get_invoice(db, invoice_id)
    if invoice is None:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return invoice


@router.put("/{invoice_id}", response_model=InvoiceRead)
def update_invoice(invoice_id: int, payload: InvoiceUpdate, db: Session = Depends(get_db)):
    invoice = invoice_service.get_invoice(db, invoice_id)
    if invoice is None:
        raise HTTPException(status_code=404, detail="Invoice not found")
    try:
        return invoice_service.update_invoice(db, invoice, payload)
    except InvoiceValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.delete("/{invoice_id}", status_code=204)
def delete_invoice(invoice_id: int, db: Session = Depends(get_db)):
    invoice = invoice_service.get_invoice(db, invoice_id)
    if invoice is None:
        raise HTTPException(status_code=404, detail="Invoice not found")
    invoice_service.delete_invoice(db, invoice)
    return None
