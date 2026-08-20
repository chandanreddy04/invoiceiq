"""
Pydantic schemas for invoices - this is the Section 10 invoice data
model made real and enforced. Two important design choices:

1. Money fields use Decimal, never float - floating point arithmetic
   can misrepresent amounts like 0.1 + 0.2, which is not acceptable
   for financial totals.
2. `subtotal` and `total` are never accepted as input from the client
   on create - they are always computed server-side from line items
   (see app/services/validation_service.py). Trusting a client-supplied
   total would let a bug (or a malicious request) desynchronize the
   stated total from the actual line items.
"""

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.models.models import InvoiceDirection, InvoiceStatus, PaymentStatus


class InvoiceItemCreate(BaseModel):
    description: str
    quantity: Decimal = Decimal("1")
    unit_price: Decimal


class InvoiceItemRead(InvoiceItemCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int
    line_total: Decimal
    category: str | None = None


class InvoiceCreate(BaseModel):
    direction: InvoiceDirection
    invoice_number: str
    vendor_id: int | None = None
    customer_id: int | None = None
    invoice_date: date
    due_date: date
    tax: Decimal = Decimal("0")
    discount: Decimal = Decimal("0")
    currency: str = Field(default="USD", min_length=3, max_length=3)
    payment_terms: str = "Net 30"
    items: list[InvoiceItemCreate]
    source_pdf_filename: str | None = None  # set only by the Upload flow - never client-supplied elsewhere


class InvoiceUpdate(BaseModel):
    invoice_number: str | None = None
    vendor_id: int | None = None
    customer_id: int | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    tax: Decimal | None = None
    discount: Decimal | None = None
    currency: str | None = None
    payment_terms: str | None = None
    payment_status: PaymentStatus | None = None
    invoice_status: InvoiceStatus | None = None
    items: list[InvoiceItemCreate] | None = None


class InvoiceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    organization_id: int
    direction: InvoiceDirection
    invoice_number: str
    vendor_id: int | None
    customer_id: int | None
    invoice_date: date
    due_date: date
    subtotal: Decimal
    tax: Decimal
    discount: Decimal
    total: Decimal
    currency: str
    payment_terms: str
    payment_status: PaymentStatus
    invoice_status: InvoiceStatus
    confidence_score: Decimal | None
    risk_score: Decimal | None
    source_pdf_filename: str | None
    public_token: str | None
    created_at: datetime
    items: list[InvoiceItemRead]
