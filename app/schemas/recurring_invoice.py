from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field

from app.models.models import RecurringFrequency
from app.schemas.invoice import InvoiceItemCreate


class RecurringInvoiceCreate(BaseModel):
    customer_id: int
    name: str
    frequency: RecurringFrequency
    next_run_date: date
    due_days: int = 30
    currency: str = Field(default="USD", min_length=3, max_length=3)
    tax: Decimal = Decimal("0")
    discount: Decimal = Decimal("0")
    payment_terms: str = "Net 30"
    items: list[InvoiceItemCreate]
