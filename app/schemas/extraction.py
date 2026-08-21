"""
The shape we ask the LLM to fill in. Passing this schema's JSON Schema
straight to Ollama's `format` parameter forces the model's output to
conform to it - no fragile "please respond with JSON" prompting and
hoping, no free-text parsing of the reply.
"""

from decimal import Decimal
from pydantic import BaseModel, Field


class LLMExtractedItem(BaseModel):
    description: str
    quantity: float = 1
    unit_price: float = 0


class LLMExtractedInvoice(BaseModel):
    vendor_name: str | None = Field(default=None, description="The vendor/seller's business name issuing this invoice - not the buyer's name")
    vendor_address: str | None = Field(default=None, description="The vendor's own mailing address, if printed on the invoice - not the buyer's address")
    vendor_tax_id: str | None = Field(default=None, description="The vendor's Tax ID, EIN, VAT number, or business license number, if printed on the invoice")
    invoice_number: str | None = Field(default=None, description="The invoice's own reference number")
    invoice_date: str | None = Field(default=None, description="ISO format YYYY-MM-DD")
    due_date: str | None = Field(default=None, description="ISO format YYYY-MM-DD")
    currency: str = "USD"
    tax: float = 0
    discount: float = 0
    line_items: list[LLMExtractedItem] = Field(default_factory=list)
