"""Pydantic schemas for customers and vendors - the parties invoices
are sent to or received from."""

from pydantic import BaseModel, ConfigDict


class CustomerCreate(BaseModel):
    name: str
    email: str | None = None
    address: str | None = None


class CustomerRead(CustomerCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int


class VendorCreate(BaseModel):
    name: str
    email: str | None = None
    address: str | None = None
    tax_id: str | None = None


class VendorRead(VendorCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int
