"""
ORM models for Phase 1: the core invoicing tables from the Section 9
database design. Auth (users/roles) and the AI-related tables
(agent_actions, fraud_flags, approval_requests, etc.) are deliberately
left out here - they get added in the phases that actually need them,
so each phase's schema changes are easy to see and explain.

We use plain autoincrementing integer IDs rather than UUIDs for this
phase - simpler to read and type while learning/debugging. Swapping to
UUIDs later is a column-type change, not a redesign.
"""

import enum

from sqlalchemy import (
    Column, Integer, String, Numeric, Date, DateTime, ForeignKey, Enum
)
from sqlalchemy.orm import relationship

from app.utils.time import utcnow_naive

from app.database.session import Base


class InvoiceDirection(str, enum.Enum):
    incoming = "incoming"   # a vendor billing us
    outgoing = "outgoing"   # us billing a customer


class PaymentStatus(str, enum.Enum):
    unpaid = "unpaid"
    partially_paid = "partially_paid"
    paid = "paid"
    overdue = "overdue"


class InvoiceStatus(str, enum.Enum):
    draft = "draft"
    pending_review = "pending_review"
    validated = "validated"
    approved = "approved"
    rejected = "rejected"
    sent = "sent"


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=utcnow_naive)

    customers = relationship("Customer", back_populates="organization")
    vendors = relationship("Vendor", back_populates="organization")
    invoices = relationship("Invoice", back_populates="organization")


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    name = Column(String(255), nullable=False)
    email = Column(String(255))
    address = Column(String(500))

    organization = relationship("Organization", back_populates="customers")
    invoices = relationship("Invoice", back_populates="customer")


class Vendor(Base):
    __tablename__ = "vendors"

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    name = Column(String(255), nullable=False)
    email = Column(String(255))
    address = Column(String(500))
    created_at = Column(DateTime, default=utcnow_naive)

    organization = relationship("Organization", back_populates="vendors")
    invoices = relationship("Invoice", back_populates="vendor")


class Invoice(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    direction = Column(Enum(InvoiceDirection), nullable=False)

    invoice_number = Column(String(100), nullable=False)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=True)

    invoice_date = Column(Date, nullable=False)
    due_date = Column(Date, nullable=False)

    subtotal = Column(Numeric(12, 2), nullable=False, default=0)
    tax = Column(Numeric(12, 2), nullable=False, default=0)
    discount = Column(Numeric(12, 2), nullable=False, default=0)
    total = Column(Numeric(12, 2), nullable=False, default=0)
    currency = Column(String(3), nullable=False, default="USD")

    payment_terms = Column(String(50), default="Net 30")
    payment_status = Column(Enum(PaymentStatus), nullable=False, default=PaymentStatus.unpaid)
    invoice_status = Column(Enum(InvoiceStatus), nullable=False, default=InvoiceStatus.draft)

    confidence_score = Column(Numeric(4, 3), nullable=True)   # set by Extraction Agent in Phase 2+
    risk_score = Column(Numeric(4, 3), nullable=True)          # set by Fraud Agent in Phase 7+

    created_at = Column(DateTime, default=utcnow_naive)

    organization = relationship("Organization", back_populates="invoices")
    vendor = relationship("Vendor", back_populates="invoices")
    customer = relationship("Customer", back_populates="invoices")
    items = relationship("InvoiceItem", back_populates="invoice", cascade="all, delete-orphan")
    payments = relationship("Payment", back_populates="invoice", cascade="all, delete-orphan")


class InvoiceItem(Base):
    __tablename__ = "invoice_items"

    id = Column(Integer, primary_key=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=False)
    description = Column(String(500), nullable=False)
    quantity = Column(Numeric(12, 3), nullable=False, default=1)
    unit_price = Column(Numeric(12, 2), nullable=False, default=0)
    line_total = Column(Numeric(12, 2), nullable=False, default=0)
    category = Column(String(100), nullable=True)  # set by Classification Agent in Phase 4+

    invoice = relationship("Invoice", back_populates="items")


class Payment(Base):
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=False)
    amount = Column(Numeric(12, 2), nullable=False)
    paid_date = Column(Date, nullable=False)
    method = Column(String(50), default="bank_transfer")
    status = Column(String(50), default="completed")

    invoice = relationship("Invoice", back_populates="payments")


class FraudFlag(Base):
    """
    Phase 4: one row per fraud/risk assessment. Kept separate from the
    Invoice table (even though Invoice also carries a denormalized
    risk_score for quick display) because a full assessment has
    several human-readable reasons and an LLM-written explanation
    attached to it - that's assessment history, not a single number.
    """
    __tablename__ = "fraud_flags"

    id = Column(Integer, primary_key=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=False)
    risk_score = Column(Numeric(4, 3), nullable=False)
    reasons_json = Column(String(2000), nullable=False)  # JSON-encoded list[str] of rule-based reasons
    explanation = Column(String(2000), nullable=True)     # LLM-written prose version of the reasons
    created_at = Column(DateTime, default=utcnow_naive)

    invoice = relationship("Invoice")


class AgentLog(Base):
    """
    Phase 5: one row per agent step the Orchestrator runs. This is
    Section 21's observability requirement made real - it's what
    powers the Agent Activity page, and it's the actual evidence (not
    just a claim) of which agent ran, why, and whether it succeeded.
    """
    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True)
    task_id = Column(String(64), nullable=False)          # groups all steps from one pipeline run
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=True)
    agent_name = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)             # success | failed
    input_summary = Column(String(500), nullable=True)
    output_summary = Column(String(500), nullable=True)
    duration_ms = Column(Integer, nullable=False)
    error = Column(String(1000), nullable=True)
    created_at = Column(DateTime, default=utcnow_naive)


class Communication(Base):
    """
    Phase 7: a drafted (and, once approved, sent) message. Nothing in
    this app ever actually sends a real email - there's no SMTP
    integration - so "sent" is a simulated status change after a human
    clicks approve, the same pattern the project design uses for
    simulated payments (Section 2). The point being demonstrated is
    the human-in-the-loop gate itself, not real email delivery.
    """
    __tablename__ = "communications"

    id = Column(Integer, primary_key=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=False)
    recipient = Column(String(255), nullable=False)
    subject = Column(String(255), nullable=False)
    body = Column(String(3000), nullable=False)
    status = Column(String(20), nullable=False, default="draft")  # draft | sent
    created_at = Column(DateTime, default=utcnow_naive)
    sent_at = Column(DateTime, nullable=True)

    invoice = relationship("Invoice")


class ApprovalRequest(Base):
    """
    Phase 8: the single unified human-in-the-loop queue (Section 39).
    Before this, "requires approval" was two disconnected, ad-hoc
    mechanisms - an invoice_status value a human could freely edit
    away, and a Communications page button. This table is what makes
    approval a real, tracked event: which agent asked for it, why, and
    who decided what. `type` + `related_id` point at whatever needs
    approval (an Invoice or a Communication) rather than one column
    per approvable thing, since the list of approvable action types
    will keep growing.
    """
    __tablename__ = "approval_requests"

    id = Column(Integer, primary_key=True)
    type = Column(String(50), nullable=False)              # "high_risk_invoice" | "send_communication"
    related_id = Column(Integer, nullable=False)
    requested_by_agent = Column(String(100), nullable=False)
    reason = Column(String(1000), nullable=False)
    status = Column(String(20), nullable=False, default="pending")  # pending | approved | rejected
    created_at = Column(DateTime, default=utcnow_naive)
    decided_at = Column(DateTime, nullable=True)


class UserRole(str, enum.Enum):
    owner = "owner"          # can approve/reject, everything a bookkeeper can
    bookkeeper = "bookkeeper"  # can create/edit/upload invoices, cannot approve


class User(Base):
    """
    Phase 11: the first real login in this project. Password is never
    stored or compared in plain text - see app/security/auth.py, which
    uses hashlib.scrypt (Python's stdlib, no third-party dependency)
    rather than storing anything reversible.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.bookkeeper)
    created_at = Column(DateTime, default=utcnow_naive)


class AuditLog(Base):
    """
    Phase 11: Section 9's audit_logs table, finally built. Distinct
    from AgentLog (which records agent decisions) - this records
    HUMAN actions: who created, edited, or deleted what, and when.
    Append-only in spirit - nothing in this app ever updates or
    deletes an AuditLog row once written.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    entity_type = Column(String(50), nullable=False)   # "invoice" | "communication" | ...
    entity_id = Column(Integer, nullable=False)
    action = Column(String(50), nullable=False)          # "create" | "update" | "delete"
    performed_by = Column(String(255), nullable=False)    # user email, or "system" for unauthenticated/API actions
    timestamp = Column(DateTime, default=utcnow_naive)
    details_json = Column(String(1000), nullable=True)
