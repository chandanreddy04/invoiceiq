"""
What the LLM is allowed to produce when parsing a natural-language
question - a set of filter parameters, never a query string. This is
the entire mechanism that keeps "ask anything in English" safe: the
model's only output is data that fits this schema, which then drives
a fixed, parameterized tool call (app/tools/invoice_tools.py). It
cannot express "and also delete everything" because this schema has
no field for that.

Fields cover every commonly-asked angle on an invoice: who it's
with, its number, its expense category, whether it's overdue or
flagged risky, and its amount/status/direction. Deliberately still
NOT here: anything requiring aggregation across invoices (e.g. "which
vendor do I spend the most with") - that needs a different tool
function entirely, not just another filter field, and is documented
as a real, current gap rather than silently unsupported.
"""

from pydantic import BaseModel


class QueryIntent(BaseModel):
    wants_summary: bool = False
    direction: str | None = None       # "incoming" or "outgoing"
    payment_status: str | None = None  # "unpaid" | "partially_paid" | "paid" | "overdue"
    min_total: float | None = None
    max_total: float | None = None
    overdue_only: bool = False
    party_name: str | None = None      # vendor or customer name (partial match, e.g. "golden grain")
    invoice_number: str | None = None  # partial match, e.g. "GGM"
    category: str | None = None        # expense category, e.g. "Raw Ingredients"
    risky_only: bool = False           # invoices flagged 50%+ risk by the Fraud/Risk Agent
