"""
What the LLM is allowed to produce when parsing a natural-language
question - a set of filter parameters, never a query string. This is
the entire mechanism that keeps "ask anything in English" safe: the
model's only output is data that fits this schema, which then drives
a fixed, parameterized tool call (app/tools/invoice_tools.py). It
cannot express "and also delete everything" because this schema has
no field for that.
"""

from pydantic import BaseModel


class QueryIntent(BaseModel):
    wants_summary: bool = False
    direction: str | None = None       # "incoming" or "outgoing"
    payment_status: str | None = None  # "unpaid" | "partially_paid" | "paid" | "overdue"
    min_total: float | None = None
    max_total: float | None = None
    overdue_only: bool = False
