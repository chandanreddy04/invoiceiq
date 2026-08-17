"""Structured output shape for the Expense Classification Agent."""

from pydantic import BaseModel


class ClassifiedItem(BaseModel):
    description: str
    category: str
    confidence: float = 0.5


class ClassificationResult(BaseModel):
    items: list[ClassifiedItem]
