"""
Generates a simple text-based PDF invoice for testing the Phase 2
upload/extraction pipeline - avoids needing a real scanned invoice or
another dependency (reportlab etc.) just to produce test input.
Uses PyMuPDF itself, which we already need for extraction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "invoices" / "test_invoice_sunrise.pdf"

TEXT = """Sunrise Packaging Co.
21 Industrial Way

INVOICE

Invoice Number: SPC-2201
Invoice Date: 2026-07-15
Due Date: 2026-08-14

Bill To: Maple Street Bakery Supply Co.

Description                Qty   Unit Price
Cardboard boxes (case)     20    15.00
Packing tape (roll)        50    2.50

Subtotal: 425.00
Tax: 10.00
Total Due: $435.00

Payment Terms: Net 30
"""


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), TEXT, fontsize=11)
    doc.save(str(OUT_PATH))
    doc.close()
    print(f"Test invoice PDF written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
