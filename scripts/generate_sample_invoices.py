"""
Generates a small set of varied sample invoice PDFs for manually
testing the upload/extraction pipeline: `python scripts/generate_sample_invoices.py`.

No proprietary or confidential invoice data is needed to demonstrate
this project - every sample here is synthetic, built with PyMuPDF
(already a dependency, so this adds nothing new to install).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

from app.core.config import UPLOAD_DIR

OUT_DIR = UPLOAD_DIR / "samples"

SAMPLES = [
    (
        "sample_normal_invoice.pdf",
        """Golden Grain Milling
500 Mill Rd

INVOICE

Invoice Number: SAMPLE-NORMAL-001
Invoice Date: 2026-06-01
Due Date: 2026-07-01

Bill To: Maple Street Bakery Supply Co.

Description                Qty   Unit Price
All-purpose flour (50lb)   20    39.50
Baking powder (5lb)        4     12.00

Subtotal: 838.00
Tax: 16.76
Total Due: $854.76

Payment Terms: Net 30
""",
    ),
    (
        "sample_single_item_invoice.pdf",
        """City Electric Utility
1 Power Plant Way

INVOICE

Invoice Number: SAMPLE-UTILITY-001
Invoice Date: 2026-06-05
Due Date: 2026-06-20

Bill To: Maple Street Bakery Supply Co.

Description                     Qty   Unit Price
Commercial electricity - June   1     612.40

Subtotal: 612.40
Tax: 0.00
Total Due: $612.40

Payment Terms: Net 15
""",
    ),
    (
        "sample_multi_item_invoice.pdf",
        """Sunrise Packaging Co.
21 Industrial Way

INVOICE

Invoice Number: SAMPLE-MULTI-001
Invoice Date: 2026-06-10
Due Date: 2026-07-10

Bill To: Maple Street Bakery Supply Co.

Description                    Qty   Unit Price
Corrugated boxes (case)        30    3.75
Packing tape (roll)            15    2.25
Bubble wrap (roll)             5     18.00
Shipping labels (pack of 100)  10    6.50

Subtotal: 289.75
Tax: 5.80
Total Due: $295.55

Payment Terms: Net 30
""",
    ),
]


def make_pdf(path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text, fontsize=11)
    doc.save(str(path))
    doc.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, text in SAMPLES:
        path = OUT_DIR / filename
        make_pdf(path, text)
        print(f"Generated {path}")
    print(f"\n{len(SAMPLES)} sample invoice(s) written to {OUT_DIR}")
    print("Upload any of these on the /web/upload page to test extraction.")


if __name__ == "__main__":
    main()
