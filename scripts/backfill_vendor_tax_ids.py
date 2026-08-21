"""
One-time backfill for vendors that predate the address/tax_id fields:
re-extracts those values from PDFs already uploaded and saved on disk
(Invoice.source_pdf_filename), for any vendor still missing them.

Deliberately does NOT fabricate a value for a vendor whose original
PDF isn't available (e.g. wiped by a free-tier cloud redeploy, or the
invoice was created by hand/API with no upload at all) - those are
reported as skipped, left null, same as any vendor that genuinely
never had this information on file.

Safe to re-run: only touches a vendor still missing address or tax_id,
and only writes a real value actually read from its own saved PDF.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.session import SessionLocal
from app.models.models import Invoice, InvoiceDirection, Vendor
from app.core.config import UPLOAD_DIR
from app.services import extraction_service, llm_extraction_service


def main() -> None:
    db = SessionLocal()
    updated = 0
    skipped_already_set = 0
    skipped_no_pdf = 0

    invoices = (
        db.query(Invoice)
        .filter(
            Invoice.direction == InvoiceDirection.incoming,
            Invoice.source_pdf_filename.isnot(None),
            Invoice.vendor_id.isnot(None),
        )
        .all()
    )
    print(f"Found {len(invoices)} incoming invoices with a linked source PDF.")

    seen_vendor_ids = set()
    for inv in invoices:
        vendor = db.get(Vendor, inv.vendor_id)
        if vendor is None or vendor.id in seen_vendor_ids:
            continue
        if vendor.address and vendor.tax_id:
            seen_vendor_ids.add(vendor.id)
            skipped_already_set += 1
            continue

        pdf_path = UPLOAD_DIR / inv.source_pdf_filename
        if not pdf_path.exists():
            skipped_no_pdf += 1
            continue

        raw_text = extraction_service.extract_text_from_pdf(pdf_path.read_bytes())
        try:
            result = llm_extraction_service.extract_invoice_with_llm(raw_text)
            address, tax_id = result.vendor_address, result.vendor_tax_id
        except llm_extraction_service.LLMUnavailableError:
            naive = extraction_service.naive_parse_invoice_fields(raw_text)
            address, tax_id = naive.vendor_address, naive.vendor_tax_id

        changed = False
        if not vendor.address and address:
            vendor.address = address
            changed = True
        if not vendor.tax_id and tax_id:
            vendor.tax_id = tax_id
            changed = True

        seen_vendor_ids.add(vendor.id)
        if changed:
            db.commit()
            updated += 1
            print(f"  Updated vendor #{vendor.id} ({vendor.name}): address={vendor.address!r}, tax_id={vendor.tax_id!r}")

    print(
        f"\nDone. Updated {updated} vendor(s). "
        f"Skipped {skipped_already_set} already complete, {skipped_no_pdf} with no PDF still on disk."
    )
    db.close()


if __name__ == "__main__":
    main()
