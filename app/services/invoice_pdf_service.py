"""
Generates a real, downloadable PDF for an outgoing (customer) invoice.

Deterministic - no LLM anywhere in this file, same principle as
invoice_tools.py's CSV export: there is nothing here for an LLM to
usefully decide, the layout is fixed and the numbers already exist on
the Invoice/InvoiceItem rows. Generated on-demand from the current
invoice data each time it's requested, never cached to disk - an
invoice can be edited after creation (see invoice_service.update_invoice),
and a stored PDF would silently go stale the next time totals or line
items changed.

This is the outgoing-side counterpart to source_pdf_filename (Phase 9):
that lets you view the original PDF a vendor sent you (incoming); this
lets you produce the PDF you'd send to a customer (outgoing). Neither
makes sense for the other direction - a vendor's invoice was already a
PDF someone else made; an outgoing invoice never had one until now.
"""

import io
from decimal import Decimal

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_RIGHT

from app.models.models import CreditNote, Invoice

BUSINESS_NAME = "Maple Street Bakery Supply Co."


def _format_quantity(q: Decimal) -> str:
    """2.000 -> "2", 2.500 -> "2.5" - strips insignificant trailing
    zeros without Decimal.normalize()'s surprise switch to exponential
    notation for round numbers (e.g. Decimal('100').normalize() ==
    Decimal('1E+2'))."""
    text = f"{q:f}".rstrip("0").rstrip(".")
    return text or "0"


def generate_invoice_pdf(invoice: Invoice) -> bytes:
    """Renders `invoice` as a one-page PDF invoice document. Assumes
    invoice.customer and invoice.items are already loaded (callers use
    invoice_service.get_invoice(), which eager-loads items; customer is
    a normal lazy relationship, fine within an open session)."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    right_style = ParagraphStyle("right", parent=styles["Normal"], alignment=TA_RIGHT)

    story = [
        Paragraph(BUSINESS_NAME, styles["Title"]),
        Spacer(1, 0.15 * inch),
        Paragraph(f"INVOICE #{invoice.invoice_number}", styles["Heading2"]),
        Spacer(1, 0.2 * inch),
    ]

    customer = invoice.customer
    bill_to_lines = [f"<b>Bill To:</b> {customer.name if customer else '-'}"]
    if customer and customer.address:
        bill_to_lines.append(customer.address)
    if customer and customer.email:
        bill_to_lines.append(customer.email)
    story.append(Paragraph("<br/>".join(bill_to_lines), styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    meta_table = Table(
        [
            ["Invoice date", str(invoice.invoice_date), "Due date", str(invoice.due_date)],
            ["Payment terms", invoice.payment_terms or "-", "Currency", invoice.currency],
        ],
        colWidths=[1.2 * inch, 1.6 * inch, 1.2 * inch, 1.6 * inch],
    )
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.3 * inch))

    rows = [["Description", "Qty", "Unit Price", "Line Total"]]
    for item in invoice.items:
        rows.append([
            item.description,
            _format_quantity(item.quantity),
            f"{item.unit_price:.2f}",
            f"{item.line_total:.2f}",
        ])
    items_table = Table(rows, colWidths=[3.2 * inch, 0.8 * inch, 1.2 * inch, 1.3 * inch])
    items_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(items_table)
    story.append(Spacer(1, 0.2 * inch))

    def money(label: str, value: Decimal, bold: bool = False) -> str:
        style = "<b>%s</b>" if bold else "%s"
        return style % f"{label}: {value:.2f} {invoice.currency}"

    totals_lines = [money("Subtotal", invoice.subtotal)]
    if invoice.tax:
        totals_lines.append(money("Tax", invoice.tax))
    if invoice.discount:
        totals_lines.append(money("Discount", -invoice.discount))
    totals_lines.append(money("Total Due", invoice.total, bold=True))
    story.append(Paragraph("<br/>".join(totals_lines), right_style))

    doc.build(story)
    return buffer.getvalue()


def generate_credit_note_pdf(credit_note: CreditNote, invoice: Invoice) -> bytes:
    """Same visual language as generate_invoice_pdf() above, but its
    own much simpler document - a credit note references the original
    invoice, it never repeats or re-derives its line items."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()

    customer = invoice.customer
    story = [
        Paragraph(BUSINESS_NAME, styles["Title"]),
        Spacer(1, 0.15 * inch),
        Paragraph(f"CREDIT NOTE #{credit_note.credit_note_number}", styles["Heading2"]),
        Spacer(1, 0.2 * inch),
        Paragraph(f"<b>Issued to:</b> {customer.name if customer else '-'}", styles["Normal"]),
        Spacer(1, 0.15 * inch),
        Paragraph(f"<b>Against invoice:</b> #{invoice.invoice_number}", styles["Normal"]),
        Paragraph(f"<b>Date issued:</b> {credit_note.created_at.date()}", styles["Normal"]),
        Spacer(1, 0.2 * inch),
        Paragraph(f"<b>Reason:</b> {credit_note.reason}", styles["Normal"]),
        Spacer(1, 0.3 * inch),
        Paragraph(f"<b>Credit amount: {credit_note.amount:.2f} {credit_note.currency}</b>", styles["Heading3"]),
    ]

    doc.build(story)
    return buffer.getvalue()
