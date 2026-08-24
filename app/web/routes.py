"""
Server-rendered HTML pages. Every route here either reads via the
shared service layer or builds a Pydantic schema from form data and
hands it to the exact same invoice_service functions the JSON API
uses - so validation and totals math behave identically no matter
which interface was used.

Every route except /login itself requires a logged-in user
(Depends(require_login)); approve/reject additionally requires the
owner role (Depends(require_owner)) - see app/security/deps.py.
"""

import csv
import io
import json
import logging
import uuid
from datetime import date
from decimal import Decimal, InvalidOperation

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.core.config import UPLOAD_DIR, MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB, GROQ_API_KEY
from app.utils.time import utcnow_naive
from app.database.session import get_db
from app.models.models import (
    Invoice, Customer, Vendor, InvoiceDirection, PaymentStatus, InvoiceStatus,
    FraudFlag, AgentLog, Communication, ApprovalRequest, User, AuditLog, CreditNote,
)
from app.schemas.invoice import InvoiceCreate, InvoiceItemCreate, InvoiceUpdate
from app.schemas.party import VendorCreate, CustomerCreate
from app.services import invoice_service, extraction_service, llm_extraction_service, invoice_pdf_service, email_service, recurring_invoice_service, credit_note_service
from app.schemas.recurring_invoice import RecurringInvoiceCreate
from app.services.validation_service import InvoiceValidationError
from app.agents import orchestrator, communication_agent, payment_ap_agent, collections_agent, fraud_risk_agent
from app.tools import invoice_tools
from app.security.auth import verify_password, create_session_token
from app.security.deps import require_login, require_owner, get_current_user, SESSION_COOKIE_NAME

router = APIRouter(prefix="/web", tags=["web"])
templates = Jinja2Templates(directory="app/templates")
templates.env.filters["money"] = invoice_tools.format_money_by_currency
logger = logging.getLogger(__name__)

DEFAULT_ORG_ID = 1


def _parse_items_from_form(form) -> list[InvoiceItemCreate]:
    """The line-items table on the New/Edit Invoice and Recurring Invoice
    forms grows dynamically (one row at a time, as the last row gets
    filled in) rather than showing a fixed block of empty boxes, so this
    needs to read well past whatever the smallest real invoice would
    need - 50 is a generous ceiling, not an expected typical count."""
    items = []
    for i in range(50):
        desc = (form.get(f"item_description_{i}") or "").strip()
        if not desc:
            continue
        try:
            qty = Decimal(form.get(f"item_quantity_{i}") or "0")
            price = Decimal(form.get(f"item_unit_price_{i}") or "0")
        except InvalidOperation:
            continue
        items.append(InvoiceItemCreate(description=desc, quantity=qty, unit_price=price))
    return items


def _match_vendor(vendors: list, extracted_name: str | None, extracted_tax_id: str | None) -> tuple[int | None, str | None]:
    """Tax ID is checked first and wins outright when it matches - two
    real businesses never share a real tax ID, but they can easily
    share a near-identical name (a franchise, a rebrand, a typo in the
    extraction). Falls back to a case/whitespace-insensitive name
    match only when no tax ID match was found. Returns (vendor_id,
    matched_by) - matched_by is "tax_id" | "name" | None, reported
    separately from "did the document merely contain a tax ID" so the
    review page never claims a tax-ID match that didn't actually
    happen (e.g. the document has one, but the matched vendor's own
    record doesn't, so the real match was by name)."""
    tax_id = (extracted_tax_id or "").strip().lower()
    if tax_id:
        for v in vendors:
            if v.tax_id and v.tax_id.strip().lower() == tax_id:
                return v.id, "tax_id"

    name = (extracted_name or "").strip().lower()
    if name:
        for v in vendors:
            if v.name.strip().lower() == name:
                return v.id, "name"

    return None, None


def _enrich_vendor_if_blank(vendor: Vendor, form) -> None:
    """A vendor matched by name (or picked manually) can still be
    missing its address/Tax ID - filling those in from THIS invoice's
    extraction closes a real gap otherwise: a vendor identified purely
    by name would never pick up its Tax ID just because a later
    invoice happened to print one. Only ever fills in a field that was
    genuinely blank - never overwrites a value already on file, even
    if this extraction disagrees with it."""
    extracted_address = (form.get("extracted_vendor_address") or "").strip()
    extracted_tax_id = (form.get("extracted_vendor_tax_id") or "").strip()
    if not vendor.address and extracted_address:
        vendor.address = extracted_address
    if not vendor.tax_id and extracted_tax_id:
        vendor.tax_id = extracted_tax_id


def _resolve_or_create_vendor(db: Session, org_id: int, form) -> int:
    """An explicit vendor_id from the review page's dropdown always
    wins - a human deciding "use this existing vendor" beats a guess.
    That vendor still gets enriched with any address/Tax ID this
    invoice's extraction found and it was missing (see
    _enrich_vendor_if_blank). Otherwise, if extraction found a vendor
    name that didn't match anyone on file, create that vendor now
    (with whatever address/tax ID were also extracted) - the same
    record a human would end up with by going to the Vendors page and
    adding it themselves first, just without the extra trip."""
    vendor_id = form.get("vendor_id")
    if vendor_id:
        vendor_id = int(vendor_id)
        vendor = db.get(Vendor, vendor_id)
        if vendor is not None:
            _enrich_vendor_if_blank(vendor, form)
            db.commit()
        return vendor_id

    extracted_name = (form.get("extracted_vendor_name") or "").strip()
    if not extracted_name:
        raise InvoiceValidationError("No vendor selected, and no vendor name was extracted to create one from.")

    new_vendor = Vendor(
        organization_id=org_id, name=extracted_name,
        address=(form.get("extracted_vendor_address") or "").strip() or None,
        tax_id=(form.get("extracted_vendor_tax_id") or "").strip() or None,
    )
    db.add(new_vendor)
    db.commit()
    db.refresh(new_vendor)
    return new_vendor.id


def _audit(db: Session, entity_type: str, entity_id: int, action: str, performed_by: str, details: dict | None = None):
    db.add(AuditLog(
        entity_type=entity_type, entity_id=entity_id, action=action,
        performed_by=performed_by, details_json=json.dumps(details) if details else None,
    ))
    db.commit()


# --- Auth --------------------------------------------------------------

@router.get("/login", response_class=HTMLResponse)
def web_login_form(request: Request, next: str = "/web/dashboard", db: Session = Depends(get_db)):
    if get_current_user(request, db) is not None:
        return RedirectResponse(next, status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "next": next})


@router.post("/login", response_class=HTMLResponse)
def web_login_submit(
    request: Request, email: str = Form(...), password: str = Form(...),
    next: str = Form("/web/dashboard"), db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == email).first()
    if user is None or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            "login.html", {"request": request, "next": next, "error": "Invalid email or password."}, status_code=401
        )
    resp = RedirectResponse(next, status_code=303)
    resp.set_cookie(SESSION_COOKIE_NAME, create_session_token(user.id), httponly=True, samesite="lax")
    return resp


@router.get("/logout")
def web_logout():
    resp = RedirectResponse("/web/login", status_code=303)
    resp.delete_cookie(SESSION_COOKIE_NAME)
    return resp


# --- Dashboard -----------------------------------------------------------

@router.get("/dashboard", response_class=HTMLResponse)
def web_dashboard(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    stats = invoice_tools.get_dashboard_stats(db, DEFAULT_ORG_ID)
    return templates.TemplateResponse("dashboard.html", {"request": request, "stats": stats, "current_user": current_user})


# --- Invoices --------------------------------------------------------------

@router.get("/invoices", response_class=HTMLResponse)
def web_list_invoices(
    request: Request,
    direction: str | None = None,
    payment_status: str | None = None,
    q: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_login),
):
    invoices = invoice_service.list_invoices(
        db,
        DEFAULT_ORG_ID,
        InvoiceDirection(direction) if direction else None,
        PaymentStatus(payment_status) if payment_status else None,
        q=q,
    )
    return templates.TemplateResponse(
        "invoices_list.html",
        {"request": request, "invoices": invoices, "direction": direction, "payment_status": payment_status, "q": q, "current_user": current_user},
    )


@router.get("/invoices/export.csv")
def web_export_invoices_csv(
    direction: str | None = None,
    payment_status: str | None = None,
    q: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_login),
):
    """A real, buildable piece of "accounting ecosystem integration" -
    a full QuickBooks/Xero OAuth sync needs developer credentials on
    their platforms this project doesn't have, but CSV is genuinely how
    most accounting software actually imports data, QuickBooks and Xero
    both included. Respects whatever filter is active on the Invoices
    page, so exporting "just what I'm looking at" works for free."""
    invoices = invoice_service.list_invoices(
        db, DEFAULT_ORG_ID,
        InvoiceDirection(direction) if direction else None,
        PaymentStatus(payment_status) if payment_status else None,
        q=q,
    )
    # Read every field into plain Python values *before* returning -
    # StreamingResponse iterates its generator only after this function
    # has already returned, by which point the request-scoped `db`
    # session (Depends(get_db)) has closed. A real bug caught by testing:
    # touching a lazy-loaded ORM attribute (like .vendor) from inside the
    # generator raised DetachedInstanceError, since there was no open
    # session left to run that query.
    rows = [
        (
            inv.invoice_number, inv.direction.value,
            inv.vendor.name if inv.vendor else (inv.customer.name if inv.customer else ""),
            inv.invoice_date, inv.due_date, inv.subtotal, inv.tax, inv.discount, inv.total, inv.currency,
            inv.payment_status.value, inv.invoice_status.value,
            f"{inv.risk_score:.3f}" if inv.risk_score is not None else "",
        )
        for inv in invoices
    ]

    def _generate():
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow([
            "Invoice Number", "Direction", "Party", "Invoice Date", "Due Date",
            "Subtotal", "Tax", "Discount", "Total", "Currency",
            "Payment Status", "Invoice Status", "Risk Score",
        ])
        yield buffer.getvalue()
        for row in rows:
            buffer.seek(0); buffer.truncate(0)
            writer.writerow(row)
            yield buffer.getvalue()

    return StreamingResponse(
        _generate(), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=invoiceiq_invoices.csv"},
    )


@router.get("/invoices/export/general-ledger.csv")
def web_export_general_ledger_csv(
    direction: str | None = None,
    payment_status: str | None = None,
    q: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_login),
):
    """Line-item-level export instead of one row per invoice - the shape
    an actual GL import wants, since posting to the right expense
    category happens at the line-item level, not the invoice level."""
    invoices = invoice_service.list_invoices(
        db, DEFAULT_ORG_ID,
        InvoiceDirection(direction) if direction else None,
        PaymentStatus(payment_status) if payment_status else None,
        q=q,
    )
    # Same fix as the invoice export above: materialize every field into
    # plain Python values now, while the session is still open, since the
    # generator below only runs after this function has already returned.
    rows = [
        (
            inv.invoice_number, inv.invoice_date,
            inv.vendor.name if inv.vendor else (inv.customer.name if inv.customer else ""),
            item.description, item.category or "", item.quantity, item.unit_price, item.line_total, inv.currency,
        )
        for inv in invoices
        for item in inv.items
    ]

    def _generate():
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow([
            "Invoice Number", "Invoice Date", "Party", "Line Description",
            "Category", "Quantity", "Unit Price", "Line Total", "Currency",
        ])
        yield buffer.getvalue()
        for row in rows:
            buffer.seek(0); buffer.truncate(0)
            writer.writerow(row)
            yield buffer.getvalue()

    return StreamingResponse(
        _generate(), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=invoiceiq_general_ledger.csv"},
    )


@router.post("/invoices/bulk-mark-paid")
async def web_bulk_mark_paid(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    form = await request.form()
    invoice_ids = [int(v) for v in form.getlist("invoice_ids")]
    for invoice_id in invoice_ids:
        invoice = db.query(Invoice).filter(Invoice.id == invoice_id, Invoice.organization_id == DEFAULT_ORG_ID).first()
        if invoice is None:
            continue
        invoice_service.update_invoice(db, invoice, InvoiceUpdate(payment_status=PaymentStatus.paid))
        _audit(db, "invoice", invoice.id, "bulk_mark_paid", current_user.email)

    from urllib.parse import urlencode
    keep = {k: v for k in ("direction", "payment_status", "q") if (v := form.get(k))}
    qs = urlencode(keep)
    return RedirectResponse(f"/web/invoices{'?' + qs if qs else ''}", status_code=303)


@router.get("/invoices/new", response_class=HTMLResponse)
def web_new_invoice_form(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    vendors = db.query(Vendor).filter(Vendor.organization_id == DEFAULT_ORG_ID).all()
    customers = db.query(Customer).filter(Customer.organization_id == DEFAULT_ORG_ID).all()
    return templates.TemplateResponse(
        "invoice_form.html",
        {"request": request, "invoice": None, "vendors": vendors, "customers": customers, "values": {},
         "line_items": None, "extraction": None,
         "form_action": "/web/invoices/new", "current_user": current_user,
         "suggested_invoice_number": invoice_service.suggest_next_invoice_number(db, DEFAULT_ORG_ID)},
    )


@router.post("/invoices/new/upload", response_class=HTMLResponse)
async def web_new_invoice_upload(
    request: Request, file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(require_login)
):
    """The Upload Invoice page used to be a separate destination from
    New Invoice, even though both do the same thing - add a new
    (incoming) invoice, just with or without a PDF to extract from
    first. This folds the extraction step directly into the New
    Invoice page: it re-renders the same invoice_form.html, pre-filled
    from whatever was extracted, instead of a separate review page."""
    vendors = db.query(Vendor).filter(Vendor.organization_id == DEFAULT_ORG_ID).all()
    customers = db.query(Customer).filter(Customer.organization_id == DEFAULT_ORG_ID).all()

    def _blank_form_with_error(error: str):
        return templates.TemplateResponse(
            "invoice_form.html",
            {"request": request, "invoice": None, "vendors": vendors, "customers": customers, "values": {},
             "line_items": None, "extraction": None, "error": error,
             "form_action": "/web/invoices/new", "current_user": current_user,
             "suggested_invoice_number": invoice_service.suggest_next_invoice_number(db, DEFAULT_ORG_ID)},
            status_code=422,
        )

    if not (file.filename or "").lower().endswith(".pdf"):
        return _blank_form_with_error("Only PDF files are supported.")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        return _blank_form_with_error(f"File is larger than the {MAX_UPLOAD_SIZE_MB}MB limit.")

    # Save the original upload under a generated name - never the
    # client-supplied filename, which closes off path traversal
    # (e.g. "../../etc/passwd.pdf") entirely rather than trying to
    # sanitize it.
    saved_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.pdf"
    saved_path.write_bytes(file_bytes)

    try:
        raw_text = extraction_service.extract_text_from_pdf(file_bytes)
    except Exception:
        logger.exception("Failed to read PDF %s", saved_path)
        return _blank_form_with_error("Could not read this PDF - it may be corrupted or password-protected.")

    try:
        llm_result = llm_extraction_service.extract_invoice_with_llm(raw_text)
        extracted = {
            "method": "llm",
            "vendor_name": llm_result.vendor_name,
            "vendor_address": llm_result.vendor_address,
            "vendor_tax_id": llm_result.vendor_tax_id,
            "invoice_number": llm_result.invoice_number,
            "invoice_date": llm_result.invoice_date,
            "due_date": llm_result.due_date,
            "currency": llm_result.currency,
            "tax": llm_result.tax,
            "discount": llm_result.discount,
            "line_items": [item.model_dump() for item in llm_result.line_items],
            "total_hint": None,
        }
    except llm_extraction_service.LLMUnavailableError as e:
        logger.info("Falling back to naive regex extraction: %s", e)
        naive = extraction_service.naive_parse_invoice_fields(raw_text)
        extracted = {
            "method": "regex_fallback",
            "vendor_name": naive.vendor_name,
            "vendor_address": naive.vendor_address,
            "vendor_tax_id": naive.vendor_tax_id,
            "invoice_number": naive.invoice_number,
            "invoice_date": naive.invoice_date.isoformat() if naive.invoice_date else None,
            "due_date": naive.due_date.isoformat() if naive.due_date else None,
            "currency": "USD",
            "tax": 0,
            "discount": 0,
            "line_items": [],
            "total_hint": naive.total,
        }

    matched_vendor_id, matched_by = _match_vendor(vendors, extracted.get("vendor_name"), extracted.get("vendor_tax_id"))

    values = {
        "direction": "incoming",
        "vendor_id": matched_vendor_id,
        "invoice_number": extracted.get("invoice_number") or "",
        "invoice_date": extracted.get("invoice_date") or "",
        "due_date": extracted.get("due_date") or "",
        "currency": extracted.get("currency") or "USD",
        "tax": extracted.get("tax") or 0,
        "discount": extracted.get("discount") or 0,
    }
    extraction = {
        "method": extracted["method"],
        "vendor_name": extracted.get("vendor_name"),
        "vendor_address": extracted.get("vendor_address"),
        "vendor_tax_id": extracted.get("vendor_tax_id"),
        "total_hint": extracted.get("total_hint"),
        "raw_text": raw_text,
        "matched_vendor_id": matched_vendor_id,
        "matched_by": matched_by,
        "saved_pdf_filename": saved_path.name,
        "llm_backend_label": _llm_backend_label(),
    }

    return templates.TemplateResponse(
        "invoice_form.html",
        {
            "request": request, "invoice": None, "vendors": vendors, "customers": customers,
            "values": values, "line_items": extracted.get("line_items"), "extraction": extraction,
            "form_action": "/web/invoices/new", "current_user": current_user,
            "suggested_invoice_number": invoice_service.suggest_next_invoice_number(db, DEFAULT_ORG_ID),
        },
    )


def _llm_backend_label() -> str:
    """A real, found-live inaccuracy: this page used to hardcode "the
    local LLM (phi3.5)" regardless of which backend actually served the
    request - true locally, but flatly wrong on the cloud deployment,
    which has no local model at all and runs on Groq instead."""
    return f"Groq ({llm_extraction_service.MODEL_NAME})" if GROQ_API_KEY else f"the local LLM ({llm_extraction_service.MODEL_NAME})"


@router.post("/invoices/new", response_class=HTMLResponse)
async def web_create_invoice(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    form = await request.form()
    items = _parse_items_from_form(form)
    vendors = db.query(Vendor).filter(Vendor.organization_id == DEFAULT_ORG_ID).all()
    customers = db.query(Customer).filter(Customer.organization_id == DEFAULT_ORG_ID).all()
    direction = form.get("direction")

    def _retry_render(error: str, vendor_id_for_dropdown: int | None):
        values = {
            "direction": direction,
            "invoice_number": form.get("invoice_number"),
            "invoice_date": form.get("invoice_date"),
            "due_date": form.get("due_date"),
            "currency": form.get("currency") or "USD",
            "tax": form.get("tax") or 0,
            "discount": form.get("discount") or 0,
            "vendor_id": vendor_id_for_dropdown,
        }
        # Only present if this submission came from the PDF-upload path
        # (carried forward via hidden fields) - keeps the vendor
        # match/auto-create hint accurate on a retry after a validation
        # error, instead of losing the extraction context entirely.
        extraction = None
        if form.get("extracted_vendor_name"):
            extraction = {
                "method": None,
                "vendor_name": form.get("extracted_vendor_name"),
                "vendor_address": form.get("extracted_vendor_address"),
                "vendor_tax_id": form.get("extracted_vendor_tax_id"),
                "total_hint": None,
                "raw_text": None,
                "matched_vendor_id": vendor_id_for_dropdown,
                "matched_by": None,
                "saved_pdf_filename": form.get("source_pdf_filename"),
            }
        return templates.TemplateResponse(
            "invoice_form.html",
            {
                "request": request, "invoice": None, "vendors": vendors, "customers": customers,
                "values": values, "line_items": [item.model_dump() for item in items], "extraction": extraction,
                "form_action": "/web/invoices/new", "error": error, "current_user": current_user,
                "suggested_invoice_number": invoice_service.suggest_next_invoice_number(db, DEFAULT_ORG_ID),
            },
            status_code=422,
        )

    vendor_id = None
    if direction == "incoming" and (
        form.get("vendor_id") or form.get("extracted_vendor_name") or form.get("source_pdf_filename")
    ):
        # Only resolve/auto-create/enrich when there's actually a signal
        # to resolve from - a plain manual entry with no vendor picked
        # stays vendor-less, same as before this page gained the
        # upload/extraction path. But a submission carrying
        # source_pdf_filename came from that path, so if extraction
        # found no vendor name AND nothing was picked manually either,
        # that should still surface a clear error rather than silently
        # saving a vendor-less invoice.
        try:
            vendor_id = _resolve_or_create_vendor(db, DEFAULT_ORG_ID, form)
        except InvoiceValidationError as e:
            return _retry_render(str(e), None)

    try:
        payload = InvoiceCreate(
            direction=direction,
            invoice_number=form["invoice_number"],
            vendor_id=vendor_id,
            customer_id=int(form["customer_id"]) if form.get("customer_id") else None,
            invoice_date=form["invoice_date"],
            due_date=form["due_date"],
            tax=Decimal(form.get("tax") or "0"),
            discount=Decimal(form.get("discount") or "0"),
            currency=(form.get("currency") or "USD").upper(),
            payment_terms=form.get("payment_terms") or "Net 30",
            items=items,
            source_pdf_filename=form.get("source_pdf_filename") or None,
        )
        invoice = invoice_service.create_invoice(db, DEFAULT_ORG_ID, payload)
        _audit(db, "invoice", invoice.id, "create", current_user.email, {"invoice_number": invoice.invoice_number})
    except (InvoiceValidationError, ValueError) as e:
        return _retry_render(str(e), vendor_id)
    return RedirectResponse("/web/invoices", status_code=303)


@router.get("/invoices/{invoice_id}", response_class=HTMLResponse)
def web_view_invoice(invoice_id: int, request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    invoice = invoice_service.get_invoice(db, invoice_id)
    vendors = db.query(Vendor).filter(Vendor.organization_id == DEFAULT_ORG_ID).all()
    customers = db.query(Customer).filter(Customer.organization_id == DEFAULT_ORG_ID).all()
    fraud_flag = (
        db.query(FraudFlag).filter(FraudFlag.invoice_id == invoice_id).order_by(FraudFlag.created_at.desc()).first()
        if invoice else None
    )
    comms = (
        db.query(Communication).filter(Communication.invoice_id == invoice_id).order_by(Communication.created_at.desc()).all()
        if invoice else []
    )
    credit_notes = credit_note_service.get_credit_notes_for_invoice(db, invoice_id) if invoice else []
    remaining_creditable = credit_note_service.remaining_creditable(db, invoice) if invoice else None
    return templates.TemplateResponse(
        "invoice_form.html",
        {
            "request": request, "invoice": invoice, "vendors": vendors, "customers": customers,
            "values": {}, "line_items": None, "extraction": None,
            "form_action": f"/web/invoices/{invoice_id}", "fraud_flag": fraud_flag, "comms": comms,
            "credit_notes": credit_notes, "remaining_creditable": remaining_creditable,
            "current_user": current_user,
        },
    )


@router.post("/invoices/{invoice_id}/override-status")
def web_override_invoice_status(
    invoice_id: int, new_status: str = Form(...), db: Session = Depends(get_db), current_user: User = Depends(require_owner),
):
    """A real gap found live: invoice_status was write-once outside of
    invoice creation - the Approvals page can only decide a *pending*
    request exactly once (_decide_approval() refuses anything already
    decided), and the invoice detail page rendered the status as plain
    text with no field at all. There was no way back if an owner
    approved something by mistake, or wanted to reopen a rejected
    invoice. Deliberately owner-only and a separate route from the
    general edit form (require_login) - a bookkeeper must never be able
    to self-approve a flagged invoice by editing a field, which is
    exactly the scenario the whole Approvals gate exists to prevent."""
    invoice = invoice_service.get_invoice(db, invoice_id)
    if invoice is not None:
        old_status = invoice.invoice_status.value
        invoice.invoice_status = InvoiceStatus(new_status)
        db.commit()
        _audit(db, "invoice", invoice.id, "status_override", current_user.email, {"from": old_status, "to": new_status})
    return RedirectResponse(f"/web/invoices/{invoice_id}", status_code=303)


@router.get("/invoices/{invoice_id}/risk-explanation/stream")
def web_stream_risk_explanation(invoice_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    """Server-Sent Events endpoint powering the "regenerate live" button
    on the invoice detail page. Re-narrates the SAME already-computed
    risk_score/reasons from the most recent FraudFlag row - never
    recomputes the verdict, only re-generates the explanation sentence,
    token by token, so a person watching isn't staring at a blank space
    for the 15-30s a real local model call takes."""
    fraud_flag = (
        db.query(FraudFlag).filter(FraudFlag.invoice_id == invoice_id).order_by(FraudFlag.created_at.desc()).first()
    )

    def _generate():
        if fraud_flag is None:
            yield "data: No risk assessment exists for this invoice.\n\n"
            yield "event: done\ndata: \n\n"
            return
        reasons = json.loads(fraud_flag.reasons_json) if fraud_flag.reasons_json else []
        try:
            for chunk in fraud_risk_agent.explain_risk_with_llm_stream(float(fraud_flag.risk_score), reasons):
                yield f"data: {chunk.replace(chr(10), ' ')}\n\n"
        except llm_extraction_service.LLMUnavailableError:
            yield f"data: [LLM unavailable] {' '.join(reasons)}\n\n"
        yield "event: done\ndata: \n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.get("/invoices/{invoice_id}/borderline-review/stream")
def web_stream_borderline_review(invoice_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    """SSE endpoint for the Fraud/Risk Agent's on-demand reasoning-model
    second opinion (app.agents.fraud_risk_agent.deliberate_on_borderline_case_stream).
    Deliberately NOT run automatically on invoice creation or from
    run_fraud_check() - measured live at 3-6+ minutes on a CPU-only
    laptop for deepseek-r1:8b, an unacceptable block on saving an invoice
    for a review step that's advisory only to begin with. Streaming lets
    a person who clicks "Get AI's second opinion" watch the model think
    in real time (a genuine, visible difference from the fast one-shot
    narration streamed above) instead of staring at a blank space for
    several minutes. Persists the finished trace onto the same FraudFlag
    row once streaming completes, same as the score/explanation it sits
    alongside - so it survives a page reload instead of vanishing."""
    fraud_flag = (
        db.query(FraudFlag).filter(FraudFlag.invoice_id == invoice_id).order_by(FraudFlag.created_at.desc()).first()
    )

    def _generate():
        if fraud_flag is None:
            yield "data: No risk assessment exists for this invoice.\n\n"
            yield "event: done\ndata: \n\n"
            return
        risk_score = float(fraud_flag.risk_score)
        if not (fraud_risk_agent.BORDERLINE_BAND[0] <= risk_score <= fraud_risk_agent.BORDERLINE_BAND[1]):
            yield "data: This invoice's score isn't in the ambiguous band - no second opinion to give.\n\n"
            yield "event: done\ndata: \n\n"
            return
        reasons = json.loads(fraud_flag.reasons_json) if fraud_flag.reasons_json else []
        thinking_so_far = []
        try:
            for chunk in fraud_risk_agent.deliberate_on_borderline_case_stream(risk_score, reasons):
                if chunk["type"] == "thinking":
                    thinking_so_far.append(chunk["text"])
                    yield f"data: {chunk['text'].replace(chr(10), ' ')}\n\n"
        except llm_extraction_service.LLMUnavailableError:
            # Backend-specific troubleshooting, same idea as the narration
            # stream's fallback above - a bad/missing GROQ_API_KEY and a
            # not-running local Ollama fail the same way from the caller's
            # side (LLMUnavailableError), but the fix looks nothing alike.
            if GROQ_API_KEY:
                yield "data: [reasoning model unavailable - check GROQ_API_KEY and Groq's status]\n\n"
            else:
                yield "data: [reasoning model unavailable - is `ollama pull deepseek-r1:8b` done and `ollama serve` running?]\n\n"
            yield "event: done\ndata: \n\n"
            return
        fraud_flag.reasoning_trace = "".join(thinking_so_far).strip()
        db.commit()
        yield "event: done\ndata: \n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.post("/invoices/{invoice_id}", response_class=HTMLResponse)
async def web_update_invoice(invoice_id: int, request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    form = await request.form()
    invoice = invoice_service.get_invoice(db, invoice_id)
    items = _parse_items_from_form(form)

    try:
        payload = InvoiceUpdate(
            invoice_number=form.get("invoice_number"),
            vendor_id=int(form["vendor_id"]) if form.get("vendor_id") else None,
            customer_id=int(form["customer_id"]) if form.get("customer_id") else None,
            invoice_date=form.get("invoice_date") or None,
            due_date=form.get("due_date") or None,
            tax=Decimal(form["tax"]) if form.get("tax") else None,
            discount=Decimal(form["discount"]) if form.get("discount") else None,
            currency=(form.get("currency") or None),
            payment_terms=form.get("payment_terms"),
            payment_status=form.get("payment_status"),
            # invoice_status is intentionally NOT settable from this form (Phase 8) -
            # it can only change via the Approvals queue or automatic agent actions.
            # Omitted entirely (not passed as None) so exclude_unset leaves it untouched.
            items=items,
        )
        invoice_service.update_invoice(db, invoice, payload)
        _audit(db, "invoice", invoice_id, "update", current_user.email, {"fields": list(payload.model_dump(exclude_unset=True).keys())})
    except (InvoiceValidationError, ValueError) as e:
        vendors = db.query(Vendor).filter(Vendor.organization_id == DEFAULT_ORG_ID).all()
        customers = db.query(Customer).filter(Customer.organization_id == DEFAULT_ORG_ID).all()
        return templates.TemplateResponse(
            "invoice_form.html",
            {
                "request": request, "invoice": invoice, "vendors": vendors, "customers": customers,
                "values": {}, "form_action": f"/web/invoices/{invoice_id}", "error": str(e), "current_user": current_user,
            },
            status_code=422,
        )
    return RedirectResponse(f"/web/invoices/{invoice_id}", status_code=303)


@router.post("/invoices/{invoice_id}/delete")
def web_delete_invoice(invoice_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    invoice = invoice_service.get_invoice(db, invoice_id)
    if invoice is not None:
        invoice_service.delete_invoice(db, invoice)
        _audit(db, "invoice", invoice_id, "delete", current_user.email)
    return RedirectResponse("/web/invoices", status_code=303)


@router.post("/invoices/{invoice_id}/draft-reminder")
def web_draft_reminder(invoice_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    invoice = invoice_service.get_invoice(db, invoice_id)
    if invoice is not None:
        try:
            communication_agent.draft_reminder(db, invoice)
        except Exception:
            logger.exception("Failed to draft reminder for invoice %s", invoice_id)
    return RedirectResponse(f"/web/invoices/{invoice_id}", status_code=303)


@router.post("/invoices/{invoice_id}/draft-invoice-email")
def web_draft_invoice_email(invoice_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    invoice = invoice_service.get_invoice(db, invoice_id)
    if invoice is not None:
        try:
            communication_agent.draft_invoice_email(db, invoice)
        except Exception:
            logger.exception("Failed to draft invoice email for invoice %s", invoice_id)
    return RedirectResponse(f"/web/invoices/{invoice_id}", status_code=303)


@router.post("/invoices/{invoice_id}/communications/{comm_id}/edit")
async def web_edit_communication(
    invoice_id: int, comm_id: int, request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login),
):
    """Requested directly: a drafted message was previously read-only
    (approve or reject as-is) - no way to fix a typo or reword something
    before it goes to Approvals. Same permission level as drafting one in
    the first place (require_login, not owner-only) since editing text
    isn't the risk-sensitive action - approving/sending it still is, and
    that gate (require_owner on /approvals/*) is untouched. Only allowed
    while status is still "draft"; a sent or rejected message is a
    historical record and shouldn't be rewritable after the fact."""
    form = await request.form()
    comm = db.get(Communication, comm_id)
    if comm is not None and comm.invoice_id == invoice_id and comm.status == "draft":
        comm.subject = form.get("subject", comm.subject)
        comm.body = form.get("body", comm.body)
        db.commit()
        _audit(db, "communication", comm.id, "edit", current_user.email)
    return RedirectResponse(f"/web/invoices/{invoice_id}", status_code=303)


# --- Recurring invoices -------------------------------------------------

@router.get("/recurring", response_class=HTMLResponse)
def web_list_recurring(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    templates_list = recurring_invoice_service.list_recurring_invoices(db, DEFAULT_ORG_ID)
    return templates.TemplateResponse("recurring_list.html", {
        "request": request, "templates_list": templates_list, "today": date.today(), "current_user": current_user,
    })


@router.get("/recurring/new", response_class=HTMLResponse)
def web_new_recurring_form(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    customers = db.query(Customer).filter(Customer.organization_id == DEFAULT_ORG_ID).all()
    return templates.TemplateResponse("recurring_form.html", {"request": request, "customers": customers, "error": None, "current_user": current_user})


@router.post("/recurring/new", response_class=HTMLResponse)
async def web_create_recurring(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    form = await request.form()
    items = _parse_items_from_form(form)
    customers = db.query(Customer).filter(Customer.organization_id == DEFAULT_ORG_ID).all()

    try:
        payload = RecurringInvoiceCreate(
            customer_id=int(form["customer_id"]),
            name=form["name"],
            frequency=form["frequency"],
            next_run_date=form["next_run_date"],
            due_days=int(form.get("due_days") or 30),
            tax=Decimal(form.get("tax") or "0"),
            discount=Decimal(form.get("discount") or "0"),
            currency=form.get("currency") or "USD",
            payment_terms=form.get("payment_terms") or "Net 30",
            items=items,
        )
    except (ValueError, KeyError) as e:
        return templates.TemplateResponse(
            "recurring_form.html",
            {"request": request, "customers": customers, "error": str(e), "current_user": current_user},
            status_code=422,
        )

    template = recurring_invoice_service.create_recurring_invoice(db, DEFAULT_ORG_ID, payload)
    _audit(db, "recurring_invoice", template.id, "create", current_user.email)
    return RedirectResponse("/web/recurring", status_code=303)


@router.post("/recurring/{recurring_id}/toggle")
def web_toggle_recurring(recurring_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    template = recurring_invoice_service.get_recurring_invoice(db, recurring_id)
    if template is not None:
        recurring_invoice_service.set_active(db, template, not template.is_active)
        _audit(db, "recurring_invoice", template.id, "toggle_active", current_user.email, {"is_active": template.is_active})
    return RedirectResponse("/web/recurring", status_code=303)


@router.post("/recurring/{recurring_id}/delete")
def web_delete_recurring(recurring_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    template = recurring_invoice_service.get_recurring_invoice(db, recurring_id)
    if template is not None:
        recurring_invoice_service.delete_recurring_invoice(db, template)
        _audit(db, "recurring_invoice", recurring_id, "delete", current_user.email)
    return RedirectResponse("/web/recurring", status_code=303)


@router.post("/recurring/generate-due")
def web_generate_due_recurring(db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    """The manual trigger for a person clicking "Generate due invoices
    now". Also the same route an external scheduler (Render Cron Jobs,
    or any host that can hit a URL on a schedule) would call for real
    automation - this app has no background job runner of its own, see
    recurring_invoice_service.py's module docstring."""
    created = recurring_invoice_service.generate_due_invoices(db, DEFAULT_ORG_ID)
    for invoice in created:
        _audit(db, "invoice", invoice.id, "create", "system (recurring)", {"recurring": True})
    return RedirectResponse("/web/recurring", status_code=303)


# --- Payments / Communications ---------------------------------------------

@router.get("/payments", response_class=HTMLResponse)
def web_payments(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    result = orchestrator.run_payment_scan(db, DEFAULT_ORG_ID)
    return templates.TemplateResponse("payments.html", {"request": request, "result": result, "current_user": current_user})


@router.get("/payments/narration/stream")
def web_stream_payment_narration(db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    """SSE twin of web_payments()'s narration paragraph. Recomputes the
    same ranking (rank_unpaid_invoices() is deterministic - no LLM in
    it) and streams a fresh narration of it live."""
    ranked, held = payment_ap_agent.rank_unpaid_invoices(db, DEFAULT_ORG_ID)

    def _generate():
        try:
            for chunk in payment_ap_agent.explain_payments_stream(ranked, held):
                yield f"data: {chunk.replace(chr(10), ' ')}\n\n"
        except llm_extraction_service.LLMUnavailableError:
            yield "data: [LLM unavailable]\n\n"
        yield "event: done\ndata: \n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.get("/collections", response_class=HTMLResponse)
def web_collections(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    result = orchestrator.run_collections_scan(db, DEFAULT_ORG_ID)
    return templates.TemplateResponse("collections.html", {"request": request, "result": result, "current_user": current_user})


@router.get("/collections/narration/stream")
def web_stream_collections_narration(db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    """SSE twin of web_collections()'s narration paragraph. Recomputes the
    same ranking (rank_outstanding_receivables() is deterministic - no
    LLM in it) and streams a fresh narration of it live."""
    ranked, escalate = collections_agent.rank_outstanding_receivables(db, DEFAULT_ORG_ID)

    def _generate():
        try:
            for chunk in collections_agent.explain_collections_stream(ranked, escalate):
                yield f"data: {chunk.replace(chr(10), ' ')}\n\n"
        except llm_extraction_service.LLMUnavailableError:
            yield "data: [LLM unavailable]\n\n"
        yield "event: done\ndata: \n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.get("/communications", response_class=HTMLResponse)
def web_communications(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    comms = db.query(Communication).order_by(Communication.created_at.desc()).all()
    return templates.TemplateResponse("communications.html", {"request": request, "comms": comms, "current_user": current_user})


# --- Approvals (owner-only decisions) ---------------------------------------

@router.get("/approvals", response_class=HTMLResponse)
def web_approvals(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    pending = db.query(ApprovalRequest).filter(ApprovalRequest.status == "pending").order_by(ApprovalRequest.created_at.asc()).all()
    decided = db.query(ApprovalRequest).filter(ApprovalRequest.status != "pending").order_by(ApprovalRequest.decided_at.desc()).limit(20).all()
    return templates.TemplateResponse(
        "approvals.html", {"request": request, "requests": pending, "decided": decided, "current_user": current_user}
    )


def _send_or_simulate(db: Session, comm: Communication) -> None:
    """The approval-time send step. Same simulated behavior as before
    (unchanged) when SMTP isn't configured. When it is, actually sends
    over SMTP - attaching the real generated invoice PDF (see
    invoice_pdf_service.py) for an outgoing invoice's email, since
    that's the whole point of sending this one for real. A real send
    failure marks the draft "send_failed" rather than pretending it
    went out - the human still sees it and can retry or send manually."""
    if not email_service.is_configured():
        comm.status = "sent"
        comm.sent_at = utcnow_naive()
        logger.info("SIMULATED SEND to %s: %s", comm.recipient, comm.subject)
        return

    attachment = None
    invoice = invoice_service.get_invoice(db, comm.invoice_id)
    if invoice is not None and invoice.direction == InvoiceDirection.outgoing:
        attachment = (f"{invoice.invoice_number}.pdf", invoice_pdf_service.generate_invoice_pdf(invoice))

    try:
        email_service.send_email(comm.recipient, comm.subject, comm.body, attachment=attachment)
        comm.status = "sent"
        comm.sent_at = utcnow_naive()
        logger.info("REAL SEND to %s: %s", comm.recipient, comm.subject)
    except email_service.EmailSendError:
        comm.status = "send_failed"
        logger.exception("Real email send failed for communication %s", comm.id)


def _decide_approval(db: Session, request_id: int, approve: bool, performed_by: str) -> None:
    req = db.get(ApprovalRequest, request_id)
    if req is None or req.status != "pending":
        return

    if req.type == "high_risk_invoice":
        invoice = invoice_service.get_invoice(db, req.related_id)
        if invoice is not None:
            invoice.invoice_status = InvoiceStatus.validated if approve else InvoiceStatus.rejected
    elif req.type == "send_communication":
        comm = db.get(Communication, req.related_id)
        if comm is not None and comm.status == "draft":
            if approve:
                _send_or_simulate(db, comm)
            else:
                comm.status = "rejected"

    req.status = "approved" if approve else "rejected"
    req.decided_at = utcnow_naive()
    db.commit()
    _audit(db, "approval_request", req.id, "approved" if approve else "rejected", performed_by, {"type": req.type, "related_id": req.related_id})


@router.post("/approvals/{request_id}/approve")
def web_approve(request_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_owner)):
    _decide_approval(db, request_id, approve=True, performed_by=current_user.email)
    return RedirectResponse("/web/approvals", status_code=303)


@router.post("/approvals/{request_id}/reject")
def web_reject(request_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_owner)):
    _decide_approval(db, request_id, approve=False, performed_by=current_user.email)
    return RedirectResponse("/web/approvals", status_code=303)


# --- AI Assistant ------------------------------------------------------------

@router.get("/assistant", response_class=HTMLResponse)
def web_assistant_form(request: Request, current_user: User = Depends(require_login)):
    return templates.TemplateResponse("assistant.html", {"request": request, "current_user": current_user})


@router.post("/assistant", response_class=HTMLResponse)
async def web_assistant_ask(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    form = await request.form()
    question = form.get("question", "").strip()
    result = orchestrator.route_query(db, DEFAULT_ORG_ID, question) if question else None
    return templates.TemplateResponse(
        "assistant.html", {"request": request, "question": question, "result": result, "current_user": current_user}
    )


# --- Agent Activity ----------------------------------------------------------

@router.get("/agent-activity", response_class=HTMLResponse)
def web_agent_activity(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    logs = db.query(AgentLog).order_by(AgentLog.created_at.desc()).limit(100).all()
    return templates.TemplateResponse("agent_activity.html", {"request": request, "logs": logs, "current_user": current_user})


# --- Audit Log -----------------------------------------------------------

@router.get("/audit-log", response_class=HTMLResponse)
def web_audit_log(request: Request, entity_type: str | None = None, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    """AuditLog rows are written on every invoice create/update/delete and
    every approval decision (app/web/routes.py::_audit) - this is the
    first page that actually lets a person browse them. The data existed
    from Phase 11 onward; there was just never a UI for it until now."""
    query = db.query(AuditLog).order_by(AuditLog.timestamp.desc())
    if entity_type:
        query = query.filter(AuditLog.entity_type == entity_type)
    entries = query.limit(200).all()
    return templates.TemplateResponse(
        "audit_log.html", {"request": request, "entries": entries, "entity_type": entity_type, "current_user": current_user}
    )


# --- Vendors / Customers --------------------------------------------------

@router.get("/vendors", response_class=HTMLResponse)
def web_list_vendors(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    vendors = db.query(Vendor).filter(Vendor.organization_id == DEFAULT_ORG_ID).order_by(Vendor.name).all()
    return templates.TemplateResponse("vendors_list.html", {"request": request, "vendors": vendors, "current_user": current_user})


@router.post("/vendors", response_class=HTMLResponse)
async def web_create_vendor(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    form = await request.form()
    vendor = Vendor(organization_id=DEFAULT_ORG_ID, **VendorCreate(
        name=form.get("name", ""), email=form.get("email") or None, address=form.get("address") or None,
        tax_id=form.get("tax_id") or None,
    ).model_dump())
    db.add(vendor)
    db.commit()
    _audit(db, "vendor", vendor.id, "create", current_user.email, {"name": vendor.name})
    return RedirectResponse("/web/vendors", status_code=303)


@router.get("/vendors/{vendor_id}", response_class=HTMLResponse)
def web_vendor_detail(vendor_id: int, request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    vendor = db.query(Vendor).filter(Vendor.id == vendor_id, Vendor.organization_id == DEFAULT_ORG_ID).first()
    if vendor is None:
        return RedirectResponse("/web/vendors", status_code=303)
    invoices = invoice_tools.search_invoices(db, DEFAULT_ORG_ID, party_name=vendor.name)
    invoices = [i for i in invoices if i.vendor_id == vendor_id]  # party_name search also matches customers by name
    total_billed = invoice_tools.format_money_by_currency(invoice_tools.totals_by_currency(invoices))
    risky = [i for i in invoices if i.risk_score is not None and float(i.risk_score) >= 0.5]
    return templates.TemplateResponse("vendor_detail.html", {
        "request": request, "vendor": vendor, "invoices": invoices, "total_billed": total_billed,
        "risky_count": len(risky), "current_user": current_user,
    })


@router.get("/customers", response_class=HTMLResponse)
def web_list_customers(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    customers = db.query(Customer).filter(Customer.organization_id == DEFAULT_ORG_ID).order_by(Customer.name).all()
    return templates.TemplateResponse("customers_list.html", {"request": request, "customers": customers, "current_user": current_user})


@router.post("/customers", response_class=HTMLResponse)
async def web_create_customer(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    form = await request.form()
    customer = Customer(organization_id=DEFAULT_ORG_ID, **CustomerCreate(
        name=form.get("name", ""), email=form.get("email") or None, address=form.get("address") or None,
    ).model_dump())
    db.add(customer)
    db.commit()
    _audit(db, "customer", customer.id, "create", current_user.email, {"name": customer.name})
    return RedirectResponse("/web/customers", status_code=303)


@router.get("/customers/{customer_id}", response_class=HTMLResponse)
def web_customer_detail(customer_id: int, request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    customer = db.query(Customer).filter(Customer.id == customer_id, Customer.organization_id == DEFAULT_ORG_ID).first()
    if customer is None:
        return RedirectResponse("/web/customers", status_code=303)
    invoices = invoice_tools.search_invoices(db, DEFAULT_ORG_ID, party_name=customer.name)
    invoices = [i for i in invoices if i.customer_id == customer_id]
    total_billed = invoice_tools.format_money_by_currency(invoice_tools.totals_by_currency(invoices))
    return templates.TemplateResponse("customer_detail.html", {
        "request": request, "customer": customer, "invoices": invoices, "total_billed": total_billed,
        "current_user": current_user,
    })


# --- (Upload/extraction now lives inline on the New Invoice page - see
# web_new_invoice_upload above, under "Invoices".) ------------------------


@router.get("/invoices/{invoice_id}/source-pdf")
def web_view_source_pdf(invoice_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    """The other half of the real gap found live: the Upload flow saved
    every PDF to disk but never linked it to the invoice it created, so
    there was no way to look at the original again. Only ever serves
    the exact filename already stored on the invoice record - never a
    filename taken from the request - so there's no path-traversal
    surface here at all."""
    invoice = invoice_service.get_invoice(db, invoice_id)
    if invoice is None or not invoice.source_pdf_filename:
        raise HTTPException(status_code=404, detail="No original PDF is linked to this invoice.")

    file_path = UPLOAD_DIR / invoice.source_pdf_filename
    if not file_path.exists():
        # Exactly the scenario this fix was built to make visible instead
        # of silent: a cloud redeploy wipes the ephemeral disk, but the
        # database record survives - so this is a clear message, not a
        # crash or a broken link that looks like a bug.
        raise HTTPException(
            status_code=404,
            detail="This invoice has a linked PDF, but the file is no longer on disk "
                   "(a free-tier cloud redeploy wipes locally-saved files - the extracted invoice data is unaffected).",
        )
    return FileResponse(file_path, media_type="application/pdf", filename=f"{invoice.invoice_number}.pdf")


@router.get("/invoices/{invoice_id}/pdf")
def web_download_invoice_pdf(invoice_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    """The outgoing-side counterpart to source-pdf above: generates a
    real PDF for an invoice WE issued to a customer (source-pdf is the
    opposite direction - a PDF a vendor already sent US). Only makes
    sense for outgoing invoices - an incoming one already has its own
    original document (source-pdf) if one was uploaded, and generating
    a fake "invoice we're sending" for money we owe a vendor would be
    actively wrong, not just unavailable."""
    invoice = invoice_service.get_invoice(db, invoice_id)
    if invoice is None:
        raise HTTPException(status_code=404, detail="Invoice not found.")
    if invoice.direction != InvoiceDirection.outgoing:
        raise HTTPException(status_code=400, detail="PDF generation is only available for outgoing (customer) invoices.")

    pdf_bytes = invoice_pdf_service.generate_invoice_pdf(invoice)
    return Response(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{invoice.invoice_number}.pdf"'},
    )


@router.post("/invoices/{invoice_id}/generate-pay-link")
def web_generate_pay_link(invoice_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    """Older outgoing invoices (created before public_token existed)
    have none yet - create_invoice() generates one automatically for
    every new one now, this is just the catch-up path for existing data."""
    invoice = invoice_service.get_invoice(db, invoice_id)
    if invoice is not None:
        invoice_service.get_or_create_public_token(db, invoice)
    return RedirectResponse(f"/web/invoices/{invoice_id}", status_code=303)


@router.post("/invoices/{invoice_id}/credit-notes/new")
async def web_create_credit_note(invoice_id: int, request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    form = await request.form()
    invoice = invoice_service.get_invoice(db, invoice_id)
    if invoice is not None:
        try:
            amount = Decimal(form.get("amount") or "0")
            credit_note_service.create_credit_note(
                db, DEFAULT_ORG_ID, invoice, reason=form.get("reason", "").strip(), amount=amount, created_by=current_user.email,
            )
            _audit(db, "credit_note", invoice.id, "create", current_user.email, {"amount": str(amount)})
        except (credit_note_service.CreditNoteError, InvalidOperation) as e:
            logger.warning("Credit note rejected for invoice %s: %s", invoice_id, e)
    return RedirectResponse(f"/web/invoices/{invoice_id}", status_code=303)


@router.get("/credit-notes/{credit_note_id}/pdf")
def web_download_credit_note_pdf(credit_note_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    note = db.get(CreditNote, credit_note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Credit note not found.")
    invoice = invoice_service.get_invoice(db, note.invoice_id)
    pdf_bytes = invoice_pdf_service.generate_credit_note_pdf(note, invoice)
    return Response(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{note.credit_note_number}.pdf"'},
    )
