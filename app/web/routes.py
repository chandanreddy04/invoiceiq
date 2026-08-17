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

import json
import logging
import uuid
from decimal import Decimal, InvalidOperation

from fastapi import APIRouter, Depends, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.core.config import UPLOAD_DIR, MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB
from app.utils.time import utcnow_naive
from app.database.session import get_db
from app.models.models import (
    Customer, Vendor, InvoiceDirection, PaymentStatus, InvoiceStatus,
    FraudFlag, AgentLog, Communication, ApprovalRequest, User, AuditLog,
)
from app.schemas.invoice import InvoiceCreate, InvoiceItemCreate, InvoiceUpdate
from app.services import invoice_service, extraction_service, llm_extraction_service
from app.services.validation_service import InvoiceValidationError
from app.agents import orchestrator, communication_agent, payment_ap_agent
from app.tools import invoice_tools
from app.security.auth import verify_password, create_session_token
from app.security.deps import require_login, require_owner, get_current_user, SESSION_COOKIE_NAME

router = APIRouter(prefix="/web", tags=["web"])
templates = Jinja2Templates(directory="app/templates")
logger = logging.getLogger(__name__)

DEFAULT_ORG_ID = 1


def _parse_items_from_form(form) -> list[InvoiceItemCreate]:
    items = []
    for i in range(8):
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
    db: Session = Depends(get_db),
    current_user: User = Depends(require_login),
):
    invoices = invoice_service.list_invoices(
        db,
        DEFAULT_ORG_ID,
        InvoiceDirection(direction) if direction else None,
        PaymentStatus(payment_status) if payment_status else None,
    )
    return templates.TemplateResponse(
        "invoices_list.html",
        {"request": request, "invoices": invoices, "direction": direction, "payment_status": payment_status, "current_user": current_user},
    )


@router.get("/invoices/new", response_class=HTMLResponse)
def web_new_invoice_form(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    vendors = db.query(Vendor).filter(Vendor.organization_id == DEFAULT_ORG_ID).all()
    customers = db.query(Customer).filter(Customer.organization_id == DEFAULT_ORG_ID).all()
    return templates.TemplateResponse(
        "invoice_form.html",
        {"request": request, "invoice": None, "vendors": vendors, "customers": customers, "values": {},
         "form_action": "/web/invoices/new", "current_user": current_user},
    )


@router.post("/invoices/new", response_class=HTMLResponse)
async def web_create_invoice(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    form = await request.form()
    items = _parse_items_from_form(form)
    vendors = db.query(Vendor).filter(Vendor.organization_id == DEFAULT_ORG_ID).all()
    customers = db.query(Customer).filter(Customer.organization_id == DEFAULT_ORG_ID).all()

    try:
        payload = InvoiceCreate(
            direction=form["direction"],
            invoice_number=form["invoice_number"],
            vendor_id=int(form["vendor_id"]) if form.get("vendor_id") else None,
            customer_id=int(form["customer_id"]) if form.get("customer_id") else None,
            invoice_date=form["invoice_date"],
            due_date=form["due_date"],
            tax=Decimal(form.get("tax") or "0"),
            discount=Decimal(form.get("discount") or "0"),
            currency=(form.get("currency") or "USD").upper(),
            payment_terms=form.get("payment_terms") or "Net 30",
            items=items,
        )
        invoice = invoice_service.create_invoice(db, DEFAULT_ORG_ID, payload)
        _audit(db, "invoice", invoice.id, "create", current_user.email, {"invoice_number": invoice.invoice_number})
    except (InvoiceValidationError, ValueError) as e:
        return templates.TemplateResponse(
            "invoice_form.html",
            {
                "request": request, "invoice": None, "vendors": vendors, "customers": customers,
                "values": dict(form), "form_action": "/web/invoices/new", "error": str(e), "current_user": current_user,
            },
            status_code=422,
        )
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
    return templates.TemplateResponse(
        "invoice_form.html",
        {
            "request": request, "invoice": invoice, "vendors": vendors, "customers": customers,
            "values": {}, "form_action": f"/web/invoices/{invoice_id}", "fraud_flag": fraud_flag, "comms": comms,
            "current_user": current_user,
        },
    )


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


# --- Payments / Communications ---------------------------------------------

@router.get("/payments", response_class=HTMLResponse)
def web_payments(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    result = payment_ap_agent.prioritize_payments(db, DEFAULT_ORG_ID)
    return templates.TemplateResponse("payments.html", {"request": request, "result": result, "current_user": current_user})


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
                comm.status = "sent"
                comm.sent_at = utcnow_naive()
                logger.info("SIMULATED SEND to %s: %s", comm.recipient, comm.subject)
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


# --- Upload / extraction ------------------------------------------------------

@router.get("/upload", response_class=HTMLResponse)
def web_upload_form(request: Request, current_user: User = Depends(require_login)):
    return templates.TemplateResponse("upload.html", {"request": request, "current_user": current_user})


@router.post("/upload", response_class=HTMLResponse)
async def web_upload_extract(
    request: Request, file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(require_login)
):
    # Section 16/58: reject anything that isn't a PDF, and enforce a
    # size limit, before doing any real work with the upload.
    if not (file.filename or "").lower().endswith(".pdf"):
        return templates.TemplateResponse(
            "upload.html", {"request": request, "current_user": current_user, "error": "Only PDF files are supported."},
            status_code=422,
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        return templates.TemplateResponse(
            "upload.html",
            {"request": request, "current_user": current_user,
             "error": f"File is larger than the {MAX_UPLOAD_SIZE_MB}MB limit."},
            status_code=422,
        )

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
        return templates.TemplateResponse(
            "upload.html",
            {"request": request, "current_user": current_user, "error": "Could not read this PDF - it may be corrupted or password-protected."},
            status_code=422,
        )

    try:
        llm_result = llm_extraction_service.extract_invoice_with_llm(raw_text)
        extracted = {
            "method": "llm",
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
            "invoice_number": naive.invoice_number,
            "invoice_date": naive.invoice_date.isoformat() if naive.invoice_date else None,
            "due_date": naive.due_date.isoformat() if naive.due_date else None,
            "currency": "USD",
            "tax": 0,
            "discount": 0,
            "line_items": [],
            "total_hint": naive.total,
        }

    vendors = db.query(Vendor).filter(Vendor.organization_id == DEFAULT_ORG_ID).all()

    return templates.TemplateResponse(
        "upload_review.html",
        {"request": request, "raw_text": raw_text, "fields": extracted, "vendors": vendors, "current_user": current_user},
    )


@router.post("/upload/confirm", response_class=HTMLResponse)
async def web_upload_confirm(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_login)):
    form = await request.form()
    items = _parse_items_from_form(form)
    vendors = db.query(Vendor).filter(Vendor.organization_id == DEFAULT_ORG_ID).all()

    try:
        payload = InvoiceCreate(
            direction=InvoiceDirection.incoming,
            invoice_number=form["invoice_number"],
            vendor_id=int(form["vendor_id"]),
            customer_id=None,
            invoice_date=form["invoice_date"],
            due_date=form["due_date"],
            tax=Decimal(form.get("tax") or "0"),
            discount=Decimal(form.get("discount") or "0"),
            currency=(form.get("currency") or "USD").upper(),
            payment_terms=form.get("payment_terms") or "Net 30",
            items=items,
        )
        invoice = invoice_service.create_invoice(db, DEFAULT_ORG_ID, payload)
        _audit(db, "invoice", invoice.id, "create", current_user.email, {"invoice_number": invoice.invoice_number, "source": "upload"})
    except (InvoiceValidationError, ValueError) as e:
        retry_fields = {
            "method": "retry",
            "invoice_number": form.get("invoice_number"),
            "invoice_date": form.get("invoice_date"),
            "due_date": form.get("due_date"),
            "currency": form.get("currency") or "USD",
            "tax": form.get("tax") or 0,
            "discount": form.get("discount") or 0,
            "line_items": [item.model_dump() for item in items],
            "total_hint": None,
        }
        return templates.TemplateResponse(
            "upload_review.html",
            {"request": request, "raw_text": "", "fields": retry_fields, "vendors": vendors, "error": str(e), "current_user": current_user},
            status_code=422,
        )
    return RedirectResponse(f"/web/invoices/{invoice.id}", status_code=303)
