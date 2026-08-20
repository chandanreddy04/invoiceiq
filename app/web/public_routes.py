"""
Public, unauthenticated routes - the one part of this app deliberately
reachable without a staff login, because the person on the other end
of a /pay/<token> link is a customer, not a member of staff. Access
control here is the token itself (unguessable, see
invoice_service.get_or_create_public_token()), never require_login.

Nothing here ever trusts client input to decide an invoice is paid -
only the signature-verified Stripe webhook does that (see
payment_service.verify_and_parse_webhook()). The /success redirect page
is purely informational; a customer could navigate to it directly
without paying anything and it would change nothing.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.core.config import APP_BASE_URL
from app.database.session import get_db
from app.models.models import PaymentStatus
from app.services import invoice_service, invoice_pdf_service, payment_service
from app.tools import invoice_tools

router = APIRouter(tags=["public"])
templates = Jinja2Templates(directory="app/templates")
templates.env.filters["money"] = invoice_tools.format_money_by_currency
logger = logging.getLogger(__name__)


def _base_url(request: Request) -> str:
    """Prefers the configured public origin (correct behind Render's
    proxy, where request.base_url can report the wrong scheme/host) -
    falls back to the request's own view of itself for local dev where
    APP_BASE_URL is often left unset."""
    return APP_BASE_URL.rstrip("/") if APP_BASE_URL else str(request.base_url).rstrip("/")


@router.get("/pay/{token}", response_class=HTMLResponse)
def public_view_invoice(token: str, request: Request, just_paid: bool = False, just_cancelled: bool = False, db: Session = Depends(get_db)):
    invoice = invoice_service.get_invoice_by_public_token(db, token)
    if invoice is None:
        raise HTTPException(status_code=404, detail="Invoice not found.")
    return templates.TemplateResponse("pay.html", {
        "request": request, "invoice": invoice, "business_name": invoice_pdf_service.BUSINESS_NAME,
        "payments_enabled": payment_service.is_configured(),
        "just_paid": just_paid, "just_cancelled": just_cancelled,
    })


@router.get("/pay/{token}/pdf")
def public_download_invoice_pdf(token: str, db: Session = Depends(get_db)):
    invoice = invoice_service.get_invoice_by_public_token(db, token)
    if invoice is None:
        raise HTTPException(status_code=404, detail="Invoice not found.")
    pdf_bytes = invoice_pdf_service.generate_invoice_pdf(invoice)
    return Response(
        content=pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{invoice.invoice_number}.pdf"'},
    )


@router.post("/pay/{token}/checkout")
def public_start_checkout(token: str, request: Request, db: Session = Depends(get_db)):
    invoice = invoice_service.get_invoice_by_public_token(db, token)
    if invoice is None:
        raise HTTPException(status_code=404, detail="Invoice not found.")
    if invoice.payment_status == PaymentStatus.paid:
        return RedirectResponse(f"/pay/{token}", status_code=303)
    if not payment_service.is_configured():
        raise HTTPException(status_code=400, detail="Online payment is not configured for this business.")

    base = _base_url(request)
    try:
        checkout_url = payment_service.create_checkout_session(
            invoice,
            success_url=f"{base}/pay/{token}?just_paid=true",
            cancel_url=f"{base}/pay/{token}?just_cancelled=true",
        )
    except payment_service.PaymentServiceError:
        logger.exception("Stripe Checkout Session creation failed for invoice %s", invoice.id)
        raise HTTPException(status_code=502, detail="Could not start checkout right now - please try again shortly.")

    return RedirectResponse(checkout_url, status_code=303)


@router.post("/stripe/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = payment_service.verify_and_parse_webhook(payload, sig_header)
    except payment_service.PaymentServiceError as e:
        logger.warning("Rejected webhook request: %s", e)
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")

    if event["type"] == "checkout.session.completed":
        # Stripe's SDK objects (StripeObject) support "in" and "[]" like
        # a dict, but NOT .get() - it raises AttributeError rather than
        # silently returning None, which is exactly why this doesn't use
        # session.get("metadata", {}).get("invoice_id") (the obvious but
        # wrong plain-dict idiom).
        session = event["data"]["object"]
        metadata = session["metadata"] if "metadata" in session else {}
        invoice_id = metadata["invoice_id"] if "invoice_id" in metadata else None
        if invoice_id:
            invoice = invoice_service.get_invoice(db, int(invoice_id))
            if invoice is not None and invoice.payment_status != PaymentStatus.paid:
                invoice.payment_status = PaymentStatus.paid
                db.commit()
                session_id = session["id"] if "id" in session else "?"
                logger.info("Invoice %s marked paid via Stripe webhook (session %s)", invoice_id, session_id)

    return {"received": True}
