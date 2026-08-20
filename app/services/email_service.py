"""
Real SMTP email sending - config-gated the same way llm_client.py
switches between Ollama and Groq: unset SMTP_HOST means "not
configured", and every caller already has a working fallback for that
(the existing simulated-send behavior in _decide_approval(), unchanged
when this stays unconfigured). No new failure mode is introduced for
anyone who never sets these variables.

Deliberately plain smtplib, not a third-party email API (SendGrid,
Mailgun, etc.) - SMTP is the one sending mechanism every real mailbox
(Gmail, Outlook, a business's own domain) already supports out of the
box via an app password, with nothing further to sign up for.
"""

import logging
import smtplib
from email.message import EmailMessage

from app.core.config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM_EMAIL

logger = logging.getLogger(__name__)


class EmailSendError(Exception):
    """Raised when a real SMTP send was attempted (is_configured() was
    True) but failed - bad credentials, unreachable host, rejected
    recipient. Distinct from "not configured", which callers should
    check with is_configured() before ever calling send_email()."""


def is_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)


def send_email(to: str, subject: str, body: str, attachment: tuple[str, bytes] | None = None) -> None:
    """Sends one real email over SMTP with STARTTLS. `attachment`, if
    given, is (filename, bytes) - used to attach the real generated
    invoice PDF (see invoice_pdf_service.py) when actually sending an
    outgoing invoice for real, not just simulating it.

    Raises EmailSendError on any failure - callers decide what "the
    send failed" means for their own record (see _decide_approval()'s
    "send_failed" status), this function never silently swallows one."""
    if not is_configured():
        raise EmailSendError("SMTP is not configured (SMTP_HOST/SMTP_USER/SMTP_PASSWORD unset).")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM_EMAIL
    msg["To"] = to
    msg.set_content(body)

    if attachment is not None:
        filename, content = attachment
        msg.add_attachment(content, maintype="application", subtype="pdf", filename=filename)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        logger.warning("SMTP send to %s failed: %s", to, e)
        raise EmailSendError(str(e)) from e
