"""
True integration tests through the real FastAPI app via TestClient -
actual HTTP requests, actual routing, actual response_model
validation. The database dependency is overridden to a fresh
in-memory DB per test; ollama.chat is mocked so these run fast.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.session import Base, get_db
from app.main import app
from app.models.models import Organization, Vendor, User, UserRole
from app.security.auth import hash_password

OWNER_EMAIL, OWNER_PASSWORD = "owner@test.example", "owner-pass-123"
BOOKKEEPER_EMAIL, BOOKKEEPER_PASSWORD = "bookkeeper@test.example", "bookkeeper-pass-123"


@pytest.fixture
def client(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    TestSessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # Seed org id=1, since routers hardcode DEFAULT_ORG_ID = 1
    db = TestSessionLocal()
    db.add(Organization(id=1, name="Test Org"))
    db.add(Vendor(id=1, organization_id=1, name="Test Vendor", email="v@test.example"))
    db.add(User(organization_id=1, email=OWNER_EMAIL, password_hash=hash_password(OWNER_PASSWORD), role=UserRole.owner))
    db.add(User(organization_id=1, email=BOOKKEEPER_EMAIL, password_hash=hash_password(BOOKKEEPER_PASSWORD), role=UserRole.bookkeeper))
    db.commit()
    db.close()

    def _fake_chat(model, messages, format=None, options=None, stream=False):
        import json
        if stream:
            return iter([{"message": {"content": "mocked "}}, {"message": {"content": "stream"}}])
        if format is not None:
            props = format.get("properties", {})
            fake = {}
            for name, spec in props.items():
                t = spec.get("type")
                fake[name] = "" if t == "string" else 0 if t in ("number", "integer") else False if t == "boolean" else [] if t == "array" else None
            return {"message": {"content": json.dumps(fake)}}
        return {"message": {"content": "mocked"}}

    import ollama
    monkeypatch.setattr(ollama, "chat", _fake_chat)

    with TestClient(app) as c:
        c.session_factory = TestSessionLocal  # lets tests inspect DB state the API doesn't return directly
        yield c

    app.dependency_overrides.clear()


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root_redirects_to_dashboard(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "/web/dashboard" in resp.headers["location"]


def test_create_and_get_invoice(client):
    payload = {
        "direction": "incoming", "invoice_number": "API-1", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 5, "discount": 0, "currency": "USD",
        "items": [{"description": "Item A", "quantity": 2, "unit_price": 10}],
    }
    create_resp = client.post("/invoices", json=payload)
    assert create_resp.status_code == 201
    body = create_resp.json()
    assert body["subtotal"] == "20.00"
    assert body["total"] == "25.00"

    get_resp = client.get(f"/invoices/{body['id']}")
    assert get_resp.status_code == 200
    assert get_resp.json()["invoice_number"] == "API-1"


def test_create_invoice_validation_error_returns_422(client):
    payload = {
        "direction": "incoming", "invoice_number": "BAD-1", "vendor_id": 1,
        "invoice_date": "2026-01-10", "due_date": "2026-01-01",  # due before invoice date
        "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "Item A", "quantity": 1, "unit_price": 10}],
    }
    resp = client.post("/invoices", json=payload)
    assert resp.status_code == 422
    assert "Due date" in resp.json()["detail"]


def test_get_nonexistent_invoice_returns_404(client):
    resp = client.get("/invoices/9999")
    assert resp.status_code == 404


def test_list_invoices_filters_by_direction(client):
    for i in range(2):
        client.post("/invoices", json={
            "direction": "incoming", "invoice_number": f"IN-{i}", "vendor_id": 1,
            "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
            "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
        })
    resp = client.get("/invoices", params={"direction": "incoming"})
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_delete_invoice(client):
    create_resp = client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "DEL-1", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
    })
    invoice_id = create_resp.json()["id"]
    del_resp = client.delete(f"/invoices/{invoice_id}")
    assert del_resp.status_code == 204
    assert client.get(f"/invoices/{invoice_id}").status_code == 404


def _login(client, email, password):
    return client.post(
        "/web/login", data={"email": email, "password": password, "next": "/web/dashboard"}, follow_redirects=False
    )


def test_dashboard_requires_login(client):
    resp = client.get("/web/dashboard", follow_redirects=False)
    assert resp.status_code == 303
    assert "/web/login" in resp.headers["location"]


def test_login_with_correct_credentials_then_dashboard_loads(client):
    login_resp = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    assert login_resp.status_code == 303

    resp = client.get("/web/dashboard")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text


def test_login_with_wrong_password_rejected(client):
    resp = _login(client, OWNER_EMAIL, "wrong-password")
    assert resp.status_code == 401
    assert "Invalid" in resp.text

    # and the dashboard should still be inaccessible
    dash_resp = client.get("/web/dashboard", follow_redirects=False)
    assert dash_resp.status_code == 303


def test_logout_clears_session(client):
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    assert client.get("/web/dashboard").status_code == 200

    client.get("/web/logout")
    resp = client.get("/web/dashboard", follow_redirects=False)
    assert resp.status_code == 303


def test_owner_can_access_approvals_actions(client):
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    # No pending approval to act on, but the route itself should not 403 an owner
    resp = client.post("/web/approvals/9999/approve", follow_redirects=False)
    assert resp.status_code == 303  # redirects back to /web/approvals either way


def test_bookkeeper_cannot_access_approvals_actions(client):
    _login(client, BOOKKEEPER_EMAIL, BOOKKEEPER_PASSWORD)
    resp = client.post("/web/approvals/9999/approve")
    assert resp.status_code == 403


def test_creating_invoice_via_web_ui_writes_audit_log(client):
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    resp = client.post("/web/invoices/new", data={
        "direction": "incoming", "invoice_number": "AUDIT-1", "vendor_id": "1",
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": "0", "discount": "0", "currency": "USD",
        "item_description_0": "Widget", "item_quantity_0": "1", "item_unit_price_0": "10",
    }, follow_redirects=False)
    assert resp.status_code == 303

    from app.models.models import AuditLog
    db = client.session_factory()
    try:
        entry = db.query(AuditLog).filter(AuditLog.entity_type == "invoice", AuditLog.action == "create").first()
        assert entry is not None
        assert entry.performed_by == OWNER_EMAIL
    finally:
        db.close()


def test_bulk_mark_paid_marks_selected_invoices_and_skips_unselected(client):
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)

    ids = []
    for n in ("BULK-1", "BULK-2", "BULK-3"):
        resp = client.post("/invoices", json={
            "direction": "incoming", "invoice_number": n, "vendor_id": 1,
            "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
            "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
        })
        ids.append(resp.json()["id"])

    resp = client.post(
        "/web/invoices/bulk-mark-paid",
        data={"invoice_ids": [str(ids[0]), str(ids[1])]},
        follow_redirects=False,
    )
    assert resp.status_code == 303

    marked = client.get(f"/invoices/{ids[0]}").json()
    also_marked = client.get(f"/invoices/{ids[1]}").json()
    untouched = client.get(f"/invoices/{ids[2]}").json()
    assert marked["payment_status"] == "paid"
    assert also_marked["payment_status"] == "paid"
    assert untouched["payment_status"] == "unpaid"


def test_owner_can_override_invoice_status_back_and_forth(client):
    """Regression test for a real gap: invoice_status was write-once
    outside of creation - the Approvals page can only decide a request
    once, and there was no other way to change it back. An owner should
    be able to move an invoice between any status, in either direction."""
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    resp = client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "OVERRIDE-1", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
    })
    invoice_id = resp.json()["id"]
    assert resp.json()["invoice_status"] == "validated"

    resp = client.post(f"/web/invoices/{invoice_id}/override-status", data={"new_status": "rejected"}, follow_redirects=False)
    assert resp.status_code == 303
    assert client.get(f"/invoices/{invoice_id}").json()["invoice_status"] == "rejected"

    # And back again - this is the actual gap being fixed, not just a one-way change
    resp = client.post(f"/web/invoices/{invoice_id}/override-status", data={"new_status": "validated"}, follow_redirects=False)
    assert resp.status_code == 303
    assert client.get(f"/invoices/{invoice_id}").json()["invoice_status"] == "validated"

    from app.models.models import AuditLog
    db = client.session_factory()
    try:
        entry = db.query(AuditLog).filter(AuditLog.entity_type == "invoice", AuditLog.action == "status_override").first()
        assert entry is not None
        assert entry.performed_by == OWNER_EMAIL
    finally:
        db.close()


def test_bookkeeper_cannot_override_invoice_status(client):
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    resp = client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "OVERRIDE-2", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
    })
    invoice_id = resp.json()["id"]

    _login(client, BOOKKEEPER_EMAIL, BOOKKEEPER_PASSWORD)
    resp = client.post(f"/web/invoices/{invoice_id}/override-status", data={"new_status": "rejected"})
    assert resp.status_code == 403
    assert client.get(f"/invoices/{invoice_id}").json()["invoice_status"] == "validated"


def test_invoice_detail_page_renders_with_communication_and_shows_full_body(client, mock_ollama_chat):
    """Regression test for a real Jinja TemplateSyntaxError (mismatched
    {% if %}/{% endif %} left over from an edit) that made this exact page
    return a 500 for any invoice with at least one drafted communication -
    every other test happened to create invoices without ever GETting
    this page with comms present, so 89 passing tests missed it entirely.
    Also covers the actual gap reported live: the invoice detail page
    used to show only a communication's status/subject/date, never the
    body - you had to leave the page to read what was actually drafted."""
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    resp = client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "COMM-DETAIL-1", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
    })
    invoice_id = resp.json()["id"]

    draft_resp = client.post(f"/web/invoices/{invoice_id}/draft-reminder", follow_redirects=False)
    assert draft_resp.status_code == 303

    detail_resp = client.get(f"/web/invoices/{invoice_id}")
    assert detail_resp.status_code == 200

    from app.models.models import Communication
    db = client.session_factory()
    try:
        comm = db.query(Communication).filter(Communication.invoice_id == invoice_id).first()
        assert comm is not None
        assert comm.body in detail_resp.text  # full content visible on the page itself, not just a summary row
    finally:
        db.close()


def test_owner_can_edit_a_draft_communication_before_approval(client, mock_ollama_chat):
    """Requested directly after the visibility fix: viewing a draft's
    content isn't enough - a typo or wording issue should be fixable
    before it goes to Approvals, without regenerating the whole draft."""
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    resp = client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "COMM-EDIT-1", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
    })
    invoice_id = resp.json()["id"]
    client.post(f"/web/invoices/{invoice_id}/draft-reminder")

    from app.models.models import Communication
    db = client.session_factory()
    try:
        comm = db.query(Communication).filter(Communication.invoice_id == invoice_id).first()
        comm_id = comm.id
    finally:
        db.close()

    edit_resp = client.post(
        f"/web/invoices/{invoice_id}/communications/{comm_id}/edit",
        data={"subject": "Edited subject line", "body": "Edited body text."},
        follow_redirects=False,
    )
    assert edit_resp.status_code == 303

    db = client.session_factory()
    try:
        comm = db.get(Communication, comm_id)
        assert comm.subject == "Edited subject line"
        assert comm.body == "Edited body text."
    finally:
        db.close()

    from app.models.models import AuditLog
    db = client.session_factory()
    try:
        entry = db.query(AuditLog).filter(AuditLog.entity_type == "communication", AuditLog.action == "edit").first()
        assert entry is not None
        assert entry.performed_by == OWNER_EMAIL
    finally:
        db.close()


def test_editing_a_sent_communication_is_a_no_op(client, mock_ollama_chat):
    """A sent/rejected message is a historical record - editing it after
    the fact would let someone quietly rewrite what was already approved
    and (simulated-)sent, which defeats the point of the approval gate."""
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    resp = client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "COMM-EDIT-2", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
    })
    invoice_id = resp.json()["id"]
    client.post(f"/web/invoices/{invoice_id}/draft-reminder")

    from app.models.models import Communication, ApprovalRequest
    db = client.session_factory()
    try:
        comm = db.query(Communication).filter(Communication.invoice_id == invoice_id).first()
        comm_id = comm.id
        original_body = comm.body
        approval = db.query(ApprovalRequest).filter(
            ApprovalRequest.type == "send_communication", ApprovalRequest.related_id == comm_id
        ).first()
        approval_id = approval.id
    finally:
        db.close()

    client.post(f"/web/approvals/{approval_id}/approve")

    client.post(
        f"/web/invoices/{invoice_id}/communications/{comm_id}/edit",
        data={"subject": "Should not apply", "body": "Should not apply."},
    )

    db = client.session_factory()
    try:
        comm = db.get(Communication, comm_id)
        assert comm.status == "sent"
        assert comm.body == original_body
    finally:
        db.close()


def test_create_vendor_via_web_ui_and_view_detail_page(client):
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    resp = client.post("/web/vendors", data={"name": "New Vendor Co.", "email": "nv@test.example"}, follow_redirects=False)
    assert resp.status_code == 303

    list_resp = client.get("/web/vendors")
    assert "New Vendor Co." in list_resp.text

    from app.models.models import Vendor
    db = client.session_factory()
    try:
        vendor = db.query(Vendor).filter(Vendor.name == "New Vendor Co.").first()
        assert vendor is not None
    finally:
        db.close()

    detail_resp = client.get(f"/web/vendors/{vendor.id}")
    assert detail_resp.status_code == 200
    assert "New Vendor Co." in detail_resp.text

    from app.models.models import AuditLog
    db = client.session_factory()
    try:
        entry = db.query(AuditLog).filter(AuditLog.entity_type == "vendor", AuditLog.action == "create").first()
        assert entry is not None
    finally:
        db.close()


def test_create_customer_via_web_ui_and_view_detail_page(client):
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    resp = client.post("/web/customers", data={"name": "New Customer LLC"}, follow_redirects=False)
    assert resp.status_code == 303

    from app.models.models import Customer
    db = client.session_factory()
    try:
        customer = db.query(Customer).filter(Customer.name == "New Customer LLC").first()
        assert customer is not None
    finally:
        db.close()

    detail_resp = client.get(f"/web/customers/{customer.id}")
    assert detail_resp.status_code == 200
    assert "New Customer LLC" in detail_resp.text


def test_audit_log_page_lists_entries_and_filters_by_entity_type(client):
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    client.post("/web/vendors", data={"name": "Audit Test Vendor"})

    resp = client.get("/web/audit-log")
    assert resp.status_code == 200
    assert "Audit Test Vendor" in resp.text or "vendor" in resp.text

    filtered = client.get("/web/audit-log", params={"entity_type": "vendor"})
    assert filtered.status_code == 200

    filtered_out = client.get("/web/audit-log", params={"entity_type": "communication"})
    assert "Audit Test Vendor" not in filtered_out.text


def test_source_pdf_is_served_when_linked_and_present(client):
    """Regression test for a real gap: the Upload flow saved every PDF
    to disk but never recorded where, so nothing could find it again.
    This proves the fix end-to-end - link a filename to an invoice at
    creation, then confirm the invoice actually serves that file back."""
    from app.core.config import UPLOAD_DIR

    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    fake_pdf = UPLOAD_DIR / "test-source.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content for testing")
    try:
        resp = client.post("/invoices", json={
            "direction": "incoming", "invoice_number": "PDF-LINK-1", "vendor_id": 1,
            "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
            "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
            "source_pdf_filename": "test-source.pdf",
        })
        invoice_id = resp.json()["id"]
        assert resp.json()["source_pdf_filename"] == "test-source.pdf"

        pdf_resp = client.get(f"/web/invoices/{invoice_id}/source-pdf")
        assert pdf_resp.status_code == 200
        assert pdf_resp.headers["content-type"] == "application/pdf"
        assert pdf_resp.content == b"%PDF-1.4 fake content for testing"
    finally:
        fake_pdf.unlink(missing_ok=True)


def test_source_pdf_returns_404_when_no_pdf_linked(client):
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    resp = client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "NO-PDF-1", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
    })
    invoice_id = resp.json()["id"]
    assert resp.json()["source_pdf_filename"] is None

    pdf_resp = client.get(f"/web/invoices/{invoice_id}/source-pdf")
    assert pdf_resp.status_code == 404
    assert "No original PDF" in pdf_resp.json()["detail"]


def test_source_pdf_returns_clear_404_when_linked_but_file_missing_from_disk(client):
    """The exact scenario this fix documents rather than hides: a cloud
    redeploy wipes the ephemeral disk, but the database record (and the
    rest of the invoice) survives untouched. This should read as a clear
    message, not a crash or a broken-looking link."""
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    resp = client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "MISSING-PDF-1", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
        "source_pdf_filename": "never-actually-saved.pdf",
    })
    invoice_id = resp.json()["id"]

    pdf_resp = client.get(f"/web/invoices/{invoice_id}/source-pdf")
    assert pdf_resp.status_code == 404
    assert "no longer on disk" in pdf_resp.json()["detail"]


def test_export_invoices_csv_contains_real_data_and_respects_filters(client):
    import csv, io

    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "CSV-EXPORT-1", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "Widget", "quantity": 2, "unit_price": 10}],
    })
    client.post("/invoices", json={
        "direction": "outgoing", "invoice_number": "CSV-EXPORT-2", "customer_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "Service", "quantity": 1, "unit_price": 50}],
    })

    resp = client.get("/web/invoices/export.csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "attachment" in resp.headers["content-disposition"]

    rows = list(csv.reader(io.StringIO(resp.text)))
    header, data_rows = rows[0], rows[1:]
    assert header[0] == "Invoice Number"
    numbers = [r[0] for r in data_rows]
    assert "CSV-EXPORT-1" in numbers
    assert "CSV-EXPORT-2" in numbers

    # Filtered export should only include the incoming one
    filtered = client.get("/web/invoices/export.csv", params={"direction": "incoming"})
    filtered_rows = list(csv.reader(io.StringIO(filtered.text)))[1:]
    filtered_numbers = [r[0] for r in filtered_rows]
    assert "CSV-EXPORT-1" in filtered_numbers
    assert "CSV-EXPORT-2" not in filtered_numbers


def test_export_general_ledger_csv_breaks_out_by_line_item(client):
    import csv, io

    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "GL-EXPORT-1", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [
            {"description": "Item A", "quantity": 1, "unit_price": 10},
            {"description": "Item B", "quantity": 2, "unit_price": 5},
        ],
    })

    resp = client.get("/web/invoices/export/general-ledger.csv")
    assert resp.status_code == 200
    rows = list(csv.reader(io.StringIO(resp.text)))
    header, data_rows = rows[0], rows[1:]
    assert header[0] == "Invoice Number"
    assert header[3] == "Line Description"

    gl_rows = [r for r in data_rows if r[0] == "GL-EXPORT-1"]
    assert len(gl_rows) == 2  # one row per line item, not one per invoice
    descriptions = {r[3] for r in gl_rows}
    assert descriptions == {"Item A", "Item B"}


def test_risk_explanation_stream_returns_sse_chunks(client):
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    resp = client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "STREAM-1", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
    })
    invoice_id = resp.json()["id"]

    stream_resp = client.get(f"/web/invoices/{invoice_id}/risk-explanation/stream")
    assert stream_resp.status_code == 200
    assert stream_resp.headers["content-type"].startswith("text/event-stream")
    assert "data: mocked" in stream_resp.text
    assert "event: done" in stream_resp.text


def test_payment_narration_stream_returns_sse_chunks(client):
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "STREAM-PAY-1", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-05", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
    })

    stream_resp = client.get("/web/payments/narration/stream")
    assert stream_resp.status_code == 200
    assert stream_resp.headers["content-type"].startswith("text/event-stream")
    assert "event: done" in stream_resp.text


def test_json_api_is_not_gated_by_login(client):
    """Documented limitation (see README): the JSON API has no auth of
    its own, unlike the web UI. Asserted explicitly (not just implied
    by other tests never calling _login()) so that if auth is added to
    the API later, this test fails loudly and has to be updated
    on purpose, rather than nobody noticing the behavior changed."""
    resp = client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "NO-AUTH-1", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
    })
    assert resp.status_code == 201  # succeeds with zero authentication


def _seed_customer(client) -> int:
    from app.models.models import Customer
    db = client.session_factory()
    try:
        c = Customer(organization_id=1, name="PDF Test Customer", email="pdftest@test.example", address="1 Test St")
        db.add(c)
        db.commit()
        db.refresh(c)
        return c.id
    finally:
        db.close()


def test_download_invoice_pdf_returns_real_pdf_for_outgoing_invoice(client):
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    customer_id = _seed_customer(client)
    resp = client.post("/invoices", json={
        "direction": "outgoing", "invoice_number": "PDF-ROUTE-1", "customer_id": customer_id,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "Consulting", "quantity": 1, "unit_price": 500}],
    })
    invoice_id = resp.json()["id"]

    pdf_resp = client.get(f"/web/invoices/{invoice_id}/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert pdf_resp.content[:5] == b"%PDF-"


def test_download_invoice_pdf_rejects_incoming_invoice(client):
    """A vendor invoice was never ours to "generate" - it already has
    its own original document (source-pdf) if one was uploaded."""
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    resp = client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "PDF-ROUTE-2", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
    })
    invoice_id = resp.json()["id"]

    pdf_resp = client.get(f"/web/invoices/{invoice_id}/pdf")
    assert pdf_resp.status_code == 400
    assert "outgoing" in pdf_resp.json()["detail"].lower()


def test_download_invoice_pdf_404s_for_missing_invoice(client):
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    resp = client.get("/web/invoices/999999/pdf")
    assert resp.status_code == 404


def test_draft_invoice_email_route_creates_pending_approval(client):
    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    customer_id = _seed_customer(client)
    resp = client.post("/invoices", json={
        "direction": "outgoing", "invoice_number": "EMAIL-ROUTE-1", "customer_id": customer_id,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "Consulting", "quantity": 1, "unit_price": 500}],
    })
    invoice_id = resp.json()["id"]

    draft_resp = client.post(f"/web/invoices/{invoice_id}/draft-invoice-email", follow_redirects=False)
    assert draft_resp.status_code == 303
    assert draft_resp.headers["location"] == f"/web/invoices/{invoice_id}"

    from app.models.models import Communication, ApprovalRequest
    db = client.session_factory()
    try:
        comm = db.query(Communication).filter(Communication.invoice_id == invoice_id).first()
        assert comm is not None
        assert comm.status == "draft"
        req = db.query(ApprovalRequest).filter(ApprovalRequest.related_id == comm.id, ApprovalRequest.type == "send_communication").first()
        assert req is not None
        assert req.status == "pending"
    finally:
        db.close()


def _get_pending_send_approval_id(client, comm_id):
    from app.models.models import ApprovalRequest
    db = client.session_factory()
    try:
        return db.query(ApprovalRequest).filter(
            ApprovalRequest.type == "send_communication", ApprovalRequest.related_id == comm_id, ApprovalRequest.status == "pending"
        ).first().id
    finally:
        db.close()


def test_approving_communication_stays_simulated_when_smtp_not_configured(client, mock_ollama_chat, monkeypatch):
    """Default behavior (no SMTP_* env vars) must stay exactly as it
    always was - approving a draft marks it "sent" without any real
    network call."""
    from app.services import email_service
    monkeypatch.setattr(email_service, "SMTP_HOST", "")

    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    resp = client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "SMTP-OFF-1", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
    })
    invoice_id = resp.json()["id"]
    client.post(f"/web/invoices/{invoice_id}/draft-reminder")

    from app.models.models import Communication
    db = client.session_factory()
    comm_id = db.query(Communication).filter(Communication.invoice_id == invoice_id).first().id
    db.close()

    client.post(f"/web/approvals/{_get_pending_send_approval_id(client, comm_id)}/approve")

    db = client.session_factory()
    try:
        comm = db.query(Communication).filter(Communication.id == comm_id).first()
        assert comm.status == "sent"
    finally:
        db.close()


def test_approving_outgoing_invoice_communication_sends_real_email_with_pdf_when_smtp_configured(client, mock_ollama_chat, monkeypatch):
    from app.services import email_service
    monkeypatch.setattr(email_service, "SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(email_service, "SMTP_USER", "biz@example.com")
    monkeypatch.setattr(email_service, "SMTP_PASSWORD", "app-password")

    sent_calls = []

    def fake_send_email(to, subject, body, attachment=None):
        sent_calls.append({"to": to, "subject": subject, "attachment": attachment})

    from app.web import routes
    monkeypatch.setattr(routes.email_service, "send_email", fake_send_email)

    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    customer_id = _seed_customer(client)
    resp = client.post("/invoices", json={
        "direction": "outgoing", "invoice_number": "SMTP-ON-1", "customer_id": customer_id,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "Consulting", "quantity": 1, "unit_price": 500}],
    })
    invoice_id = resp.json()["id"]
    client.post(f"/web/invoices/{invoice_id}/draft-invoice-email")

    from app.models.models import Communication
    db = client.session_factory()
    comm_id = db.query(Communication).filter(Communication.invoice_id == invoice_id).first().id
    db.close()

    client.post(f"/web/approvals/{_get_pending_send_approval_id(client, comm_id)}/approve")

    assert len(sent_calls) == 1
    assert sent_calls[0]["to"] == "pdftest@test.example"
    filename, pdf_bytes = sent_calls[0]["attachment"]
    assert filename == "SMTP-ON-1.pdf"
    assert pdf_bytes[:5] == b"%PDF-"

    db = client.session_factory()
    try:
        comm = db.query(Communication).filter(Communication.id == comm_id).first()
        assert comm.status == "sent"
    finally:
        db.close()


def test_approving_communication_marks_send_failed_on_real_smtp_error(client, mock_ollama_chat, monkeypatch):
    from app.services import email_service
    monkeypatch.setattr(email_service, "SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(email_service, "SMTP_USER", "biz@example.com")
    monkeypatch.setattr(email_service, "SMTP_PASSWORD", "wrong-password")

    def failing_send_email(*a, **kw):
        raise email_service.EmailSendError("535 Authentication failed")

    from app.web import routes
    monkeypatch.setattr(routes.email_service, "send_email", failing_send_email)

    _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    resp = client.post("/invoices", json={
        "direction": "incoming", "invoice_number": "SMTP-FAIL-1", "vendor_id": 1,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "X", "quantity": 1, "unit_price": 5}],
    })
    invoice_id = resp.json()["id"]
    client.post(f"/web/invoices/{invoice_id}/draft-reminder")

    from app.models.models import Communication
    db = client.session_factory()
    comm_id = db.query(Communication).filter(Communication.invoice_id == invoice_id).first().id
    db.close()

    client.post(f"/web/approvals/{_get_pending_send_approval_id(client, comm_id)}/approve")

    db = client.session_factory()
    try:
        comm = db.query(Communication).filter(Communication.id == comm_id).first()
        assert comm.status == "send_failed"
    finally:
        db.close()


def _create_outgoing_invoice_get_token(client, invoice_number="PUBLIC-PAGE-1"):
    customer_id = _seed_customer(client)
    resp = client.post("/invoices", json={
        "direction": "outgoing", "invoice_number": invoice_number, "customer_id": customer_id,
        "invoice_date": "2026-01-01", "due_date": "2026-01-31", "tax": 0, "discount": 0, "currency": "USD",
        "items": [{"description": "Consulting", "quantity": 1, "unit_price": 500}],
    })
    body = resp.json()
    return body["id"], body["public_token"]


def test_public_pay_page_shows_real_invoice_no_login_required(client):
    """No _login() call anywhere in this test - that's the point."""
    invoice_id, token = _create_outgoing_invoice_get_token(client)
    resp = client.get(f"/pay/{token}")
    assert resp.status_code == 200
    assert "PUBLIC-PAGE-1" in resp.text
    assert "500" in resp.text


def test_public_pay_page_404s_for_unknown_token(client):
    resp = client.get("/pay/not-a-real-token-at-all")
    assert resp.status_code == 404


def test_public_pay_page_never_shows_the_internal_staff_nav(client):
    """The whole reason pay.html doesn't extend base.html - a customer
    must never see Approvals/Agent Activity/Audit Log links."""
    invoice_id, token = _create_outgoing_invoice_get_token(client)
    resp = client.get(f"/pay/{token}")
    assert "Agent Activity" not in resp.text
    assert "Approvals" not in resp.text


def test_public_pdf_download_works_without_login(client):
    invoice_id, token = _create_outgoing_invoice_get_token(client)
    resp = client.get(f"/pay/{token}/pdf")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:5] == b"%PDF-"


def test_public_checkout_returns_400_when_stripe_not_configured(client):
    invoice_id, token = _create_outgoing_invoice_get_token(client)
    resp = client.post(f"/pay/{token}/checkout")
    assert resp.status_code == 400


def test_public_checkout_redirects_to_stripe_when_configured(client, monkeypatch):
    from app.services import payment_service
    monkeypatch.setattr(payment_service, "STRIPE_SECRET_KEY", "sk_test_fake")
    monkeypatch.setattr(payment_service, "create_checkout_session", lambda invoice, success_url, cancel_url: "https://checkout.stripe.com/fake")

    invoice_id, token = _create_outgoing_invoice_get_token(client)
    resp = client.post(f"/pay/{token}/checkout", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "https://checkout.stripe.com/fake"


def test_public_checkout_skips_stripe_when_already_paid(client, monkeypatch):
    from app.services import payment_service
    monkeypatch.setattr(payment_service, "STRIPE_SECRET_KEY", "sk_test_fake")
    called = []
    monkeypatch.setattr(payment_service, "create_checkout_session", lambda *a, **kw: called.append(1) or "https://x")

    invoice_id, token = _create_outgoing_invoice_get_token(client)
    db = client.session_factory()
    from app.models.models import Invoice, PaymentStatus
    inv = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    inv.payment_status = PaymentStatus.paid
    db.commit()
    db.close()

    resp = client.post(f"/pay/{token}/checkout", follow_redirects=False)
    assert resp.status_code == 303
    assert called == []  # never even attempted a new checkout for an already-paid invoice


def test_stripe_webhook_marks_invoice_paid_on_valid_signed_event(client, monkeypatch):
    import json
    import stripe as stripe_sdk
    from app.services import payment_service

    monkeypatch.setattr(payment_service, "STRIPE_WEBHOOK_SECRET", "whsec_test_secret")

    invoice_id, token = _create_outgoing_invoice_get_token(client, invoice_number="WEBHOOK-PAID-1")

    payload = json.dumps({
        "id": "evt_1", "type": "checkout.session.completed",
        "data": {"object": {"id": "cs_1", "metadata": {"invoice_id": str(invoice_id)}}},
    })
    sig = stripe_sdk.WebhookSignature.generate_signature_header(payload, "whsec_test_secret")

    resp = client.post("/stripe/webhook", content=payload, headers={"stripe-signature": sig})
    assert resp.status_code == 200

    from app.models.models import Invoice
    db = client.session_factory()
    try:
        inv = db.query(Invoice).filter(Invoice.id == invoice_id).first()
        assert inv.payment_status.value == "paid"
    finally:
        db.close()


def test_stripe_webhook_rejects_unsigned_request_and_does_not_mark_paid(client, monkeypatch):
    """The security-critical case: a POST with no valid Stripe signature
    must never be able to mark an invoice paid for free."""
    import json
    from app.services import payment_service

    monkeypatch.setattr(payment_service, "STRIPE_WEBHOOK_SECRET", "whsec_real_secret")

    invoice_id, token = _create_outgoing_invoice_get_token(client, invoice_number="WEBHOOK-SPOOF-1")

    payload = json.dumps({
        "id": "evt_2", "type": "checkout.session.completed",
        "data": {"object": {"id": "cs_2", "metadata": {"invoice_id": str(invoice_id)}}},
    })
    resp = client.post("/stripe/webhook", content=payload, headers={"stripe-signature": "fake-unsigned-header"})
    assert resp.status_code == 400

    from app.models.models import Invoice
    db = client.session_factory()
    try:
        inv = db.query(Invoice).filter(Invoice.id == invoice_id).first()
        assert inv.payment_status.value == "unpaid"
    finally:
        db.close()
