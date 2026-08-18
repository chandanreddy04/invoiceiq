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

    def _fake_chat(model, messages, format=None, options=None):
        import json
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
