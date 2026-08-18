"""
Shared test fixtures. Every test gets its own fresh in-memory SQLite
database - never the real dev database at data/invoiceiq.db - so
tests can't corrupt real (or other tests') data and can run in any
order.

Most tests never touch the LLM at all (they test the deterministic
reasoning layers directly - see each agent's own docstring for why
that split exists). The few that do exercise the full pipeline mock
`ollama.chat` via monkeypatch so the suite runs in seconds, not
minutes. Tests marked @pytest.mark.llm are the exception - they call
the real local model and are slower by design (see test_llm_e2e.py).
"""

import json
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.session import Base
from app.models import models  # noqa: F401  (registers models with Base)
from app.models.models import Organization, Vendor, Customer


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    TestSessionLocal = sessionmaker(bind=engine)
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def org(db_session):
    o = Organization(name="Test Bakery Co.")
    db_session.add(o)
    db_session.commit()
    db_session.refresh(o)
    return o


@pytest.fixture
def vendor(db_session, org):
    v = Vendor(organization_id=org.id, name="Test Vendor", email="vendor@test.example")
    db_session.add(v)
    db_session.commit()
    db_session.refresh(v)
    return v


@pytest.fixture
def customer(db_session, org):
    c = Customer(organization_id=org.id, name="Test Customer", email="customer@test.example")
    db_session.add(c)
    db_session.commit()
    db_session.refresh(c)
    return c


@pytest.fixture
def mock_ollama_chat(monkeypatch):
    """Replaces ollama.chat everywhere it's imported (extraction,
    classification, fraud explanation, communication, financial
    analysis agents) with a canned response, so pipeline/integration
    tests run in milliseconds instead of ~30s of real inference."""

    def _fake_chat(model, messages, format=None, options=None, stream=False):
        if stream:
            return iter([{"message": {"content": "Mocked "}}, {"message": {"content": "explanation text."}}])
        if format is not None:
            # A structured-output call - return something that validates
            # against most of our schemas: empty-ish but well-typed.
            schema = format
            props = schema.get("properties", {})
            fake = {}
            for name, spec in props.items():
                t = spec.get("type")
                if t == "string":
                    fake[name] = ""
                elif t == "number" or t == "integer":
                    fake[name] = 0
                elif t == "boolean":
                    fake[name] = False
                elif t == "array":
                    fake[name] = []
                else:
                    fake[name] = None
            content = json.dumps(fake)
        else:
            content = "Mocked explanation text."
        return {"message": {"content": content}}

    import ollama
    monkeypatch.setattr(ollama, "chat", _fake_chat)
    return _fake_chat


@pytest.fixture
def today():
    return date.today()
