"""
Tests for the vendor-matching/auto-create logic added to the upload
review -> confirm flow: a real gap where an extracted vendor name (and
now address/tax ID) never got matched against - or used to create -
an actual Vendor record, leaving the dropdown always blank.
"""

import pytest

from app.web.routes import _match_vendor, _resolve_or_create_vendor
from app.services.validation_service import InvoiceValidationError
from app.models.models import Vendor


def test_match_vendor_by_exact_name(db_session, org, vendor):
    result = _match_vendor([vendor], vendor.name, None)
    assert result == vendor.id


def test_match_vendor_by_name_is_case_and_whitespace_insensitive(db_session, org, vendor):
    result = _match_vendor([vendor], f"  {vendor.name.upper()}  ", None)
    assert result == vendor.id


def test_match_vendor_returns_none_when_nothing_matches(db_session, org, vendor):
    assert _match_vendor([vendor], "Totally Different Co.", None) is None


def test_match_vendor_by_tax_id_wins_even_if_name_differs(db_session, org):
    """A franchise/rebrand/typo shouldn't defeat a real tax ID match -
    the strongest identity signal a business can have."""
    v = Vendor(organization_id=org.id, name="Acme Supply Co.", tax_id="91-2345678")
    db_session.add(v)
    db_session.commit()

    result = _match_vendor([v], "Acme Supply Company Inc.", "91-2345678")
    assert result == v.id


def test_match_vendor_tax_id_checked_before_name(db_session, org):
    """If tax ID matches one vendor but the name superficially matches
    a DIFFERENT vendor, tax ID must win - it's the stronger signal."""
    same_name_diff_vendor = Vendor(organization_id=org.id, name="Acme Supply Co.", tax_id="00-0000000")
    real_match = Vendor(organization_id=org.id, name="Acme Supply Co. (Old)", tax_id="91-2345678")
    db_session.add_all([same_name_diff_vendor, real_match])
    db_session.commit()

    result = _match_vendor([same_name_diff_vendor, real_match], "Acme Supply Co.", "91-2345678")
    assert result == real_match.id


def test_resolve_or_create_vendor_uses_explicit_form_selection(db_session, org, vendor):
    form = {"vendor_id": str(vendor.id), "extracted_vendor_name": "Some Other Name"}
    result = _resolve_or_create_vendor(db_session, org.id, form)
    assert result == vendor.id  # explicit selection always wins over extraction


def test_resolve_or_create_vendor_creates_new_vendor_when_unmatched(db_session, org):
    form = {
        "vendor_id": "",
        "extracted_vendor_name": "Brand New Vendor LLC",
        "extracted_vendor_address": "42 New St",
        "extracted_vendor_tax_id": "12-3456789",
    }
    new_id = _resolve_or_create_vendor(db_session, org.id, form)

    created = db_session.get(Vendor, new_id)
    assert created is not None
    assert created.name == "Brand New Vendor LLC"
    assert created.address == "42 New St"
    assert created.tax_id == "12-3456789"


def test_resolve_or_create_vendor_raises_when_neither_provided(db_session, org):
    with pytest.raises(InvoiceValidationError):
        _resolve_or_create_vendor(db_session, org.id, {"vendor_id": ""})
