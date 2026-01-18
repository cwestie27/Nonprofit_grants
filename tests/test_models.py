"""Tests for data models."""

import pytest

from donor990.models import DonorInfo, DonorResult, GrantRecipient


class TestGrantRecipient:
    def test_create_minimal(self):
        recipient = GrantRecipient(name="Test Org")
        assert recipient.name == "Test Org"
        assert recipient.ein is None
        assert recipient.amount is None

    def test_create_full(self):
        recipient = GrantRecipient(
            name="Test Org",
            ein="12-3456789",
            city="New York",
            state="NY",
            amount=10000,
            purpose="General support"
        )
        assert recipient.name == "Test Org"
        assert recipient.ein == "12-3456789"
        assert recipient.amount == 10000

    def test_to_dict(self):
        recipient = GrantRecipient(name="Test", amount=5000)
        d = recipient.to_dict()
        assert d["name"] == "Test"
        assert d["amount"] == 5000
        assert d["ein"] is None


class TestDonorInfo:
    def test_create(self):
        donor = DonorInfo(
            name="Ford Foundation",
            ein="13-1684331",
            city="New York",
            state="NY",
            form_type="990-PF"
        )
        assert donor.name == "Ford Foundation"
        assert donor.form_type == "990-PF"


class TestDonorResult:
    def test_total_recipients(self):
        result = DonorResult(
            donor=DonorInfo(name="Test", ein="123"),
            recipients=[
                GrantRecipient(name="R1", amount=1000),
                GrantRecipient(name="R2", amount=2000),
            ]
        )
        assert result.total_recipients == 2

    def test_total_amount(self):
        result = DonorResult(
            donor=DonorInfo(name="Test", ein="123"),
            recipients=[
                GrantRecipient(name="R1", amount=1000),
                GrantRecipient(name="R2", amount=2000),
                GrantRecipient(name="R3"),  # No amount
            ]
        )
        assert result.total_amount == 3000

    def test_with_error(self):
        result = DonorResult(
            donor=DonorInfo(name="Unknown", ein="123"),
            recipients=[],
            error="Not found"
        )
        assert result.error == "Not found"
        assert result.total_recipients == 0
