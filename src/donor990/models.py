"""Data models for donor990 package."""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class GrantRecipient:
    """A recipient of a grant from a donor organization."""
    name: str
    ein: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    amount: Optional[int] = None
    purpose: Optional[str] = None
    relationship: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DonorInfo:
    """Information about a donor organization."""
    name: str
    ein: str
    city: Optional[str] = None
    state: Optional[str] = None
    tax_year: Optional[int] = None
    form_type: Optional[str] = None
    total_grants: Optional[int] = None
    total_revenue: Optional[int] = None
    total_assets: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DonorResult:
    """Result of processing a donor's 990 filing."""
    donor: DonorInfo
    recipients: list[GrantRecipient]
    error: Optional[str] = None

    @property
    def total_recipients(self) -> int:
        return len(self.recipients)

    @property
    def total_amount(self) -> int:
        return sum(r.amount or 0 for r in self.recipients)
