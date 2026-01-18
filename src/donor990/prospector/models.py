"""Data models for the prospector module."""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class NonprofitProfile:
    """Structured nonprofit profile combining ProPublica + AI research."""
    name: str
    ein: Optional[str] = None
    website: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    ntee_code: Optional[str] = None
    ntee_description: Optional[str] = None
    annual_revenue: Optional[int] = None
    annual_expenses: Optional[int] = None
    total_assets: Optional[int] = None
    founding_year: Optional[int] = None
    mission_statement: Optional[str] = None
    programs: Optional[list] = None
    population_served: Optional[str] = None
    impact_statement: Optional[str] = None
    cause_area: Optional[str] = None
    data_sources: Optional[list] = None

    def to_dict(self):
        return asdict(self)

    def revenue_bracket(self) -> str:
        """Return human-readable revenue bracket."""
        if not self.annual_revenue:
            return "Unknown"
        r = self.annual_revenue
        if r < 100_000:
            return "Under $100K"
        elif r < 500_000:
            return "$100K - $500K"
        elif r < 1_000_000:
            return "$500K - $1M"
        elif r < 5_000_000:
            return "$1M - $5M"
        elif r < 10_000_000:
            return "$5M - $10M"
        else:
            return "Over $10M"


@dataclass
class SimilarNonprofit:
    """A nonprofit identified as similar to the input."""
    profile: NonprofitProfile
    similarity_score: float = 0.0
    similarity_reasons: Optional[list] = None
