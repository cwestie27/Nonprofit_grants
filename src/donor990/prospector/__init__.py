"""Prospector module - find similar nonprofits and their donors."""

from .models import NonprofitProfile, SimilarNonprofit
from .core import DonorProspector
from .ntee_codes import NTEE_CODES, get_ntee_description, codes_match

__all__ = [
    "NonprofitProfile",
    "SimilarNonprofit",
    "DonorProspector",
    "NTEE_CODES",
    "get_ntee_description",
    "codes_match",
]
