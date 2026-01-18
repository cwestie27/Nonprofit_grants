"""
donor990 - Extract grant recipients from IRS 990 filings.

This package provides tools to look up nonprofit organizations and extract
information about grants they have made from their IRS Form 990 filings.
"""

from .models import DonorInfo, DonorResult, GrantRecipient
from .api import ProPublicaAPI
from .irs import IRSXMLFinder
from .parser import Form990Parser

__version__ = "0.1.0"
__all__ = [
    "DonorInfo",
    "DonorResult",
    "GrantRecipient",
    "ProPublicaAPI",
    "IRSXMLFinder",
    "Form990Parser",
]
