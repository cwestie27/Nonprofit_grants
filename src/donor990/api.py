"""ProPublica Nonprofit Explorer API client."""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class ProPublicaAPI:
    """Client for ProPublica Nonprofit Explorer API."""

    BASE_URL = "https://projects.propublica.org/nonprofits/api/v2"
    IRS_XML_BASE = "https://s3.amazonaws.com/irs-form-990"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "donor990/1.0"
        })

    def get_organization(self, ein: str) -> Optional[dict]:
        """Get organization details by EIN.

        Args:
            ein: Employer Identification Number (with or without dash)

        Returns:
            Organization data dict or None if not found
        """
        ein_clean = ein.replace("-", "")
        url = f"{self.BASE_URL}/organizations/{ein_clean}.json"

        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning(f"ProPublica returned status {resp.status_code} for EIN {ein}")
                return None
        except requests.RequestException as e:
            logger.error(f"Error fetching organization {ein}: {e}")
            return None

    def get_filing_xml_url(self, org_data: dict) -> Optional[str]:
        """Extract the XML filing URL from organization data.

        First tries ProPublica's xml_url, then constructs IRS S3 URL from object_id.
        Note: The IRS S3 bucket has been emptied as of late 2023.

        Args:
            org_data: Organization data from get_organization()

        Returns:
            XML URL string or None if not found
        """
        filings = org_data.get("filings_with_data", [])

        if not filings:
            logger.debug("No filings with data found")
            return None

        for filing in filings:
            # First try ProPublica's xml_url
            xml_url = filing.get("xml_url")
            if xml_url:
                return xml_url

            # Try to construct URL from object_id (IRS S3 bucket - likely empty)
            object_id = filing.get("object_id")
            if object_id:
                constructed_url = f"{self.IRS_XML_BASE}/{object_id}_public.xml"
                try:
                    head_resp = self.session.head(constructed_url, timeout=5)
                    if head_resp.status_code == 200:
                        logger.info(f"Found XML on IRS S3 for year {filing.get('tax_prd_yr')}")
                        return constructed_url
                except requests.RequestException:
                    pass

        logger.debug("No XML filings found in ProPublica or IRS S3")
        return None
