"""IRS Form 990 XML parser."""

import logging
from typing import Optional

from lxml import etree

from .models import DonorInfo, GrantRecipient

logger = logging.getLogger(__name__)


class Form990Parser:
    """Parser for IRS 990 XML filings."""

    NAMESPACES = {
        'irs': 'http://www.irs.gov/efile'
    }

    def parse_xml(self, xml_content: bytes) -> Optional[etree._Element]:
        """Parse XML content into an element tree.

        Args:
            xml_content: Raw XML bytes

        Returns:
            Root element or None if parsing fails
        """
        try:
            return etree.fromstring(xml_content)
        except etree.XMLSyntaxError as e:
            logger.error(f"Failed to parse XML: {e}")
            return None

    def parse_donor_info(self, root: etree._Element) -> DonorInfo:
        """Extract donor organization info from 990 XML.

        Args:
            root: Parsed XML root element

        Returns:
            DonorInfo dataclass with organization details
        """
        name = (
            self._get_text(root, ".//irs:BusinessName/irs:BusinessNameLine1Txt") or
            self._get_text(root, ".//irs:BusinessName/irs:BusinessNameLine1") or
            self._get_text(root, ".//irs:Filer/irs:BusinessName/irs:BusinessNameLine1Txt") or
            "Unknown"
        )

        ein = (
            self._get_text(root, ".//irs:Filer/irs:EIN") or
            self._get_text(root, ".//irs:EIN") or
            "Unknown"
        )

        city = (
            self._get_text(root, ".//irs:Filer/irs:USAddress/irs:CityNm") or
            self._get_text(root, ".//irs:USAddress/irs:CityNm")
        )

        state = (
            self._get_text(root, ".//irs:Filer/irs:USAddress/irs:StateAbbreviationCd") or
            self._get_text(root, ".//irs:USAddress/irs:StateAbbreviationCd")
        )

        tax_year = (
            self._get_text(root, ".//irs:TaxYr") or
            self._get_text(root, ".//irs:TaxYear")
        )

        # Determine form type
        form_type = None
        if root.find(".//irs:IRS990PF", self.NAMESPACES) is not None:
            form_type = "990-PF"
        elif root.find(".//irs:IRS990EZ", self.NAMESPACES) is not None:
            form_type = "990-EZ"
        elif root.find(".//irs:IRS990", self.NAMESPACES) is not None:
            form_type = "990"

        # Financial info
        total_grants = (
            self._get_int(root, ".//irs:GrantsAndSimilarAmtsCYAmt") or
            self._get_int(root, ".//irs:TotalGrantsAmt") or
            self._get_int(root, ".//irs:DistributionAmt")
        )

        total_revenue = (
            self._get_int(root, ".//irs:TotalRevenueAmt") or
            self._get_int(root, ".//irs:CYTotalRevenueAmt")
        )

        total_assets = (
            self._get_int(root, ".//irs:TotalAssetsEOYAmt") or
            self._get_int(root, ".//irs:TotalAssetsEOY")
        )

        return DonorInfo(
            name=name,
            ein=ein,
            city=city,
            state=state,
            tax_year=int(tax_year) if tax_year else None,
            form_type=form_type,
            total_grants=total_grants,
            total_revenue=total_revenue,
            total_assets=total_assets
        )

    def parse_grant_recipients(self, root: etree._Element) -> list[GrantRecipient]:
        """Extract all grant recipients from 990 XML.

        Args:
            root: Parsed XML root element

        Returns:
            List of GrantRecipient dataclasses
        """
        recipients = []

        # 990-PF: Part XV - Grants and Contributions Paid During the Year
        pf_grants = root.findall(".//irs:GrantOrContributionPdDurYrGrp", self.NAMESPACES)
        for grant in pf_grants:
            recipient = self._parse_pf_grant(grant)
            if recipient:
                recipients.append(recipient)

        # 990: Schedule I - Grants and Other Assistance
        schedule_i_grants = root.findall(".//irs:RecipientTable", self.NAMESPACES)
        for grant in schedule_i_grants:
            recipient = self._parse_schedule_i_grant(grant)
            if recipient:
                recipients.append(recipient)

        # International grants
        intl_grants = root.findall(".//irs:GrantsOtherAsstToOrgsOutsideUSGrp", self.NAMESPACES)
        for grant in intl_grants:
            recipient = self._parse_intl_grant(grant)
            if recipient:
                recipients.append(recipient)

        # Generic RecipientBusinessName paths
        recipient_groups = root.findall(".//irs:RecipientBusinessName/..", self.NAMESPACES)
        for group in recipient_groups:
            if group.tag.endswith("RecipientTable") or group.tag.endswith("Grp"):
                continue  # Already processed
            recipient = self._parse_generic_grant(group)
            if recipient and recipient.name:
                # Avoid duplicates
                if not any(r.name == recipient.name and r.amount == recipient.amount for r in recipients):
                    recipients.append(recipient)

        return recipients

    def _parse_pf_grant(self, grant_elem: etree._Element) -> Optional[GrantRecipient]:
        """Parse a 990-PF grant element."""
        name = (
            self._get_text(grant_elem, ".//irs:RecipientBusinessName/irs:BusinessNameLine1Txt") or
            self._get_text(grant_elem, ".//irs:RecipientBusinessName/irs:BusinessNameLine1") or
            self._get_text(grant_elem, ".//irs:RecipientPersonNm")
        )

        if not name:
            return None

        return GrantRecipient(
            name=name,
            ein=self._get_text(grant_elem, ".//irs:RecipientEIN"),
            city=(
                self._get_text(grant_elem, ".//irs:RecipientUSAddress/irs:CityNm") or
                self._get_text(grant_elem, ".//irs:RecipientForeignAddress/irs:CityNm")
            ),
            state=self._get_text(grant_elem, ".//irs:RecipientUSAddress/irs:StateAbbreviationCd"),
            zip_code=self._get_text(grant_elem, ".//irs:RecipientUSAddress/irs:ZIPCd"),
            amount=(
                self._get_int(grant_elem, ".//irs:Amt") or
                self._get_int(grant_elem, ".//irs:Amount")
            ),
            purpose=(
                self._get_text(grant_elem, ".//irs:GrantOrContributionPurposeTxt") or
                self._get_text(grant_elem, ".//irs:PurposeOfGrantTxt")
            ),
            relationship=self._get_text(grant_elem, ".//irs:RecipientRelationshipTxt")
        )

    def _parse_schedule_i_grant(self, grant_elem: etree._Element) -> Optional[GrantRecipient]:
        """Parse a Schedule I grant element."""
        name = (
            self._get_text(grant_elem, ".//irs:RecipientBusinessName/irs:BusinessNameLine1Txt") or
            self._get_text(grant_elem, ".//irs:RecipientBusinessName/irs:BusinessNameLine1")
        )

        if not name:
            return None

        return GrantRecipient(
            name=name,
            ein=(
                self._get_text(grant_elem, ".//irs:RecipientEIN") or
                self._get_text(grant_elem, ".//irs:EINOfRecipient")
            ),
            city=(
                self._get_text(grant_elem, ".//irs:USAddress/irs:CityNm") or
                self._get_text(grant_elem, ".//irs:RecipientUSAddress/irs:CityNm")
            ),
            state=(
                self._get_text(grant_elem, ".//irs:USAddress/irs:StateAbbreviationCd") or
                self._get_text(grant_elem, ".//irs:RecipientUSAddress/irs:StateAbbreviationCd")
            ),
            amount=(
                self._get_int(grant_elem, ".//irs:CashGrantAmt") or
                self._get_int(grant_elem, ".//irs:AmountOfCashGrant")
            ),
            purpose=(
                self._get_text(grant_elem, ".//irs:PurposeOfGrantTxt") or
                self._get_text(grant_elem, ".//irs:GrantOrContributionPurposeTxt")
            )
        )

    def _parse_intl_grant(self, grant_elem: etree._Element) -> Optional[GrantRecipient]:
        """Parse an international grant element."""
        name = self._get_text(grant_elem, ".//irs:RecipientBusinessName/irs:BusinessNameLine1Txt")

        if not name:
            return None

        return GrantRecipient(
            name=name,
            city=self._get_text(grant_elem, ".//irs:ForeignAddress/irs:CityNm"),
            amount=self._get_int(grant_elem, ".//irs:CashGrantAmt"),
            purpose=self._get_text(grant_elem, ".//irs:PurposeOfGrantTxt")
        )

    def _parse_generic_grant(self, grant_elem: etree._Element) -> Optional[GrantRecipient]:
        """Parse a generic grant element structure."""
        name = (
            self._get_text(grant_elem, ".//irs:RecipientBusinessName/irs:BusinessNameLine1Txt") or
            self._get_text(grant_elem, ".//irs:RecipientBusinessName/irs:BusinessNameLine1")
        )

        if not name:
            return None

        return GrantRecipient(
            name=name,
            ein=self._get_text(grant_elem, ".//irs:RecipientEIN"),
            city=self._get_text(grant_elem, ".//irs:CityNm"),
            state=self._get_text(grant_elem, ".//irs:StateAbbreviationCd"),
            amount=(
                self._get_int(grant_elem, ".//irs:Amt") or
                self._get_int(grant_elem, ".//irs:CashGrantAmt")
            ),
            purpose=self._get_text(grant_elem, ".//irs:PurposeOfGrantTxt")
        )

    def _get_text(self, elem: etree._Element, xpath: str) -> Optional[str]:
        """Get text content from an xpath, handling namespaces."""
        result = elem.find(xpath, self.NAMESPACES)
        if result is not None and result.text:
            return result.text.strip()
        return None

    def _get_int(self, elem: etree._Element, xpath: str) -> Optional[int]:
        """Get integer value from an xpath."""
        text = self._get_text(elem, xpath)
        if text:
            try:
                return int(text)
            except ValueError:
                return None
        return None
