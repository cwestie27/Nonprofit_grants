"""IRS TEOS XML finder and downloader."""

import csv
import io
import logging
import zipfile
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class IRSXMLFinder:
    """Find and download 990 XML filings from IRS TEOS bulk archives.

    The AWS S3 bucket (irs-form-990) was emptied in late 2023.
    This class downloads XML from IRS bulk ZIP archives instead.
    """

    IRS_BASE = "https://apps.irs.gov/pub/epostcard/990/xml"

    def __init__(self, cache_dir: Optional[Path] = None, timeout: int = 300):
        """Initialize the IRS XML finder.

        Args:
            cache_dir: Directory to cache downloaded ZIP files. If None, no disk caching.
            timeout: Request timeout in seconds for large downloads.
        """
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "donor990/1.0"
        })
        self._index_cache: dict[int, list] = {}

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def find_filing_info(self, ein: str, year: int = 2023) -> Optional[dict]:
        """Find filing info for an EIN in the IRS index.

        Args:
            ein: Employer Identification Number
            year: Starting year to search (will try previous years too)

        Returns:
            Dict with filing details (year, name, return_type, object_id, tax_period)
            or None if not found
        """
        ein_clean = ein.replace("-", "")

        # Check multiple years
        for try_year in [year, year - 1, year - 2, 2021, 2020]:
            index_data = self._get_index(try_year)
            if index_data:
                matches = [row for row in index_data if row.get("EIN") == ein_clean]
                if matches:
                    # Sort by tax period to get most recent
                    matches.sort(key=lambda x: x.get("TAX_PERIOD", ""), reverse=True)
                    row = matches[0]
                    object_id = row.get("OBJECT_ID", "").strip()

                    if object_id:
                        return {
                            "year": try_year,
                            "name": row.get("TAXPAYER_NAME", "Unknown"),
                            "return_type": row.get("RETURN_TYPE", "Unknown"),
                            "object_id": object_id,
                            "tax_period": row.get("TAX_PERIOD", ""),
                        }

        return None

    def download_xml(self, filing_info: dict, progress_callback=None) -> Optional[bytes]:
        """Download the XML file from IRS bulk ZIP archives.

        Args:
            filing_info: Dict with year, object_id from find_filing_info()
            progress_callback: Optional callback(zip_name, status) for progress updates

        Returns:
            XML content as bytes, or None if not found
        """
        year = filing_info.get("year")
        object_id = filing_info.get("object_id", "").strip()

        if not year or not object_id:
            logger.error("Missing year or object_id in filing_info")
            return None

        xml_filename = f"{object_id}_public.xml"

        # Try to determine the month from the object_id (format: YYYYMM...)
        month_guess = None
        if len(object_id) >= 6 and object_id[:4].isdigit():
            potential_month = object_id[4:6]
            if potential_month.isdigit() and 1 <= int(potential_month) <= 12:
                month_guess = int(potential_month)

        # Build list of ZIPs to try (prioritize guessed month)
        parts = ["A", "B", "C", "D"]
        zips_to_try = []

        if month_guess:
            month_str = f"{month_guess:02d}"
            for part in parts:
                zips_to_try.append(f"{year}_TEOS_XML_{month_str}{part}.zip")

        # Then try all other months
        for month in range(1, 13):
            month_str = f"{month:02d}"
            for part in parts:
                zip_name = f"{year}_TEOS_XML_{month_str}{part}.zip"
                if zip_name not in zips_to_try:
                    zips_to_try.append(zip_name)

        # Try each ZIP file
        for zip_name in zips_to_try:
            if progress_callback:
                progress_callback(zip_name, "searching")

            xml_content = self._try_zip_for_xml(year, zip_name, xml_filename)
            if xml_content:
                if progress_callback:
                    progress_callback(zip_name, "found")
                return xml_content

        logger.warning(f"XML file {xml_filename} not found in any ZIP archive")
        return None

    def _try_zip_for_xml(self, year: int, zip_name: str, xml_filename: str) -> Optional[bytes]:
        """Try to find and extract an XML file from a specific ZIP."""
        import tempfile
        import os as os_module

        zip_url = f"{self.IRS_BASE}/{year}/{zip_name}"
        zip_path: Optional[Path] = None
        is_temp = False

        # Check disk cache first
        if self.cache_dir:
            cache_file = self.cache_dir / zip_name
            if cache_file.exists():
                logger.debug(f"Using cached {zip_name}")
                zip_path = cache_file
            else:
                # Download to cache
                if not self._download_zip(zip_url, cache_file):
                    return None
                zip_path = cache_file
        else:
            # Download to temp file
            fd, temp_path = tempfile.mkstemp(suffix='.zip')
            os_module.close(fd)
            zip_path = Path(temp_path)
            is_temp = True

            if not self._download_zip(zip_url, zip_path):
                if zip_path.exists():
                    zip_path.unlink()
                return None

        # Try to extract the XML from the ZIP (reading from disk)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if name.endswith(xml_filename) or name == xml_filename:
                        logger.info(f"Found {xml_filename} in {zip_name}")
                        xml_content = zf.read(name)
                        # Clean up temp file if not using cache
                        if is_temp and zip_path.exists():
                            zip_path.unlink()
                        return xml_content
        except zipfile.BadZipFile as e:
            logger.error(f"Error reading ZIP {zip_name}: {e}")

        # Clean up temp file if not using cache and XML not found
        if is_temp and zip_path and zip_path.exists():
            zip_path.unlink()

        return None

    def _download_zip(self, url: str, target_file: Path) -> bool:
        """Download a ZIP file from URL to disk using streaming.

        Args:
            url: URL to download from
            target_file: Path to save the ZIP file

        Returns:
            True if download succeeded, False otherwise
        """
        zip_name = url.split('/')[-1]
        logger.info(f"Downloading {zip_name}...")

        try:
            resp = self.session.get(url, timeout=self.timeout, stream=True)
            if resp.status_code != 200:
                logger.debug(f"ZIP not found: {url}")
                return False

            # Get content length if available
            total_size = int(resp.headers.get('content-length', 0))

            # Download in chunks
            downloaded = 0
            chunk_size = 8 * 1024 * 1024  # 8MB chunks

            with open(target_file, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            size_mb = downloaded / 1024 / 1024
                            print(f"\r    Downloading {zip_name}: {size_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

            print()  # Newline after progress
            size_mb = downloaded / 1024 / 1024
            logger.info(f"Downloaded {size_mb:.1f} MB")

            return True

        except requests.RequestException as e:
            logger.error(f"Error downloading ZIP: {e}")
            # Clean up partial download
            if target_file.exists():
                target_file.unlink()
            return False

    def _get_index(self, year: int) -> list:
        """Download and parse the index CSV for a year."""
        if year in self._index_cache:
            return self._index_cache[year]

        index_url = f"{self.IRS_BASE}/{year}/index_{year}.csv"
        logger.info(f"Loading IRS index for {year}...")

        try:
            resp = self.session.get(index_url, timeout=120)
            if resp.status_code == 200:
                content = resp.content.decode("utf-8", errors="ignore")
                reader = csv.DictReader(io.StringIO(content))
                rows = list(reader)
                self._index_cache[year] = rows
                logger.info(f"Loaded {len(rows):,} filings from {year} index")
                return rows
            else:
                logger.debug(f"No index found for {year}")
                return []
        except requests.RequestException as e:
            logger.error(f"Error loading {year} index: {e}")
            return []

    def get_download_instructions(self, filing_info: dict) -> str:
        """Get manual download instructions for a filing."""
        year = filing_info.get("year", 2023)
        object_id = filing_info.get("object_id", "")

        return f"""
To download this 990 XML file manually:

1. Go to: https://www.irs.gov/charities-non-profits/form-990-series-downloads

2. Download the index file for {year}:
   https://apps.irs.gov/pub/epostcard/990/xml/{year}/index_{year}.csv

3. Search the index for Object ID: {object_id}
   This will tell you which monthly ZIP file contains the XML.

4. Download the appropriate ZIP file and extract the XML.
   The XML filename will be: {object_id}_public.xml
"""
