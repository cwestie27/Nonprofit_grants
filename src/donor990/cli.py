"""Command-line interface for donor990."""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import requests

from .api import ProPublicaAPI
from .irs import IRSXMLFinder
from .models import DonorInfo, DonorResult, GrantRecipient
from .parser import Form990Parser

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler()]
    )
    # Quiet down requests library
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def load_config(config_path: Path) -> list[str]:
    """Load EINs from a config file.

    Supports JSON (list or {"eins": [...]}) or plain text (one EIN per line).
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    content = config_path.read_text().strip()

    # Try JSON first
    if config_path.suffix == ".json" or content.startswith("[") or content.startswith("{"):
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "eins" in data:
            return data["eins"]
        else:
            raise ValueError("JSON config must be a list or have an 'eins' key")

    # Plain text: one EIN per line
    return [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]


def process_ein(
    ein: str,
    api: ProPublicaAPI,
    irs_finder: IRSXMLFinder,
    parser: Form990Parser,
    verbose: bool = False
) -> DonorResult:
    """Process a single EIN and extract grant recipients.

    Args:
        ein: Employer Identification Number
        api: ProPublica API client
        irs_finder: IRS XML finder
        parser: Form 990 parser
        verbose: Whether to print verbose output

    Returns:
        DonorResult with donor info and recipients
    """
    ein_clean = ein.replace("-", "")

    # Step 1: Get organization info from ProPublica
    if verbose:
        print(f"\n[Step 1] Fetching organization data from ProPublica...")

    org_data = api.get_organization(ein_clean)

    if not org_data:
        return DonorResult(
            donor=DonorInfo(name="Unknown", ein=ein_clean),
            recipients=[],
            error="Organization not found in ProPublica"
        )

    org_info = org_data.get("organization", {})
    if verbose:
        print(f"    Found: {org_info.get('name', 'Unknown')}")
        print(f"    Location: {org_info.get('city', '')}, {org_info.get('state', '')}")

    # Step 2: Find XML 990 filing
    if verbose:
        print(f"\n[Step 2] Finding XML 990 filing...")

    xml_content = None

    # First try ProPublica
    xml_url = api.get_filing_xml_url(org_data)
    if xml_url:
        if verbose:
            print(f"    Trying ProPublica URL...")
        try:
            resp = requests.get(xml_url, timeout=60, allow_redirects=True)
            if resp.status_code == 200:
                xml_content = resp.content
                if verbose:
                    print(f"    [OK] Fetched from ProPublica")
        except requests.RequestException:
            pass

    # If not found, search IRS index
    if not xml_content:
        if verbose:
            print("    Searching IRS TEOS index...")

        filing_info = irs_finder.find_filing_info(ein_clean, year=2023)

        if filing_info:
            if verbose:
                print(f"    Found: {filing_info['name']}")
                print(f"    Year: {filing_info['year']}, Type: {filing_info['return_type']}")
                print(f"    Object ID: {filing_info['object_id']}")
                print(f"\n[Step 3] Downloading XML from IRS archives...")

            def progress_cb(zip_name, status):
                if verbose:
                    if status == "searching":
                        print(f"    Trying {zip_name}...", end=" ", flush=True)
                    elif status == "found":
                        print(f"[OK]")

            xml_content = irs_finder.download_xml(filing_info, progress_callback=progress_cb)

            if not xml_content:
                return DonorResult(
                    donor=DonorInfo(name=org_info.get("name", "Unknown"), ein=ein_clean),
                    recipients=[],
                    error="XML download failed"
                )
        else:
            return DonorResult(
                donor=DonorInfo(name=org_info.get("name", "Unknown"), ein=ein_clean),
                recipients=[],
                error="No e-filed 990 found in IRS index"
            )

    if not xml_content:
        return DonorResult(
            donor=DonorInfo(name=org_info.get("name", "Unknown"), ein=ein_clean),
            recipients=[],
            error="No XML filing available"
        )

    # Step 4: Parse the XML
    if verbose:
        print(f"\n[Step 4] Parsing 990 XML...")

    root = parser.parse_xml(xml_content)
    if root is None:
        return DonorResult(
            donor=DonorInfo(name=org_info.get("name", "Unknown"), ein=ein_clean),
            recipients=[],
            error="Could not parse XML"
        )

    # Extract donor info
    donor_info = parser.parse_donor_info(root)
    if verbose:
        print(f"    Form Type: {donor_info.form_type}")
        print(f"    Tax Year: {donor_info.tax_year}")
        if donor_info.total_grants:
            print(f"    Total Grants: ${donor_info.total_grants:,}")

    # Step 5: Extract grant recipients
    if verbose:
        print(f"\n[Step 5] Extracting grant recipients...")

    recipients = parser.parse_grant_recipients(root)
    if verbose:
        print(f"    Found {len(recipients)} grant recipients")

    return DonorResult(donor=donor_info, recipients=recipients)


def write_csv(results: list[DonorResult], output_path: Path):
    """Write results to CSV file."""
    fieldnames = [
        "donor_name", "donor_ein", "donor_city", "donor_state",
        "tax_year", "form_type",
        "recipient_name", "recipient_ein", "recipient_city",
        "recipient_state", "recipient_zip", "amount", "purpose", "relationship"
    ]

    rows = []
    for result in results:
        if result.error:
            continue

        donor = result.donor
        for recipient in result.recipients:
            rows.append({
                "donor_name": donor.name,
                "donor_ein": donor.ein,
                "donor_city": donor.city or "",
                "donor_state": donor.state or "",
                "tax_year": donor.tax_year or "",
                "form_type": donor.form_type or "",
                "recipient_name": recipient.name,
                "recipient_ein": recipient.ein or "",
                "recipient_city": recipient.city or "",
                "recipient_state": recipient.state or "",
                "recipient_zip": recipient.zip_code or "",
                "amount": recipient.amount or "",
                "purpose": recipient.purpose or "",
                "relationship": recipient.relationship or "",
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def write_json(results: list[DonorResult], output_path: Path):
    """Write results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for result in results:
        data.append({
            "donor": result.donor.to_dict(),
            "recipients": [r.to_dict() for r in result.recipients],
            "total_recipients": result.total_recipients,
            "total_amount": result.total_amount,
            "error": result.error
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="donor990",
        description="Extract grant recipients from IRS 990 filings"
    )

    parser.add_argument(
        "eins",
        nargs="*",
        help="EIN(s) to process (e.g., 13-1684331)"
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to config file with list of EINs (JSON or text)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output/donor_grants.csv"),
        help="Output file path (default: output/donor_grants.csv)"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory to cache downloaded ZIP files"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (minimal output)"
    )

    args = parser.parse_args()

    # Setup logging
    if args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        setup_logging(args.verbose)

    # Collect EINs
    eins = list(args.eins) if args.eins else []

    if args.config:
        try:
            eins.extend(load_config(args.config))
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)

    if not eins:
        print("Error: No EINs provided. Use positional arguments or --config file.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Remove duplicates while preserving order
    seen = set()
    unique_eins = []
    for ein in eins:
        ein_clean = ein.replace("-", "")
        if ein_clean not in seen:
            seen.add(ein_clean)
            unique_eins.append(ein)
    eins = unique_eins

    if not args.quiet:
        print(f"\n{'='*70}")
        print("DONOR 990 PARSER")
        print(f"{'='*70}")
        print(f"Processing {len(eins)} EIN(s)")

    # Initialize clients
    api = ProPublicaAPI()
    irs_finder = IRSXMLFinder(cache_dir=args.cache_dir)
    form_parser = Form990Parser()

    # Process each EIN
    results = []
    for i, ein in enumerate(eins, 1):
        if not args.quiet:
            print(f"\n{'#'*70}")
            print(f"# PROCESSING {i}/{len(eins)}: {ein}")
            print(f"{'#'*70}")

        result = process_ein(ein, api, irs_finder, form_parser, verbose=args.verbose)
        results.append(result)

        if result.error:
            if not args.quiet:
                print(f"\n[!] Error: {result.error}")
        else:
            if not args.quiet:
                print(f"\n[OK] Found {result.total_recipients} recipients")

    # Write output
    output_path = args.output
    if args.format == "json":
        if output_path.suffix != ".json":
            output_path = output_path.with_suffix(".json")
        write_json(results, output_path)
    else:
        if output_path.suffix != ".csv":
            output_path = output_path.with_suffix(".csv")
        total_rows = write_csv(results, output_path)

    # Summary
    if not args.quiet:
        print(f"\n{'='*70}")
        print("COMPLETE")
        print(f"{'='*70}")
        print(f"Processed: {len(eins)} donor(s)")
        successful = sum(1 for r in results if not r.error)
        print(f"Successful: {successful}")
        total_recipients = sum(r.total_recipients for r in results)
        print(f"Total recipients: {total_recipients}")
        print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
