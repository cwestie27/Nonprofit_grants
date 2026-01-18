"""Command-line interface for donor990."""

import argparse
import asyncio
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


def setup_logging(verbose: bool = False, quiet: bool = False):
    """Configure logging."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler()]
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def load_config(config_path: Path) -> list[str]:
    """Load EINs from a config file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    content = config_path.read_text().strip()

    if config_path.suffix == ".json" or content.startswith("[") or content.startswith("{"):
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "eins" in data:
            return data["eins"]
        else:
            raise ValueError("JSON config must be a list or have an 'eins' key")

    return [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]


# =============================================================================
# LOOKUP COMMAND - Extract grant recipients from donor 990 filings
# =============================================================================

def process_ein(
    ein: str,
    api: ProPublicaAPI,
    irs_finder: IRSXMLFinder,
    parser: Form990Parser,
    verbose: bool = False
) -> DonorResult:
    """Process a single EIN and extract grant recipients."""
    ein_clean = ein.replace("-", "")

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

    if verbose:
        print(f"\n[Step 2] Finding XML 990 filing...")

    xml_content = None

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

    if verbose:
        print(f"\n[Step 4] Parsing 990 XML...")

    root = parser.parse_xml(xml_content)
    if root is None:
        return DonorResult(
            donor=DonorInfo(name=org_info.get("name", "Unknown"), ein=ein_clean),
            recipients=[],
            error="Could not parse XML"
        )

    donor_info = parser.parse_donor_info(root)
    if verbose:
        print(f"    Form Type: {donor_info.form_type}")
        print(f"    Tax Year: {donor_info.tax_year}")
        if donor_info.total_grants:
            print(f"    Total Grants: ${donor_info.total_grants:,}")

    if verbose:
        print(f"\n[Step 5] Extracting grant recipients...")

    recipients = parser.parse_grant_recipients(root)
    if verbose:
        print(f"    Found {len(recipients)} grant recipients")

    return DonorResult(donor=donor_info, recipients=recipients)


def write_csv(results: list[DonorResult], output_path: Path) -> int:
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


def cmd_lookup(args):
    """Execute the lookup command."""
    setup_logging(args.verbose, args.quiet)

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
        sys.exit(1)

    # Remove duplicates
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
        print("DONOR 990 LOOKUP")
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
        write_csv(results, output_path)

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


# =============================================================================
# PROSPECT COMMAND - Find similar nonprofits and their donors
# =============================================================================

def cmd_prospect(args):
    """Execute the prospect command."""
    setup_logging(args.verbose, args.quiet)

    try:
        from .prospector import DonorProspector
    except ImportError as e:
        print(f"Error: Missing dependencies for prospect command.", file=sys.stderr)
        print(f"Install with: pip install donor990[prospect]", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    nonprofit_name = args.name
    if not nonprofit_name:
        print("Error: Please provide a nonprofit name.", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prospector = DonorProspector(
        model=args.model,
        num_similar=args.num_similar,
        max_concurrent=args.max_concurrent
    )

    asyncio.run(prospector.run(
        nonprofit_name,
        state=args.state,
        city=args.city,
        ein=args.ein,
        website=args.website,
        output_dir=str(output_dir)
    ))


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="donor990",
        description="Nonprofit grant and donor analysis tools"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -------------------------------------------------------------------------
    # LOOKUP subcommand
    # -------------------------------------------------------------------------
    lookup_parser = subparsers.add_parser(
        "lookup",
        help="Extract grant recipients from donor 990 filings",
        description="Takes donor EIN(s) and extracts all organizations they gave grants to."
    )
    lookup_parser.add_argument(
        "eins", nargs="*",
        help="EIN(s) to process (e.g., 13-1684331)"
    )
    lookup_parser.add_argument(
        "-c", "--config", type=Path,
        help="Path to config file with list of EINs"
    )
    lookup_parser.add_argument(
        "-o", "--output", type=Path, default=Path("output/donor_grants.csv"),
        help="Output file path (default: output/donor_grants.csv)"
    )
    lookup_parser.add_argument(
        "--format", choices=["csv", "json"], default="csv",
        help="Output format (default: csv)"
    )
    lookup_parser.add_argument(
        "--cache-dir", type=Path,
        help="Directory to cache downloaded ZIP files"
    )
    lookup_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    lookup_parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Quiet mode"
    )
    lookup_parser.set_defaults(func=cmd_lookup)

    # -------------------------------------------------------------------------
    # PROSPECT subcommand
    # -------------------------------------------------------------------------
    prospect_parser = subparsers.add_parser(
        "prospect",
        help="Find similar nonprofits and their donors",
        description="Takes a nonprofit name and finds similar organizations and their donors."
    )
    prospect_parser.add_argument(
        "name", nargs="?",
        help="Nonprofit name to analyze"
    )
    prospect_parser.add_argument(
        "-s", "--state",
        help="State abbreviation (e.g., TN, CA)"
    )
    prospect_parser.add_argument(
        "-c", "--city",
        help="City name"
    )
    prospect_parser.add_argument(
        "-e", "--ein",
        help="EIN for exact match"
    )
    prospect_parser.add_argument(
        "-w", "--website",
        help="Official website URL"
    )
    prospect_parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("output"),
        help="Output directory (default: output/)"
    )
    prospect_parser.add_argument(
        "--model", default="gemini-2.0-flash",
        help="LLM model to use (default: gemini-2.0-flash)"
    )
    prospect_parser.add_argument(
        "--num-similar", type=int, default=5,
        help="Number of similar nonprofits to find (default: 5)"
    )
    prospect_parser.add_argument(
        "--max-concurrent", type=int, default=5,
        help="Max concurrent requests (default: 5)"
    )
    prospect_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    prospect_parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Quiet mode"
    )
    prospect_parser.set_defaults(func=cmd_prospect)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
