# donor990

Extract grant recipients from IRS Form 990 filings.

This tool takes one or more nonprofit EINs (Employer Identification Numbers) and retrieves their IRS Form 990 filings to extract all organizations they have given grants to.

## Features

- Look up nonprofits via ProPublica Nonprofit Explorer API
- Download 990 XML filings from IRS TEOS bulk archives
- Parse Form 990 and 990-PF to extract grant recipients
- Output to CSV or JSON format
- Support for batch processing multiple EINs
- Optional ZIP file caching to speed up repeated runs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/donor990.git
cd donor990

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Usage

### Command Line

```bash
# Process a single EIN
donor990 13-1684331

# Process multiple EINs
donor990 13-1684331 58-1716970 62-0812197

# Use a config file
donor990 --config eins.json

# Specify output file and format
donor990 13-1684331 --output results.csv
donor990 13-1684331 --output results.json --format json

# Enable ZIP caching (faster for multiple runs)
donor990 13-1684331 --cache-dir ./cache

# Verbose output
donor990 13-1684331 -v
```

### Config File

Create a JSON file with a list of EINs:

```json
[
    "13-1684331",
    "58-1716970",
    "62-0812197"
]
```

Or a text file with one EIN per line:

```text
# My donor list
13-1684331
58-1716970
62-0812197
```

### Python API

```python
from donor990 import ProPublicaAPI, IRSXMLFinder, Form990Parser

# Initialize clients
api = ProPublicaAPI()
irs = IRSXMLFinder()
parser = Form990Parser()

# Look up organization
org_data = api.get_organization("13-1684331")
print(org_data["organization"]["name"])  # "Ford Foundation"

# Find and download 990 XML
filing_info = irs.find_filing_info("13-1684331")
xml_content = irs.download_xml(filing_info)

# Parse the XML
root = parser.parse_xml(xml_content)
donor_info = parser.parse_donor_info(root)
recipients = parser.parse_grant_recipients(root)

print(f"{donor_info.name} made {len(recipients)} grants")
for r in recipients[:5]:
    print(f"  - {r.name}: ${r.amount:,}")
```

## Output Format

### CSV Columns

| Column | Description |
|--------|-------------|
| donor_name | Name of the grant-making organization |
| donor_ein | EIN of the donor |
| donor_city | City of the donor |
| donor_state | State of the donor |
| tax_year | Tax year of the filing |
| form_type | Type of 990 form (990, 990-PF, 990-EZ) |
| recipient_name | Name of the grant recipient |
| recipient_ein | EIN of the recipient (if available) |
| recipient_city | City of the recipient |
| recipient_state | State of the recipient |
| recipient_zip | ZIP code of the recipient |
| amount | Grant amount in dollars |
| purpose | Purpose of the grant |
| relationship | Relationship to donor (if any) |

## Data Sources

- **ProPublica Nonprofit Explorer API**: Organization lookup and metadata
- **IRS TEOS (Tax Exempt Organization Search)**: Form 990 XML bulk downloads

Note: The AWS S3 bucket that previously hosted IRS 990 XML files was emptied in late 2023. This tool downloads XML from the IRS bulk ZIP archives instead, which may require downloading large files (100MB-1GB).

## Supported Form Types

- **Form 990**: Standard form for most nonprofits (Schedule I grants)
- **Form 990-PF**: Private foundations (Part XV grants)
- **Form 990-EZ**: Simplified form (limited grant data)

## License

MIT License - see [LICENSE](LICENSE) for details.
