# donor990

Nonprofit grant analysis tools - extract grant recipients and find donor prospects.

## Overview

This package provides two utilities for nonprofit fundraising research:

1. **lookup** - Takes donor EIN(s) and extracts all organizations they've given grants to from IRS 990 filings
2. **prospect** - Takes a nonprofit name and finds similar organizations along with foundations that have funded them

## Installation

```bash
# Clone the repository
git clone https://github.com/cwestie27/Nonprofit_grants.git
cd Nonprofit_grants

# Install base package (lookup command only)
pip install -e .

# Install with prospect command dependencies
pip install -e ".[prospect]"

# Install with all dependencies (prospect + dev tools)
pip install -e ".[all]"
```

## Commands

### lookup - Extract Grant Recipients

Takes one or more donor EINs and retrieves their IRS Form 990 filings to extract all grant recipients.

```bash
# Process a single EIN
donor990 lookup 13-1684331

# Process multiple EINs
donor990 lookup 13-1684331 58-1716970 62-0812197

# Use a config file with list of EINs
donor990 lookup --config eins.json

# Specify output file and format
donor990 lookup 13-1684331 --output results.csv
donor990 lookup 13-1684331 --output results.json --format json

# Enable ZIP caching (faster for repeated runs)
donor990 lookup 13-1684331 --cache-dir ./cache

# Verbose output
donor990 lookup 13-1684331 -v
```

#### Config File Formats

JSON array:
```json
[
    "13-1684331",
    "58-1716970",
    "62-0812197"
]
```

Or text file with one EIN per line:
```text
# My donor list
13-1684331
58-1716970
62-0812197
```

#### Output CSV Columns

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

---

### prospect - Find Similar Nonprofits & Their Donors

Takes a nonprofit name and uses AI to find similar organizations, then identifies foundations that have funded them.

**Requirements:**
- Set `GEMINI_API_KEY` environment variable (get one at https://makersuite.google.com/app/apikey)
- Install prospect dependencies: `pip install -e ".[prospect]"`

```bash
# Basic usage - find donors for similar nonprofits
donor990 prospect "Nashville Food Project"

# Narrow search with location
donor990 prospect "Nashville Food Project" --state TN --city Nashville

# Provide EIN for exact match
donor990 prospect "Nashville Food Project" --ein 62-0812197

# Provide website for better AI research
donor990 prospect "Nashville Food Project" --website https://thenashvillefoodproject.org

# Customize number of similar nonprofits to find
donor990 prospect "Nashville Food Project" --num-similar 10

# Specify output directory
donor990 prospect "Nashville Food Project" --output-dir ./results

# Use a different AI model
donor990 prospect "Nashville Food Project" --model gemini-2.0-flash

# Verbose output
donor990 prospect "Nashville Food Project" -v
```

#### How It Works

1. **Research Input Nonprofit** - Fetches data from ProPublica API and uses AI to research the nonprofit's mission, programs, and impact
2. **Find Similar Nonprofits** - Uses AI to identify organizations with similar missions, then validates them against ProPublica
3. **Extract Donors** - For each nonprofit (input + similar), searches web for donors and extracts from 990 Schedule I filings
4. **Score Similarity** - Ranks similar nonprofits by how well they match the input nonprofit
5. **Generate Report** - Outputs a comprehensive JSON report with all findings

#### Output

Creates files in the output directory:
- `{nonprofit_name}_prospect_report.json` - Full report with all data
- Console output shows summary of findings

---

## Python API

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

## Data Sources

- **ProPublica Nonprofit Explorer API** - Organization lookup and metadata
- **IRS TEOS (Tax Exempt Organization Search)** - Form 990 XML bulk downloads
- **Google Gemini AI** - Research and similarity analysis (prospect command only)

Note: The AWS S3 bucket that previously hosted IRS 990 XML files was emptied in late 2023. This tool downloads XML from the IRS bulk ZIP archives instead, which may require downloading large files (100MB-1GB).

## Supported Form Types

- **Form 990** - Standard form for most nonprofits (Schedule I grants)
- **Form 990-PF** - Private foundations (Part XV grants)
- **Form 990-EZ** - Simplified form (limited grant data)

## License

MIT License - see [LICENSE](LICENSE) for details.
