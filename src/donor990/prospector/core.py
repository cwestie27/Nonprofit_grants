"""Core prospector functionality - find similar nonprofits and their donors."""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from .models import NonprofitProfile, SimilarNonprofit
from .ntee_codes import (
    NTEE_CODES,
    get_ntee_description,
    get_major_category,
    get_major_description,
    codes_match,
)

load_dotenv()

# Suppress noisy debug logging from HTTP libraries and Google GenAI
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("charset_normalizer").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_LLM_MODEL = "gemini-2.0-flash"
DEFAULT_NUM_SIMILAR = 5
DEFAULT_MAX_CONCURRENT = 5


# =============================================================================
# PROPUBLICA API CLIENT (Async)
# =============================================================================

class AsyncProPublicaAPI:
    """Async client for ProPublica Nonprofit Explorer API."""

    BASE_URL = "https://projects.propublica.org/nonprofits/api/v2"

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def search_organizations(self, query: str, state: str = None,
                                   ntee_code: str = None, page: int = 0) -> list:
        """Search for nonprofits by name."""
        url = f"{self.BASE_URL}/search.json"
        params = {"q": query, "page": page}

        if state:
            params["state[id]"] = state
        if ntee_code:
            params["ntee[id]"] = ntee_code[0] if ntee_code else None

        try:
            async with self.session.get(url, params=params, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("organizations", [])
                return []
        except Exception as e:
            print(f"    [!] ProPublica search error: {e}")
            return []

    async def get_organization(self, ein) -> dict:
        """Get detailed organization info by EIN."""
        ein_clean = str(ein).replace("-", "")
        url = f"{self.BASE_URL}/organizations/{ein_clean}.json"

        try:
            async with self.session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("organization", {})
                return {}
        except Exception as e:
            print(f"    [!] ProPublica org lookup error: {e}")
            return {}

    async def get_filings(self, ein) -> list:
        """Get 990 filings for an organization."""
        ein_clean = str(ein).replace("-", "")
        url = f"{self.BASE_URL}/organizations/{ein_clean}.json"

        try:
            async with self.session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("filings_with_data", [])
                return []
        except Exception as e:
            print(f"    [!] ProPublica filings error: {e}")
            return []

    def parse_organization_to_profile(self, org_data: dict, filings: list = None) -> NonprofitProfile:
        """Convert ProPublica response to NonprofitProfile."""
        latest_filing = filings[0] if filings else {}

        ntee_code = org_data.get("ntee_code")
        # Use full subcode description (e.g., "Scholarships & Student Financial Aid")
        # Falls back to major category if subcode not found
        ntee_desc = get_ntee_description(ntee_code) if ntee_code else None

        return NonprofitProfile(
            name=org_data.get("name", ""),
            ein=org_data.get("ein"),
            city=org_data.get("city"),
            state=org_data.get("state"),
            ntee_code=ntee_code,
            ntee_description=ntee_desc,
            annual_revenue=latest_filing.get("totrevenue"),
            annual_expenses=latest_filing.get("totfuncexpns"),
            total_assets=latest_filing.get("totassetsend"),
            data_sources=["ProPublica 990"]
        )


# =============================================================================
# AI RESEARCH CLIENT
# =============================================================================

class AIResearcher:
    """AI-powered nonprofit research using Gemini."""

    def __init__(self, model: str = DEFAULT_LLM_MODEL, api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    async def research_nonprofit(self, name: str, website: str = None,
                                  existing_profile: NonprofitProfile = None) -> dict:
        """Research a nonprofit using AI."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._research_nonprofit_sync, name, website, existing_profile
        )

    def _research_nonprofit_sync(self, name: str, website: str = None,
                                  existing_profile: NonprofitProfile = None) -> dict:
        """Synchronous AI research."""
        from google.genai import types

        context = ""
        location_hint = ""
        if existing_profile:
            if existing_profile.city and existing_profile.state:
                location_hint = f" located in {existing_profile.city}, {existing_profile.state}"
            context = f"""
VERIFIED INFORMATION FROM IRS 990 DATA:
- Organization Name: {existing_profile.name}
- Location: {existing_profile.city}, {existing_profile.state}
- NTEE Category: {existing_profile.ntee_code} ({existing_profile.ntee_description})
- EIN: {existing_profile.ein}
{f'- Annual Revenue: ${existing_profile.annual_revenue:,}' if existing_profile.annual_revenue else ''}

IMPORTANT: Research the specific organization{location_hint}.
"""

        prompt = f"""Research this specific nonprofit organization: "{name}"{location_hint}
{f'Website: {website}' if website else ''}
{context}

Find and return the following in JSON format:
{{
    "mission_statement": "Their EXACT official mission statement if you can find it, otherwise null",
    "programs": ["List", "of", "specific", "programs"],
    "population_served": "Specifically who they serve",
    "impact_statement": "Key impact metrics or outcomes",
    "website": "Their official website URL",
    "founding_year": null or year as integer
}}

Return ONLY valid JSON."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            result = json.loads(response.text)
            if isinstance(result, list):
                result = result[0] if result else {}
            return result
        except Exception as e:
            print(f"    [!] AI research error: {e}")
            return {}

    async def find_similar_nonprofits(self, profile: NonprofitProfile, count: int = 5) -> list:
        """Use AI to find similar nonprofits."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._find_similar_sync, profile, count)

    def _find_similar_sync(self, profile: NonprofitProfile, count: int = 5) -> list:
        """Synchronous similar nonprofit search."""
        from google.genai import types

        programs_str = ', '.join(profile.programs) if profile.programs else 'Not specified'

        prompt = f"""Find {count + 3} nonprofits that are HIGHLY SIMILAR to this organization.

TARGET NONPROFIT:
Name: {profile.name}
Location: {profile.city}, {profile.state}
Revenue: {profile.revenue_bracket()}
NTEE Code: {profile.ntee_code} - {profile.ntee_description or 'Unknown'}
Mission: {profile.mission_statement or 'Not available'}
Population Served: {profile.population_served or 'Not specified'}
Programs: {programs_str}

MATCHING CRITERIA (in order of importance):
1. NTEE CODE - STRONGLY prefer same NTEE subcode (e.g., {profile.ntee_code}), then same group (e.g., {profile.ntee_code[:2] if profile.ntee_code and len(profile.ntee_code) >= 2 else 'B8'}x)
2. POPULATION SERVED - same specific population
3. MISSION ALIGNMENT - similar core mission
4. PROGRAM TYPE - similar programs
5. GEOGRAPHIC REGION - prefer same state
6. SIZE/REVENUE - similar budget range

REQUIREMENTS:
- Only REAL, currently operating 501(c)(3) nonprofits
- Must have an active website
- Do NOT include {profile.name} itself
- MUST include NTEE code if known (e.g., B82, O31, P30)

Return JSON array:
[
    {{
        "name": "Official nonprofit name",
        "city": "City",
        "state": "State abbreviation",
        "website": "https://...",
        "ntee_code": "NTEE code like B82, O31, P30 (required if known)",
        "cause_area": "Their specific cause area",
        "population_served": "Who they serve",
        "estimated_revenue": "small/medium/large",
        "programs": ["program 1", "program 2"],
        "similarity_reasons": ["reason 1", "reason 2"]
    }}
]

Return ONLY valid JSON array."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            result = json.loads(response.text)
            if isinstance(result, dict):
                result = list(result.values())[0] if result.values() else []
            return result[:count + 3]
        except Exception as e:
            print(f"    [!] AI similar search error: {e}")
            return []


# =============================================================================
# WEBSITE SCRAPER
# =============================================================================

class AsyncScraper:
    """Async website scraper for nonprofit content."""

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        self._needs_wayback = set()

    async def scrape_page(self, url: str, use_wayback: bool = False) -> dict:
        """Scrape a webpage and extract content."""
        original_url = url
        actual_url = url

        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        if base in self._needs_wayback:
            use_wayback = True

        if use_wayback and not url.startswith('https://web.archive.org'):
            actual_url = f"https://web.archive.org/web/2024/{url}"

        try:
            async with self.session.get(actual_url, headers=self.headers, timeout=20) as resp:
                content = await resp.read()
                content_len = len(content)

                is_blocked = (
                    resp.status in [403, 202, 503] or
                    content_len < 1000 or
                    b'cloudflare' in content.lower() or
                    b'challenge-platform' in content.lower()
                )

                if is_blocked and not use_wayback:
                    self._needs_wayback.add(base)
                    return await self.scrape_page(original_url, use_wayback=True)

                if resp.status != 200:
                    if not use_wayback:
                        return await self.scrape_page(original_url, use_wayback=True)
                    return None

                soup = BeautifulSoup(content, 'lxml')

                for element in soup(["script", "style", "noscript", "iframe"]):
                    element.decompose()

                text = soup.get_text(separator=' ', strip=True)[:30000]
                title = soup.title.string if soup.title else ""

                if len(text) < 500 and not use_wayback:
                    self._needs_wayback.add(base)
                    return await self.scrape_page(original_url, use_wayback=True)

                # Extract image clues for logos
                image_clues = []
                for img in soup.find_all('img'):
                    src = img.get('src', '') or img.get('data-src', '')
                    alt = img.get('alt', '')
                    img_class = ' '.join(img.get('class', []))

                    keywords = ['logo', 'sponsor', 'partner', 'supporter', 'donor', 'funder']
                    context_str = (alt + img_class + src).lower()
                    if any(k in context_str for k in keywords) or (alt and len(alt) > 2):
                        filename = src.split('/')[-1].split('?')[0] if src else ''
                        image_clues.append(f"[Logo: alt='{alt}' file='{filename}']")

                # Extract links
                links = []
                for a in soup.find_all('a', href=True):
                    link_text = a.get_text(strip=True)
                    href = a['href']
                    if link_text and len(link_text) > 2 and href.startswith('http'):
                        skip = ['facebook', 'twitter', 'instagram', 'linkedin', 'youtube']
                        if not any(s in href.lower() for s in skip):
                            domain = urlparse(href).netloc.replace('www.', '')
                            links.append(f"[Link: text='{link_text}' domain='{domain}']")

                return {
                    "title": title,
                    "text": text,
                    "images": "\n".join(image_clues[:100]),
                    "links": "\n".join(links[:50]),
                    "url": original_url,
                    "used_wayback": use_wayback
                }

        except asyncio.TimeoutError:
            if not use_wayback:
                self._needs_wayback.add(base)
                return await self.scrape_page(original_url, use_wayback=True)
            return None
        except Exception:
            if not use_wayback:
                return await self.scrape_page(original_url, use_wayback=True)
            return None

    async def find_accessible_pages(self, base_url: str) -> list:
        """Find accessible pages on a nonprofit's website."""
        common_paths = [
            '/', '/partners', '/sponsors', '/our-partners', '/our-sponsors',
            '/supporters', '/donors', '/funders', '/about', '/about-us',
        ]

        use_wayback = base_url in self._needs_wayback

        async def check_page(path):
            try:
                test_url = urljoin(base_url, path)
                check_url = test_url
                if use_wayback:
                    check_url = f"https://web.archive.org/web/2024/{test_url}"

                async with self.session.get(check_url, headers=self.headers,
                                            timeout=8, allow_redirects=True) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        if len(content) > 1000 and b'cloudflare' not in content.lower():
                            return test_url
            except:
                pass
            return None

        # Test if base URL is accessible
        try:
            async with self.session.get(base_url, headers=self.headers, timeout=8) as resp:
                content = await resp.read()
                is_blocked = (
                    resp.status in [403, 202, 503] or
                    len(content) < 1000 or
                    b'cloudflare' in content.lower()
                )
                if is_blocked:
                    self._needs_wayback.add(base_url)
                    use_wayback = True
        except:
            self._needs_wayback.add(base_url)
            use_wayback = True

        tasks = [check_page(path) for path in common_paths]
        results = await asyncio.gather(*tasks)
        found = [r for r in results if r]

        if not found:
            return [urljoin(base_url, p) for p in ['/', '/partners', '/about']]

        return found


# =============================================================================
# DONOR EXTRACTOR
# =============================================================================

class DonorExtractor:
    """Extract donors from scraped content using AI."""

    def __init__(self, model: str = DEFAULT_LLM_MODEL, api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    async def extract_donors(self, scraped_data: dict, org_name: str) -> list:
        """Extract donor names from scraped content."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_donors_sync, scraped_data, org_name)

    def _extract_donors_sync(self, scraped_data: dict, org_name: str) -> list:
        """Synchronous donor extraction."""
        from google.genai import types

        prompt = f"""Extract ALL corporate sponsors, partners, donors, and funders from this webpage for "{org_name}".

Page: {scraped_data['url']}
Title: {scraped_data['title']}

TEXT CONTENT:
{scraped_data['text'][:20000]}

LOGO/IMAGE METADATA:
{scraped_data['images']}

OUTBOUND LINKS:
{scraped_data['links']}

EXTRACTION INSTRUCTIONS:
1. Look for sections labeled "Partners", "Sponsors", "Donors", "Supporters"
2. Extract company names from text, logo alt text, logo filenames, link text
3. Include: corporations, foundations, banks, local businesses
4. EXCLUDE: {org_name} itself, social media platforms, generic terms

Return a JSON array of unique company/organization names:
["Company A", "Foundation B", "Bank C"]

If no sponsors/partners found, return: []
Return ONLY valid JSON array."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            result = json.loads(response.text)
            if isinstance(result, dict):
                result = list(result.values())[0] if result.values() else []

            invalid = ['affiliates', 'partners', 'sponsors', 'donors', 'donate', 'contact']
            cleaned = []
            for d in result:
                if isinstance(d, str) and len(d) >= 4 and d.lower() not in invalid:
                    cleaned.append(d.strip())
            return cleaned
        except:
            return []


# =============================================================================
# SIMILARITY SCORER
# =============================================================================

class SimilarityScorer:
    """Score similarity between nonprofits."""

    # Weights rebalanced to include NTEE code matching
    WEIGHTS = {
        "ntee": 0.20,       # NTEE code match (new)
        "population": 0.30,  # reduced from 0.35
        "mission": 0.20,     # reduced from 0.25
        "programs": 0.15,    # reduced from 0.20
        "geography": 0.10,   # reduced from 0.12
        "revenue": 0.05      # reduced from 0.08
    }

    CAUSE_KEYWORDS = {
        "fatherhood": ["father", "dad", "paternal", "fatherless"],
        "mentoring": ["mentor", "mentoring", "role model"],
        "youth": ["youth", "young", "teen", "child", "kid"],
        "mental_health": ["mental health", "counseling", "therapy"],
        "family": ["family", "parent", "parenting"],
    }

    GENERIC_PATTERNS = [
        "family service", "children's service", "community center",
        "united way", "ymca", "ywca", "human services"
    ]

    @classmethod
    def score(cls, input_profile: NonprofitProfile, candidate: dict) -> float:
        """Calculate similarity score (0-100)."""
        score = 0.0
        candidate_name = candidate.get("name", "").lower()

        # Check for generic organizations - cap their score
        is_generic = any(p in candidate_name for p in cls.GENERIC_PATTERNS)
        if is_generic:
            return 40.0

        # NTEE code match (20%) - NEW
        # Full match: 100%, Group match (same tens digit): 70%, Major match: 40%
        if input_profile.ntee_code and candidate.get("ntee_code"):
            input_ntee = input_profile.ntee_code
            cand_ntee = candidate.get("ntee_code", "")

            if codes_match(input_ntee, cand_ntee, level="full"):
                score += 100 * cls.WEIGHTS["ntee"]
            elif codes_match(input_ntee, cand_ntee, level="group"):
                score += 70 * cls.WEIGHTS["ntee"]
            elif codes_match(input_ntee, cand_ntee, level="major"):
                score += 40 * cls.WEIGHTS["ntee"]

        # Population served match (30%)
        if input_profile.population_served and candidate.get("population_served"):
            pop_score = cls._keyword_overlap(
                input_profile.population_served,
                candidate.get("population_served", "")
            )
            score += pop_score * 100 * cls.WEIGHTS["population"]

        # Mission alignment (20%)
        if input_profile.mission_statement:
            input_mission = f"{input_profile.mission_statement or ''} {input_profile.cause_area or ''}"
            cand_mission = f"{candidate.get('cause_area', '')} {' '.join(candidate.get('similarity_reasons', []))}"
            mission_score = cls._keyword_overlap(input_mission, cand_mission)
            score += mission_score * 100 * cls.WEIGHTS["mission"]

        # Programs match (15%)
        if input_profile.programs and candidate.get("programs"):
            input_progs = " ".join(input_profile.programs).lower()
            cand_progs = " ".join(candidate.get("programs", [])).lower()
            prog_score = cls._keyword_overlap(input_progs, cand_progs)
            score += prog_score * 100 * cls.WEIGHTS["programs"]

        # Geographic proximity (10%)
        if input_profile.state and candidate.get("state"):
            if input_profile.state == candidate.get("state"):
                score += 100 * cls.WEIGHTS["geography"]
            else:
                score += 20 * cls.WEIGHTS["geography"]

        # Revenue similarity (5%)
        if candidate.get("estimated_revenue"):
            score += 50 * cls.WEIGHTS["revenue"]

        return min(100, round(score, 1))

    @staticmethod
    def _keyword_overlap(text1: str, text2: str) -> float:
        """Calculate keyword overlap between two text strings."""
        if not text1 or not text2:
            return 0.0

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        if text1_lower == text2_lower:
            return 1.0

        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())

        stop_words = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'for', 'in', 'on', 'with'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words

        if not words1 or not words2:
            return 0.0

        overlap = len(words1 & words2)
        total = len(words1 | words2)

        return overlap / total if total > 0 else 0.0


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class DonorProspector:
    """Main orchestrator for the donor prospecting workflow."""

    def __init__(self, model: str = DEFAULT_LLM_MODEL,
                 num_similar: int = DEFAULT_NUM_SIMILAR,
                 max_concurrent: int = DEFAULT_MAX_CONCURRENT):
        self.model = model
        self.num_similar = num_similar
        self.max_concurrent = max_concurrent
        self.propublica = None
        self.ai_researcher = None
        self.scraper = None
        self.donor_extractor = None
        self.scorer = SimilarityScorer()

    async def run(self, nonprofit_name: str, state: str = None, city: str = None,
                  ein: str = None, website: str = None, output_dir: str = "."):
        """Execute the full donor prospecting workflow."""
        print("\n" + "=" * 70)
        print("DONOR PROSPECTOR")
        print(f"Model: {self.model}")
        print("=" * 70)
        print(f"\nAnalyzing: {nonprofit_name}")
        if state or city:
            print(f"Location filter: {', '.join(filter(None, [city, state]))}")

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            self.propublica = AsyncProPublicaAPI(session)
            self.ai_researcher = AIResearcher(model=self.model)
            self.scraper = AsyncScraper(session)
            self.donor_extractor = DonorExtractor(model=self.model)

            # Step 1: Research Input Nonprofit
            print(f"\n[Step 1] Researching your nonprofit...")
            input_profile = await self._research_nonprofit(
                nonprofit_name, state=state, city=city, ein=ein, website=website
            )

            if not input_profile:
                print("[!] Could not research nonprofit. Exiting.")
                return None

            self._print_profile(input_profile)

            # Step 2: Find Similar Nonprofits
            print(f"\n[Step 2] Finding similar nonprofits...")
            similar_nonprofits = await self._find_similar_nonprofits(input_profile)

            if not similar_nonprofits:
                print("[!] Could not find similar nonprofits.")
                return None

            print(f"\n    Results (scored and ranked):")
            for i, sn in enumerate(similar_nonprofits, 1):
                print(f"    {i}. {sn.profile.name} ({sn.profile.city}, {sn.profile.state}) - Score: {sn.similarity_score}%")

            # Step 3: Find Donors
            print(f"\n[Step 3] Searching for donors ({self.max_concurrent} concurrent)...")
            all_results = await self._find_donors_parallel(input_profile, similar_nonprofits)

            # Step 4: Generate Report
            print(f"\n[Step 4] Generating report...")
            elapsed = time.time() - start_time
            report = self._generate_report(input_profile, similar_nonprofits, all_results, elapsed)

            print("\n")
            print(report["text"])

            # Save files
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            clean_name = nonprofit_name.replace(' ', '_').replace("'", "")[:30]

            report_file = os.path.join(output_dir, f"donor_report_{clean_name}_{timestamp}.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report["text"])
            print(f"\nReport saved to: {report_file}")

            json_file = os.path.join(output_dir, f"donor_data_{clean_name}_{timestamp}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report["data"], f, indent=2, default=str)
            print(f"JSON data saved to: {json_file}")

            return report

    async def _research_nonprofit(self, name: str, state: str = None, city: str = None,
                                   ein: str = None, website: str = None) -> NonprofitProfile:
        """Research nonprofit using ProPublica then AI."""
        profile = None

        # Step 1a: ProPublica Lookup
        print("    [1a] ProPublica API lookup...")

        if ein:
            org_detail = await self.propublica.get_organization(ein)
            if org_detail:
                filings = await self.propublica.get_filings(ein)
                profile = self.propublica.parse_organization_to_profile(org_detail, filings)
                print(f"        [OK] Found: {profile.name}")
        else:
            results = await self.propublica.search_organizations(name, state=state)
            if results:
                if city:
                    filtered = [o for o in results if o.get("city", "").lower() == city.lower()]
                    if filtered:
                        results = filtered

                org = results[0]
                org_ein = org.get("ein")
                print(f"        [OK] Selected: {org.get('name')} (EIN: {org_ein})")

                if org_ein:
                    org_detail = await self.propublica.get_organization(org_ein)
                    filings = await self.propublica.get_filings(org_ein)
                    profile = self.propublica.parse_organization_to_profile(org_detail or org, filings)

        if not profile:
            profile = NonprofitProfile(name=name, city=city, state=state)
            profile.data_sources = []

        if website:
            profile.website = website

        # Step 1b: AI Research
        print("    [1b] AI research...")
        ai_results = await self.ai_researcher.research_nonprofit(
            name=profile.name, website=profile.website, existing_profile=profile
        )

        if ai_results:
            print(f"        [OK] Found mission, programs, population")
            profile.mission_statement = ai_results.get("mission_statement")
            profile.programs = ai_results.get("programs")
            profile.population_served = ai_results.get("population_served")
            profile.impact_statement = ai_results.get("impact_statement")
            if profile.ntee_description:
                profile.cause_area = profile.ntee_description
            if not profile.website and ai_results.get("website"):
                profile.website = ai_results.get("website")
            if profile.data_sources:
                profile.data_sources.append("AI Research")
            else:
                profile.data_sources = ["AI Research"]

        return profile

    async def _find_similar_nonprofits(self, input_profile: NonprofitProfile) -> list:
        """Find similar nonprofits using ProPublica + AI in parallel."""
        tasks = []

        if input_profile.ntee_code and input_profile.state:
            print(f"    [2a] ProPublica: NTEE={input_profile.ntee_code} in {input_profile.state}")
            tasks.append(self._propublica_similar_search(input_profile))

        print(f"    [2b] AI: finding similar nonprofits")
        tasks.append(self.ai_researcher.find_similar_nonprofits(input_profile, count=self.num_similar))

        results = await asyncio.gather(*tasks)

        # Combine and deduplicate
        all_candidates = []
        seen_names = {input_profile.name.lower()}

        for result_set in results:
            if isinstance(result_set, list):
                for item in result_set:
                    name = item.get("name", "") if isinstance(item, dict) else ""
                    if name and name.lower() not in seen_names:
                        seen_names.add(name.lower())
                        all_candidates.append(item)

        # Score and rank
        scored = []
        for candidate in all_candidates:
            score = self.scorer.score(input_profile, candidate)
            profile = NonprofitProfile(
                name=candidate.get("name"),
                city=candidate.get("city"),
                state=candidate.get("state"),
                website=candidate.get("website"),
                cause_area=candidate.get("cause_area"),
                population_served=candidate.get("population_served"),
            )
            scored.append(SimilarNonprofit(
                profile=profile,
                similarity_score=score,
                similarity_reasons=candidate.get("similarity_reasons", [])
            ))

        scored.sort(key=lambda x: x.similarity_score, reverse=True)
        return scored[:self.num_similar]

    async def _propublica_similar_search(self, input_profile: NonprofitProfile) -> list:
        """Search ProPublica for similar nonprofits."""
        try:
            results = await self.propublica.search_organizations(
                query="", state=input_profile.state, ntee_code=input_profile.ntee_code
            )

            candidates = []
            for org in results[:10]:
                if org.get("name", "").lower() != input_profile.name.lower():
                    candidates.append({
                        "name": org.get("name"),
                        "city": org.get("city"),
                        "state": org.get("state"),
                        "cause_area": input_profile.ntee_description,
                        "similarity_reasons": [f"Same NTEE ({input_profile.ntee_code})"]
                    })
            return candidates
        except Exception as e:
            print(f"    [!] ProPublica similar search error: {e}")
            return []

    async def _find_donors_parallel(self, input_profile: NonprofitProfile,
                                     similar_nonprofits: list) -> list:
        """Find donors for all nonprofits in parallel."""
        nonprofits_to_scan = [{"profile": input_profile, "is_input": True}]
        for sn in similar_nonprofits:
            nonprofits_to_scan.append({"profile": sn.profile, "is_input": False})

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def search_with_semaphore(np_info):
            async with semaphore:
                return await self._find_donors_for_nonprofit(np_info["profile"], np_info["is_input"])

        tasks = [search_with_semaphore(np) for np in nonprofits_to_scan]
        return await asyncio.gather(*tasks)

    async def _find_donors_for_nonprofit(self, profile: NonprofitProfile, is_input: bool) -> dict:
        """Find donors for a single nonprofit."""
        label = "(Your nonprofit)" if is_input else ""
        print(f"    Searching: {profile.name} {label}")

        donors = {}
        website = profile.website
        if not website:
            clean_name = profile.name.lower().replace(' ', '').replace("'", "")
            website = f"https://{clean_name}.org"

        if not website.startswith('http'):
            website = f"https://{website}"

        base_url = f"{urlparse(website).scheme}://{urlparse(website).netloc}"
        pages = await self.scraper.find_accessible_pages(base_url)

        if not pages:
            pages = [base_url]

        for url in pages[:5]:
            data = await self.scraper.scrape_page(url)
            if data and len(data.get('text', '')) > 500:
                extracted = await self.donor_extractor.extract_donors(data, profile.name)
                for donor in extracted:
                    if donor not in donors:
                        donors[donor] = {"source": url}

        print(f"        Found {len(donors)} donors")

        return {
            "nonprofit": profile.name,
            "website": website,
            "is_input": is_input,
            "donors": donors
        }

    def _print_profile(self, profile: NonprofitProfile):
        """Print nonprofit profile summary."""
        print(f"\n    Profile:")
        print(f"      Name: {profile.name}")
        print(f"      Location: {profile.city}, {profile.state}")
        if profile.ein:
            print(f"      EIN: {profile.ein}")
        if profile.ntee_code:
            print(f"      NTEE: {profile.ntee_code} ({profile.ntee_description})")
        if profile.annual_revenue:
            print(f"      Revenue: ${profile.annual_revenue:,}")
        if profile.mission_statement:
            mission = profile.mission_statement[:100] + "..." if len(profile.mission_statement) > 100 else profile.mission_statement
            print(f"      Mission: {mission}")
        if profile.population_served:
            print(f"      Serves: {profile.population_served}")

    def _generate_report(self, input_profile: NonprofitProfile,
                         similar_nonprofits: list, donor_results: list,
                         elapsed_time: float) -> dict:
        """Generate comprehensive report."""
        lines = []
        lines.append("=" * 70)
        lines.append("DONOR PROSPECTOR REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"Processing Time: {elapsed_time:.1f} seconds")
        lines.append("=" * 70)

        # Input nonprofit
        lines.append(f"\nYOUR NONPROFIT: {input_profile.name}")
        lines.append(f"  Location: {input_profile.city}, {input_profile.state}")
        if input_profile.mission_statement:
            lines.append(f"  Mission: {input_profile.mission_statement}")

        # Similar nonprofits
        lines.append("\n" + "-" * 70)
        lines.append("SIMILAR NONPROFITS")
        lines.append("-" * 70)

        for i, sn in enumerate(similar_nonprofits, 1):
            lines.append(f"\n{i}. {sn.profile.name} (Score: {sn.similarity_score}%)")
            lines.append(f"   Location: {sn.profile.city}, {sn.profile.state}")
            if sn.similarity_reasons:
                lines.append(f"   Why Similar: {'; '.join(sn.similarity_reasons[:2])}")

        # Donors by nonprofit
        lines.append("\n" + "-" * 70)
        lines.append("DONORS BY NONPROFIT")
        lines.append("-" * 70)

        all_donors = {}
        input_donors = []

        for result in donor_results:
            np_name = result["nonprofit"]
            donors = result["donors"]
            is_input = result.get("is_input", False)

            lines.append(f"\n{np_name} {'(YOUR NONPROFIT)' if is_input else ''}")
            lines.append(f"  Donors Found: {len(donors)}")

            if donors:
                for donor_name in sorted(donors.keys())[:15]:
                    lines.append(f"    - {donor_name}")
                if len(donors) > 15:
                    lines.append(f"    ... and {len(donors) - 15} more")

                for donor_name in donors:
                    if is_input:
                        input_donors.append(donor_name)
                    if donor_name not in all_donors:
                        all_donors[donor_name] = []
                    all_donors[donor_name].append(np_name)

        # Consolidated prospects
        lines.append("\n" + "=" * 70)
        lines.append("PRIORITY OUTREACH LIST")
        lines.append("=" * 70)

        new_prospects = {k: v for k, v in all_donors.items() if k not in input_donors}

        lines.append(f"\nTotal Unique Donors: {len(all_donors)}")
        lines.append(f"New Prospects: {len(new_prospects)}")

        if new_prospects:
            sorted_prospects = sorted(new_prospects.items(), key=lambda x: len(x[1]), reverse=True)
            lines.append(f"\n{'Donor':<40} {'# Orgs':<8}")
            lines.append("-" * 50)
            for donor, orgs in sorted_prospects[:30]:
                lines.append(f"{donor[:40]:<40} {len(orgs):<8}")

        lines.append("\n" + "=" * 70)

        # Build JSON data
        json_data = {
            "input_nonprofit": input_profile.to_dict(),
            "similar_nonprofits": [
                {"profile": sn.profile.to_dict(), "similarity_score": sn.similarity_score}
                for sn in similar_nonprofits
            ],
            "donor_results": donor_results,
            "new_prospects": new_prospects,
            "existing_donors": input_donors,
        }

        return {"text": "\n".join(lines), "data": json_data}
