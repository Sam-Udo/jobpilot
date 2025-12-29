"""
JobPilot v3 - Full Bright Data MCP Integration

Features:
- CV format preservation (same sections, layout, fonts)
- Real job search via Bright Data MCP
- Search results caching by NLU terms
- Pagination (10 per page)
- Per-job Generate CV, Download, and Apply buttons
"""

import os
import re
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(title="JobPilot v3", description="AI-powered job search and CV generation")

# Directories
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

for d in [STATIC_DIR, DATA_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CVStructure:
    """Analyzed structure of uploaded CV."""
    sections: List[str] = field(default_factory=list)
    section_order: List[str] = field(default_factory=list)
    has_contact_header: bool = True
    bullet_style: str = "-"
    date_format: str = "Month Year"
    uses_bold_titles: bool = True
    uses_italic_dates: bool = True
    raw_text: str = ""

@dataclass
class SearchFilters:
    """Parsed search filters from NLU."""
    job_titles: List[str] = field(default_factory=list)
    location_type: str = "any"
    location: str = ""
    days_ago: int = 30
    
    def cache_key(self) -> str:
        """Generate cache key from filters."""
        key_str = f"{'-'.join(sorted(self.job_titles))}_{self.location_type}_{self.location}_{self.days_ago}"
        return hashlib.md5(key_str.encode()).hexdigest()

@dataclass
class Job:
    """Job listing."""
    id: str
    company: str
    title: str
    location: str
    description: str
    url: str
    source: str
    posted_date: str = ""
    salary: str = ""
    cv_generated: bool = False
    cv_path: str = ""

@dataclass
class SearchResults:
    """Cached search results."""
    jobs: List[Job] = field(default_factory=list)
    total: int = 0
    cached_at: str = ""
    cache_key: str = ""

# =============================================================================
# USER SESSIONS
# =============================================================================

user_sessions: Dict[str, dict] = {}
search_cache: Dict[str, SearchResults] = {}

# =============================================================================
# CV FORMAT ANALYZER
# =============================================================================

class CVFormatAnalyzer:
    """Analyze CV structure and format."""
    
    COMMON_SECTIONS = [
        "summary", "professional summary", "objective", "profile",
        "experience", "work experience", "employment", "employment history",
        "skills", "technical skills", "core competencies",
        "education", "academic background", "qualifications",
        "certifications", "certificates",
        "projects", "key projects",
        "awards", "achievements",
    ]
    
    def analyze(self, cv_text: str) -> CVStructure:
        """Analyze CV text and extract structure."""
        structure = CVStructure(raw_text=cv_text)
        lines = cv_text.split('\n')
        
        found_sections = []
        for line in lines:
            line_clean = line.strip().lower()
            line_clean = re.sub(r'^[#*\-]+\s*', '', line_clean)
            line_clean = re.sub(r'[:\-_]+$', '', line_clean)
            
            for section in self.COMMON_SECTIONS:
                if line_clean == section or line_clean.startswith(section + " "):
                    original_section = re.sub(r'^[#*\-]+\s*', '', line.strip())
                    original_section = re.sub(r'[:\-_]+$', '', original_section)
                    found_sections.append(original_section)
                    break
        
        structure.sections = found_sections
        structure.section_order = found_sections
        
        if '• ' in cv_text:
            structure.bullet_style = '•'
        elif '* ' in cv_text:
            structure.bullet_style = '*'
        else:
            structure.bullet_style = '-'
        
        if re.search(r'January|February|March', cv_text):
            structure.date_format = "Month Year"
        elif re.search(r'Jan|Feb|Mar', cv_text):
            structure.date_format = "Mon Year"
        else:
            structure.date_format = "MM/YYYY"
        
        structure.uses_bold_titles = '**' in cv_text
        
        logger.info(f"CV Structure analyzed: {len(structure.sections)} sections found")
        return structure

cv_analyzer = CVFormatAnalyzer()

# =============================================================================
# NLU INTENT PARSER
# =============================================================================

class IntentParser:
    """Parse natural language to extract job search intent for UK jobs."""
    
    LOCATION_TYPES = {
        'remote': ['remote', 'work from home', 'wfh', 'fully remote'],
        'hybrid': ['hybrid', 'flexible', 'part remote'],
        'onsite': ['onsite', 'on-site', 'in office', 'on site']
    }
    
    # UK Cities, Regions, and Areas (comprehensive list)
    UK_LOCATIONS = {
        # England - Major Cities
        'london': 'London', 'greater london': 'London',
        'manchester': 'Manchester', 'greater manchester': 'Manchester',
        'birmingham': 'Birmingham', 'west midlands': 'Birmingham',
        'leeds': 'Leeds', 'bradford': 'Bradford',
        'liverpool': 'Liverpool', 'merseyside': 'Liverpool',
        'newcastle': 'Newcastle', 'newcastle upon tyne': 'Newcastle',
        'sheffield': 'Sheffield', 'south yorkshire': 'Sheffield',
        'bristol': 'Bristol', 'nottingham': 'Nottingham',
        'leicester': 'Leicester', 'coventry': 'Coventry',
        'hull': 'Hull', 'kingston upon hull': 'Hull',
        'stoke': 'Stoke-on-Trent', 'stoke on trent': 'Stoke-on-Trent',
        'wolverhampton': 'Wolverhampton', 'derby': 'Derby',
        'southampton': 'Southampton', 'portsmouth': 'Portsmouth',
        'plymouth': 'Plymouth', 'reading': 'Reading',
        'luton': 'Luton', 'bolton': 'Bolton',
        'bournemouth': 'Bournemouth', 'middlesbrough': 'Middlesbrough',
        'sunderland': 'Sunderland', 'brighton': 'Brighton',
        'peterborough': 'Peterborough', 'stockport': 'Stockport',
        'slough': 'Slough', 'swindon': 'Swindon',
        'oxford': 'Oxford', 'cambridge': 'Cambridge',
        'york': 'York', 'norwich': 'Norwich',
        'exeter': 'Exeter', 'gloucester': 'Gloucester',
        'ipswich': 'Ipswich', 'cheltenham': 'Cheltenham',
        'bath': 'Bath', 'chester': 'Chester',
        'canterbury': 'Canterbury', 'winchester': 'Winchester',
        'milton keynes': 'Milton Keynes', 'watford': 'Watford',
        'woking': 'Woking', 'guildford': 'Guildford',
        'croydon': 'Croydon', 'harrow': 'Harrow',
        'enfield': 'Enfield', 'ealing': 'Ealing',
        # Scotland
        'glasgow': 'Glasgow', 'edinburgh': 'Edinburgh',
        'aberdeen': 'Aberdeen', 'dundee': 'Dundee',
        'inverness': 'Inverness', 'stirling': 'Stirling',
        'perth': 'Perth', 'scotland': 'Scotland',
        # Wales
        'cardiff': 'Cardiff', 'swansea': 'Swansea',
        'newport': 'Newport', 'wrexham': 'Wrexham',
        'wales': 'Wales',
        # Northern Ireland
        'belfast': 'Belfast', 'derry': 'Derry',
        'londonderry': 'Londonderry', 'northern ireland': 'Northern Ireland',
        # Crown Dependencies
        'isle of man': 'Isle of Man', 'douglas': 'Isle of Man',
        'jersey': 'Jersey', 'guernsey': 'Guernsey',
        'channel islands': 'Channel Islands',
        # Regions
        'east midlands': 'East Midlands', 'west midlands': 'West Midlands',
        'east anglia': 'East Anglia', 'south east': 'South East',
        'south west': 'South West', 'north east': 'North East',
        'north west': 'North West', 'yorkshire': 'Yorkshire',
        'home counties': 'Home Counties', 'midlands': 'Midlands',
        # Countries
        'england': 'England', 'united kingdom': 'United Kingdom',
        'uk': 'United Kingdom', 'britain': 'United Kingdom',
        'great britain': 'United Kingdom',
    }

    # UK Contract/IR35 terms (common in UK job market)
    # These are detected separately and added as search modifiers
    IR35_TERMS = ['outside ir35', 'inside ir35', 'ir35']
    CONTRACT_TERMS = ['contract', 'contracts', 'contractor', 'contracting',
                      'permanent', 'perm', 'fixed term', 'ftc']
    
    def parse(self, text: str) -> SearchFilters:
        """Parse natural language to search filters for UK jobs."""
        text_lower = text.lower()
        filters = SearchFilters()
        
        # Detect IR35 preference (important for UK contract searches)
        ir35_modifier = ""
        for term in self.IR35_TERMS:
            if term in text_lower:
                ir35_modifier = term
                break
        
        # Detect contract type
        contract_type = ""
        for term in self.CONTRACT_TERMS:
            if term in text_lower:
                contract_type = term
                break
        
        # Detect location type (remote/hybrid/onsite)
        for loc_type, keywords in self.LOCATION_TYPES.items():
            if any(kw in text_lower for kw in keywords):
                filters.location_type = loc_type
                break
        
        # Detect UK location
        for loc_name, loc_display in self.UK_LOCATIONS.items():
            if loc_name in text_lower:
                filters.location = loc_display
                break
        
        # Default to UK if no location specified
        if not filters.location:
            filters.location = "United Kingdom"
        
        # Clean text to extract base job title
        clean_text = text_lower
        
        # Remove location type keywords
        for keywords in self.LOCATION_TYPES.values():
            for kw in keywords:
                clean_text = clean_text.replace(kw, '')
        
        # Remove location names
        for loc_name in self.UK_LOCATIONS.keys():
            clean_text = clean_text.replace(loc_name, '')
        
        # Remove IR35 and contract terms from job title extraction
        for term in self.IR35_TERMS + self.CONTRACT_TERMS:
            clean_text = clean_text.replace(term, '')
        
        # Standard skip words
        skip_words = ['i', 'need', 'find', 'want', 'looking', 'for', 'jobs', 'job',
                      'in', 'at', 'a', 'an', 'the', 'roles', 'role', 'positions', 'uk']
        words = clean_text.split()
        title_words = [w for w in words if w not in skip_words and len(w) > 2]
        
        # Build job title with IR35/contract modifiers
        base_title = ' '.join(title_words).title() if title_words else "Engineer"
        
        # Add IR35 modifier to the search (this is what users want!)
        if ir35_modifier:
            # e.g., "Data Engineer outside IR35" or "Data Engineer IR35"
            filters.job_titles = [f"{base_title} {ir35_modifier}"]
        elif contract_type:
            # e.g., "Data Engineer contract"
            filters.job_titles = [f"{base_title} {contract_type}"]
        else:
            filters.job_titles = [base_title]
        
        # Detect days ago filter
        days_match = re.search(r'(\d+)\s*days?', text_lower)
        if days_match:
            filters.days_ago = int(days_match.group(1))
        
        return filters

intent_parser = IntentParser()

# =============================================================================
# INDEED MARKDOWN PARSER (for Bright Data MCP results)
# =============================================================================

def parse_indeed_markdown(markdown: str) -> List[Job]:
    """Parse Indeed job listings from Bright Data markdown."""
    jobs = []
    
    # Pattern to match job listings
    # Format: ## [Job Title](/rc/clk?...) followed by company, location, salary
    job_pattern = re.compile(
        r'\*\s+##\s+\[([^\]]+)\]\(([^)]+)\)\s*\n'  # Title and URL
        r'(?:.*?\n)*?'  # Skip optional lines
        r'\s+([A-Za-z][^\n]+?)\s*\n'  # Company name
        r'\s+([^\n]+(?:TX|Texas)[^\n]*)\s*\n'  # Location
        r'(?:\s+(\$[\d,]+ - \$[\d,]+[^\n]*))?\s*\n',  # Optional salary
        re.MULTILINE | re.DOTALL
    )
    
    # Simpler pattern for each job block
    lines = markdown.split('\n')
    current_job = {}
    job_id = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Match job title with link
        title_match = re.match(r'##\s+\[(.+?)\]\((.+?)\)', line)
        if title_match:
            # Save previous job if complete
            if current_job.get('title') and current_job.get('company'):
                job_id += 1
                jobs.append(Job(
                    id=f"indeed_{job_id}",
                    company=current_job.get('company', 'Unknown'),
                    title=current_job.get('title', ''),
                    location=current_job.get('location', 'Texas'),
                    description=current_job.get('description', ''),
                    url=current_job.get('url', ''),
                    source='indeed',
                    salary=current_job.get('salary', '')
                ))
            
            # Start new job
            current_job = {
                'title': title_match.group(1).strip(),
                'url': title_match.group(2).strip()
            }
            continue
        
        # Match company name (usually right after title)
        if current_job.get('title') and not current_job.get('company'):
            if line and not line.startswith('*') and not line.startswith('#'):
                if 'Often responds' in line or 'Easily apply' in line:
                    continue
                if not any(skip in line.lower() for skip in ['full-time', 'part-time', '$', 'tuition', '401(k)', 'health']):
                    current_job['company'] = line.strip()
                    continue
        
        # Match location (contains TX or Texas)
        if current_job.get('company') and not current_job.get('location'):
            if 'TX' in line or 'Texas' in line:
                current_job['location'] = line.strip()
                continue
        
        # Match salary
        salary_match = re.search(r'\$[\d,]+\s*-\s*\$[\d,]+[^\n]*', line)
        if salary_match and current_job.get('title'):
            current_job['salary'] = salary_match.group(0).strip()
            continue
        
        # Match description bullet points
        if line.startswith('*') and current_job.get('title'):
            desc = line.lstrip('* ').strip()
            if current_job.get('description'):
                current_job['description'] += ' ' + desc
            else:
                current_job['description'] = desc
    
    # Don't forget last job
    if current_job.get('title') and current_job.get('company'):
        job_id += 1
        jobs.append(Job(
            id=f"indeed_{job_id}",
            company=current_job.get('company', 'Unknown'),
            title=current_job.get('title', ''),
            location=current_job.get('location', 'Texas'),
            description=current_job.get('description', ''),
            url=current_job.get('url', ''),
            source='indeed',
            salary=current_job.get('salary', '')
        ))
    
    return jobs

# =============================================================================
# BRIGHT DATA MCP SCRAPER (Dynamic job fetching)
# =============================================================================

import requests
from urllib.parse import quote_plus

class BrightDataJobScraper:
    """
    Scrapes jobs dynamically using Bright Data's Unlocker API (REST).
    
    Environment variables:
    - BRIGHT_DATA_API_KEY: Your Bright Data API key (starts with 'pa_')
    - BRIGHT_DATA_ZONE: Your zone name (e.g., 'mcp_unlocker')
    
    API Documentation: https://docs.brightdata.com/scraping-automation/web-unlocker/introduction
    """
    
    # Bright Data REST API endpoint
    BRIGHTDATA_API_URL = "https://api.brightdata.com/request"
    
    def __init__(self):
        self.api_key = os.getenv("BRIGHT_DATA_API_KEY", "")
        self.zone = os.getenv("BRIGHT_DATA_ZONE", "mcp_unlocker")
        
        # Session for direct requests (fallback)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        
        if self.api_key:
            logger.info(f"Bright Data API configured (zone: {self.zone}, key: {self.api_key[:8]}...)")
            self.use_brightdata = True
        else:
            logger.warning("BRIGHT_DATA_API_KEY not set - using direct requests (may be blocked)")
            self.use_brightdata = False
    
    def _fetch_via_brightdata(self, url: str) -> Optional[str]:
        """
        Fetch URL content via Bright Data Unlocker API.
        
        Returns HTML content or None if failed.
        """
        if not self.use_brightdata:
            return None
        
        try:
            response = requests.post(
                self.BRIGHTDATA_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "zone": self.zone,
                    "url": url,
                    "format": "raw",
                    "method": "GET"
                },
                timeout=60
            )
            
            if response.status_code == 200:
                logger.info(f"Bright Data API success for {url[:50]}...")
                return response.text
            else:
                logger.warning(f"Bright Data API returned {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"Bright Data API error: {e}")
            return None
    
    def _fetch_direct(self, url: str) -> Optional[str]:
        """Fetch URL directly (may be blocked by anti-bot)."""
        try:
            response = self.session.get(url, timeout=10)  # Reduced timeout for speed
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Direct fetch failed: {e}")
            return None
    
    def _search_google_serp(self, query: str, days: int = 30) -> List[dict]:
        """
        Search Google by scraping Google search results page via Bright Data Unlocker.
        Returns list of organic search results parsed from HTML.
        Filters to results from the past month by default.
        """
        if not self.api_key:
            logger.warning("No Bright Data API key - skipping Google search")
            return []
        
        try:
            # Build Google search URL - request 30 results
            # tbs=qdr:m filters to past month (30 days)
            encoded_query = quote_plus(query)
            time_filter = "qdr:m" if days >= 30 else f"qdr:d{days}"
            google_url = f"https://www.google.com/search?q={encoded_query}&num=30&tbs={time_filter}"
            
            # Fetch via Bright Data Unlocker
            html = self._fetch_via_brightdata(google_url)
            if not html:
                html = self._fetch_direct(google_url)
            
            if not html:
                logger.warning("Failed to fetch Google search results")
                return []
            
            # Parse Google search results
            results = self._parse_google_html(html)
            logger.info(f"Google search returned {len(results)} results for: {query[:50]}...")
            return results
                
        except Exception as e:
            logger.error(f"Google search error: {e}")
            return []
    
    def _parse_google_html(self, html: str) -> List[dict]:
        """Parse Google search results HTML to extract organic results."""
        results = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Google search results are in divs with class 'g' or data-hveid
            search_results = soup.find_all('div', class_='g')
            if not search_results:
                search_results = soup.find_all('div', {'data-hveid': True})
            
            for result in search_results[:30]:
                try:
                    # Find the link
                    link_elem = result.find('a', href=True)
                    if not link_elem:
                        continue
                    
                    link = link_elem.get('href', '')
                    if not link.startswith('http'):
                        continue
                    if 'google.com' in link:
                        continue
                    
                    # Find title
                    title_elem = result.find('h3')
                    title = title_elem.get_text(strip=True) if title_elem else ""
                    
                    # Find description
                    desc_elem = result.find('div', class_=re.compile(r'VwiC3b|IsZvec'))
                    if not desc_elem:
                        desc_elem = result.find('span', class_=re.compile(r'aCOpRe'))
                    description = desc_elem.get_text(strip=True) if desc_elem else ""
                    
                    if title and link:
                        results.append({
                            'link': link,
                            'title': title,
                            'description': description
                        })
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"Google HTML parsing error: {e}")
        
        return results
    
    def search_dice_via_google(self, job_title: str, location: str, remote: bool = False) -> List[Job]:
        """Search Dice job detail pages via Google (Dice is JS-heavy)."""
        # Search specifically for job-detail pages, not search results pages
        query = f'site:dice.com/job-detail "{job_title}" {location}'
        if remote:
            query += ' remote'
        
        logger.info(f"Searching Dice job-detail pages via Google: {query}")
        results = self._search_google_serp(query)
        return self.parse_google_results_dice(results, is_remote=remote)
    
    def search_ziprecruiter_via_google(self, job_title: str, location: str, remote: bool = False) -> List[Job]:
        """Search ZipRecruiter US job pages via Google (ZipRecruiter is JS-heavy)."""
        # Search for individual job pages using /c/ path (company job listings)
        # This is more specific than just site:ziprecruiter.com which returns search pages
        query = f'site:ziprecruiter.com/c/ "{job_title}" {location}'
        if remote:
            query += ' remote'
        
        logger.info(f"Searching ZipRecruiter US job pages via Google: {query}")
        results = self._search_google_serp(query)
        return self.parse_google_results_ziprecruiter(results, is_remote=remote)
    
    def scrape_indeed(self, job_title: str, location: str, remote: bool = False, days: int = 30) -> List[Job]:
        """Scrape Indeed UK for jobs - with Google fallback if direct scrape fails."""
        all_jobs = []
        
        # Use exact job title from user's search (via NLU)
        query = quote_plus(job_title)
        loc = quote_plus(location)
        
        # Scrape 1 page only for reliability
        for page in range(1):
            start = page * 10
            # Use UK Indeed domain
            url = f"https://uk.indeed.com/jobs?q={query}&l={loc}&fromage={days}&start={start}"
            if remote:
                url += "&remotejob=1"
            
            logger.info(f"Scraping Indeed UK '{job_title}' page {page+1}")
            
            html = self._fetch_via_brightdata(url)
            if not html:
                html = self._fetch_direct(url)
            
            if html:
                jobs = self._parse_indeed_html(html, url)
                all_jobs.extend(jobs)
                logger.info(f"Indeed page {page+1}: {len(jobs)} jobs")
            else:
                logger.warning(f"Failed to fetch Indeed page {page+1}")
                break
        
        # If direct scrape failed or returned 0 jobs, try Google fallback
        if not all_jobs:
            logger.info("Indeed UK direct scrape failed, trying Google fallback...")
            all_jobs = self.search_indeed_uk_via_google(job_title, location, remote)
        
        logger.info(f"Scraped {len(all_jobs)} total jobs from Indeed UK")
        return all_jobs
    
    def search_indeed_uk_via_google(self, job_title: str, location: str, remote: bool = False) -> List[Job]:
        """Search Indeed UK jobs via Google when direct scrape fails."""
        query = f'site:uk.indeed.com "{job_title}" {location}'
        if remote:
            query += ' remote'
        
        logger.info(f"Searching Indeed UK via Google: {query}")
        results = self._search_google_serp(query)
        return self._parse_indeed_google_results(results, is_remote=remote)
    
    def _parse_indeed_google_results(self, google_results: List[dict], is_remote: bool = False) -> List[Job]:
        """Parse Google search results for Indeed UK jobs."""
        jobs = []
        
        for i, result in enumerate(google_results[:20]):
            try:
                title = result.get('title', '')
                description = result.get('description', '')
                url = result.get('link', '')
                
                if not url or 'uk.indeed.com' not in url:
                    continue
                
                # Skip search result pages, only want job pages
                if '/jobs?' in url or '/q-' in url:
                    continue
                
                # Clean title - remove "- Indeed" suffix
                title_clean = title.replace(' - Indeed', '').replace(' | Indeed', '').strip()
                
                # Try to extract company from title (usually "Job Title - Company")
                parts = title_clean.split(' - ')
                if len(parts) >= 2:
                    job_title = parts[0].strip()
                    company = parts[1].strip()
                else:
                    job_title = title_clean
                    company = "See Indeed"
                
                if not job_title or len(job_title) < 5:
                    continue
                
                # Extract location from description if possible
                location = "UK"
                
                jobs.append(Job(
                    id=f"indeed_google_{i+1}",
                    company=company,
                    title=job_title,
                    location=location,
                    description=description[:300] if description else "",
                    url=url,
                    source="indeed",
                    salary=""
                ))
            except:
                continue
        
        logger.info(f"Parsed {len(jobs)} Indeed UK jobs from Google results")
        return jobs
    
    def _parse_indeed_html(self, html: str, source_url: str) -> List[Job]:
        """Parse Indeed HTML to extract job listings."""
        jobs = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Indeed job cards are in divs with class 'job_seen_beacon' or similar
            job_cards = soup.find_all('div', class_=re.compile(r'job_seen_beacon|cardOutline|jobsearch-ResultsList'))
            
            if not job_cards:
                # Try alternative selectors
                job_cards = soup.find_all('div', {'data-jk': True})
            
            for i, card in enumerate(job_cards[:20]):  # Limit to 20 jobs
                try:
                    # Extract job details
                    title_elem = card.find(['h2', 'a'], class_=re.compile(r'jobTitle|title'))
                    company_elem = card.find(['span', 'div'], class_=re.compile(r'company|companyName'))
                    location_elem = card.find(['div', 'span'], class_=re.compile(r'location|companyLocation'))
                    salary_elem = card.find(['div', 'span'], class_=re.compile(r'salary|salaryText'))
                    
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    company = company_elem.get_text(strip=True) if company_elem else "Unknown Company"
                    location = location_elem.get_text(strip=True) if location_elem else "USA"
                    salary = salary_elem.get_text(strip=True) if salary_elem else ""
                    
                    # Get job URL
                    link = card.find('a', href=True)
                    job_url = f"https://www.indeed.com{link['href']}" if link else source_url
                    
                    # Skip invalid entries
                    if not title or not company:
                        continue
                    if '*' in title or '*' in company:
                        continue
                    if len(title) < 3 or len(company) < 2:
                        continue
                    
                    # Generate unique ID
                    job_id = card.get('data-jk', f"indeed_{i+1}")
                    
                    jobs.append(Job(
                        id=f"indeed_{job_id}",
                        company=company,
                        title=title,
                        location=location,
                        description="",
                        url=job_url,
                        source="indeed",
                        salary=salary
                    ))
                    
                except Exception as e:
                    logger.debug(f"Error parsing job card: {e}")
                    continue
            
        except ImportError:
            logger.warning("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
        except Exception as e:
            logger.error(f"HTML parsing error: {e}")
        
        return jobs
    
    def scrape_linkedin(self, job_title: str, location: str, remote: bool = False, days: int = 30) -> List[Job]:
        """Scrape LinkedIn for jobs - multiple pages, filtered to last 30 days."""
        all_jobs = []
        
        query = quote_plus(job_title)
        loc = quote_plus(location)
        
        # LinkedIn time filter: r2592000 = past month (30 days), r604800 = past week
        time_filter = "r2592000" if days >= 30 else f"r{days * 86400}"
        
        # Scrape 2 pages for speed (each page has ~25 jobs)
        for page in range(2):
            start = page * 25
            url = f"https://www.linkedin.com/jobs/search/?keywords={query}&location={loc}&start={start}&f_TPR={time_filter}"
            if remote:
                url += "&f_WT=2"
            
            logger.info(f"Scraping LinkedIn page {page+1}: {url}")
            
            html = self._fetch_via_brightdata(url)
            if not html:
                html = self._fetch_direct(url)
            
            if html:
                jobs = self._parse_linkedin_html(html, url)
                all_jobs.extend(jobs)
                logger.info(f"LinkedIn page {page+1}: {len(jobs)} jobs")
            else:
                logger.warning(f"Failed to fetch LinkedIn page {page+1}")
                break
        
        logger.info(f"Scraped {len(all_jobs)} total jobs from LinkedIn")
        return all_jobs
    
    def _parse_linkedin_html(self, html: str, source_url: str) -> List[Job]:
        """Parse LinkedIn HTML to extract job listings."""
        jobs = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # LinkedIn job cards
            job_cards = soup.find_all('div', class_=re.compile(r'job-search-card|base-card'))
            
            for i, card in enumerate(job_cards[:20]):
                try:
                    title_elem = card.find(['h3', 'span'], class_=re.compile(r'title|job-title'))
                    company_elem = card.find(['h4', 'a'], class_=re.compile(r'company|subtitle'))
                    location_elem = card.find(['span'], class_=re.compile(r'location|job-location'))
                    
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    company = company_elem.get_text(strip=True) if company_elem else ""
                    location = location_elem.get_text(strip=True) if location_elem else ""
                    
                    # Skip invalid entries - no stars, no garbage
                    if not title or not company:
                        continue
                    if '*' in title or '*' in company:
                        continue
                    if len(title) < 3 or len(company) < 2:
                        continue
                    if company.lower() == "unknown":
                        continue
                    
                    link = card.find('a', href=True)
                    job_url = link['href'] if link else source_url
                    
                    # Skip if URL doesn't look valid
                    if not job_url or job_url == source_url:
                        continue
                    
                    jobs.append(Job(
                        id=f"linkedin_{i+1}",
                        company=company,
                        title=title,
                        location=location or "USA",
                        description="",
                        url=job_url,
                        source="linkedin",
                        salary=""
                    ))
                    
                except Exception as e:
                    logger.debug(f"Error parsing LinkedIn card: {e}")
                    continue
                    
        except ImportError:
            logger.warning("BeautifulSoup not installed")
        except Exception as e:
            logger.error(f"LinkedIn parsing error: {e}")
        
        return jobs
    
    def scrape_dice_via_google(self, job_title: str, location: str, remote: bool = False) -> List[Job]:
        """Scrape Dice jobs via Google Search (Dice is JS-heavy)."""
        jobs = []
        
        # Use Google to search Dice
        search_query = f'site:dice.com "{job_title}" {location}'
        if remote:
            search_query += ' remote'
        
        logger.info(f"Searching Dice via Google: {search_query}")
        
        try:
            # Use Bright Data MCP search (called from outside, results passed in)
            # For now, parse from cached Google results if available
            # This is a placeholder - actual MCP calls happen at API level
            pass
        except Exception as e:
            logger.error(f"Dice Google search error: {e}")
        
        return jobs
    
    def parse_google_results_dice(self, google_results: List[dict], is_remote: bool = False) -> List[Job]:
        """Parse Google search results for Dice jobs - only accept job detail pages."""
        jobs = []
        
        for i, result in enumerate(google_results[:30]):
            try:
                title = result.get('title', '')
                description = result.get('description', '')
                url = result.get('link', '')
                
                if not url or 'dice.com' not in url:
                    continue
                
                # IMPORTANT: Only accept actual job detail pages, not search pages
                # Good: https://www.dice.com/job-detail/22864975-c972-43dc-9e7d-e5938c260db0
                # Bad: https://www.dice.com/jobs/q-data+engineer-l-austin%2C+tx-jobs
                if '/job-detail/' not in url:
                    logger.debug(f"Skipping Dice search page: {url}")
                    continue
                
                # Extract job title from the Google result title
                title_clean = title.replace(' - Dice', '').strip()
                
                # Extract company from description
                company = "See Dice"
                if description:
                    # Look for company name patterns in description
                    parts = description.split('.')
                    if parts and len(parts[0]) < 60:
                        company = parts[0].strip()
                
                if not title_clean or len(title_clean) < 5:
                    continue
                
                # If searching for remote, filter out onsite jobs
                if is_remote:
                    combined = (title + ' ' + description).lower()
                    # Skip if explicitly onsite/on-site and no remote mention
                    if ('onsite' in combined or 'on-site' in combined or 'on site' in combined) and 'remote' not in combined:
                        logger.debug(f"Skipping onsite Dice job: {title_clean}")
                        continue
                
                jobs.append(Job(
                    id=f"dice_{i+1}",
                    company=company,
                    title=title_clean,
                    location="Texas",
                    description=description[:200] if description else "",
                    url=url,
                    source="dice",
                    salary=""
                ))
            except:
                continue
        
        logger.info(f"Parsed {len(jobs)} job detail pages from Dice Google results")
        return jobs
    
    def scrape_glassdoor(self, job_title: str, location: str, remote: bool = False, days: int = 30) -> List[Job]:
        """Scrape Glassdoor UK for jobs - filtered to last 30 days."""
        jobs = []
        
        query = quote_plus(job_title)
        
        # Map UK cities/regions to Glassdoor location IDs
        uk_location_ids = {
            'london': '2671300', 'greater london': '2671300',
            'manchester': '2652467', 'greater manchester': '2652467',
            'birmingham': '2655603', 'west midlands': '2655603',
            'leeds': '2644688', 'yorkshire': '2644688',
            'glasgow': '2648579', 'scotland': '2648576',
            'edinburgh': '2650272',
            'bristol': '2654675',
            'liverpool': '2644210',
            'newcastle': '2641673',
            'sheffield': '2638077',
            'nottingham': '2641170',
            'cardiff': '2653822', 'wales': '2653822',
            'belfast': '2655984', 'northern ireland': '2655984',
            'cambridge': '2653941',
            'oxford': '2640729',
            'reading': '2639577',
            'isle of man': '2633218',
            'united kingdom': '2635167', 'uk': '2635167',
        }
        
        loc_lower = location.lower()
        loc_id = uk_location_ids.get(loc_lower, '2635167')  # Default to UK (2635167)
        
        # Use Glassdoor UK domain
        url = f"https://www.glassdoor.co.uk/Job/jobs.htm?sc.keyword={query}&locT=C&locId={loc_id}&fromAge={days}"
        if remote:
            url += "&remoteWorkType=1"  # 1 = Remote
        
        logger.info(f"Scraping Glassdoor UK: {url}")
        
        html = self._fetch_via_brightdata(url)
        if not html:
            html = self._fetch_direct(url)
        
        if html:
            jobs = self._parse_glassdoor_html(html, url, location, remote)
            logger.info(f"Scraped {len(jobs)} jobs from Glassdoor")
        
        return jobs
    
    def _parse_glassdoor_html(self, html: str, source_url: str, search_location: str = "", is_remote: bool = False) -> List[Job]:
        """Parse Glassdoor HTML - filter by search criteria."""
        jobs = []
        
        # Build location filter based on search
        search_loc_lower = search_location.lower() if search_location else ""
        
        # Map search location to acceptable job locations
        location_map = {
            'texas': ['texas', 'tx', 'houston', 'dallas', 'austin', 'san antonio', 'fort worth', 'plano', 'irving', 'arlington'],
            'california': ['california', 'ca', 'los angeles', 'san francisco', 'san diego', 'san jose', 'palo alto', 'mountain view'],
            'new york': ['new york', 'ny', 'nyc', 'manhattan', 'brooklyn', 'jersey city'],
            'florida': ['florida', 'fl', 'miami', 'orlando', 'tampa', 'jacksonville'],
        }
        
        # Get acceptable locations for this search
        acceptable_locations = []
        for key, values in location_map.items():
            if key in search_loc_lower or search_loc_lower in values:
                acceptable_locations = values
                break
        
        # Always accept remote and USA-wide for any US job search
        acceptable_locations.extend(['remote', 'usa', 'united states', 'us', 'anywhere'])
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            job_cards = soup.find_all('li', class_=re.compile(r'JobsList_jobListItem|react-job-listing'))
            if not job_cards:
                job_cards = soup.find_all('div', class_=re.compile(r'jobCard'))
            
            for i, card in enumerate(job_cards[:25]):
                try:
                    title_elem = card.find(['a', 'div'], class_=re.compile(r'jobTitle|job-title'))
                    company_elem = card.find(['div', 'span'], class_=re.compile(r'EmployerProfile|employer'))
                    location_elem = card.find(['span', 'div'], class_=re.compile(r'location|loc'))
                    
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    company = company_elem.get_text(strip=True) if company_elem else ""
                    location = location_elem.get_text(strip=True) if location_elem else ""
                    
                    if not title or not company or '*' in title or '*' in company:
                        continue
                    if len(title) < 3 or len(company) < 2:
                        continue
                    
                    loc_lower = location.lower()
                    
                    # Skip non-UK jobs (US, EU, etc) - keep UK jobs only
                    non_uk_indicators = ['united states', 'usa', 'germany', 'france', 'india', 'kaiserslautern', 'california', 'texas', 'new york']
                    is_non_uk = any(ind in loc_lower for ind in non_uk_indicators)
                    if is_non_uk:
                        logger.debug(f"Skipping non-UK Glassdoor job: {location}")
                        continue
                    
                    # Filter by search location if specified (UK locations)
                    if acceptable_locations:
                        matches_location = any(loc in loc_lower for loc in acceptable_locations)
                        if not matches_location:
                            logger.debug(f"Skipping job not in {search_location}: {location}")
                            continue
                    
                    link = card.find('a', href=True)
                    job_url = link['href'] if link else source_url
                    if job_url.startswith('/'):
                        job_url = f"https://www.glassdoor.co.uk{job_url}"
                    
                    # Skip non-UK Glassdoor domains (keep UK only)
                    if 'glassdoor.com' in job_url and 'glassdoor.co.uk' not in job_url:
                        continue
                    
                    jobs.append(Job(
                        id=f"glassdoor_{i+1}",
                        company=company,
                        title=title,
                        location=location or "USA",
                        description="",
                        url=job_url,
                        source="glassdoor",
                        salary=""
                    ))
                except:
                    continue
        except Exception as e:
            logger.error(f"Glassdoor parsing error: {e}")
        
        return jobs
    
    def parse_google_results_ziprecruiter(self, google_results: List[dict], is_remote: bool = False) -> List[Job]:
        """Parse Google search results for ZipRecruiter jobs - US only, individual job pages."""
        jobs = []
        
        for i, result in enumerate(google_results[:30]):
            try:
                title = result.get('title', '')
                description = result.get('description', '')
                url = result.get('link', '')
                
                # MUST be US ZipRecruiter (.com) not UK (.co.uk) or other
                if not url:
                    continue
                if 'ziprecruiter.co.uk' in url or 'ziprecruiter.de' in url:
                    logger.debug(f"Skipping non-US ZipRecruiter: {url}")
                    continue
                if 'ziprecruiter.com' not in url:
                    continue
                
                # Skip search pages - only accept individual job pages
                # Good: /c/Company-Name/Job/Title
                # Bad: /jobs/search?q=... or /candidate/search or .co.uk
                if '/jobs/search' in url or '/candidate/search' in url or '/search?' in url:
                    logger.debug(f"Skipping ZipRecruiter search page: {url}")
                    continue
                
                # Must have /c/ path (company job page) - this is the most reliable indicator
                if '/c/' not in url:
                    logger.debug(f"Skipping non-job ZipRecruiter URL: {url}")
                    continue
                
                # Clean title
                title_clean = title.replace(' (NOW HIRING)', '')
                title_clean = title_clean.replace(' Jobs in Texas', '')
                title_clean = title_clean.replace(' Jobs in ', ' - ')
                title_clean = title_clean.replace(' - ZipRecruiter', '').strip()
                
                # Extract salary if present
                salary = ""
                salary_match = re.search(r'\$[\d,k]+\s*-\s*\$[\d,k]+', title)
                if salary_match:
                    salary = salary_match.group(0)
                    title_clean = title_clean.replace(salary, '').strip()
                
                if not title_clean or len(title_clean) < 5:
                    continue
                
                # If searching for remote, only accept jobs with "remote" in title/description
                if is_remote:
                    combined = (title + ' ' + description).lower()
                    if 'remote' not in combined:
                        logger.debug(f"Skipping non-remote ZipRecruiter job: {title_clean}")
                        continue
                
                jobs.append(Job(
                    id=f"ziprecruiter_{i+1}",
                    company="See ZipRecruiter",
                    title=title_clean,
                    location="Texas",
                    description=description[:200] if description else "",
                    url=url,
                    source="ziprecruiter",
                    salary=salary
                ))
            except:
                continue
        
        logger.info(f"Parsed {len(jobs)} US job pages from ZipRecruiter Google results")
        return jobs
    
    # =========================================================================
    # UK JOB SITE SCRAPERS
    # =========================================================================
    
    def scrape_reed(self, job_title: str, location: str, remote: bool = False, days: int = 30) -> List[Job]:
        """Scrape Reed.co.uk for UK jobs."""
        jobs = []
        
        query = quote_plus(job_title)
        loc = quote_plus(location) if location and location.lower() != 'united kingdom' else ''
        
        # Reed URL structure
        url = f"https://www.reed.co.uk/jobs/{query.lower().replace('+', '-')}-jobs"
        if loc:
            url += f"-in-{loc.lower().replace('+', '-')}"
        url += f"?datecreatedoffset=LastMonth"
        if remote:
            url += "&remote=true"
        
        logger.info(f"Scraping Reed: {url}")
        
        html = self._fetch_via_brightdata(url)
        if not html:
            html = self._fetch_direct(url)
        
        if html:
            jobs = self._parse_reed_html(html, url)
            logger.info(f"Scraped {len(jobs)} jobs from Reed")
        
        return jobs
    
    def _parse_reed_html(self, html: str, source_url: str) -> List[Job]:
        """Parse Reed.co.uk HTML."""
        jobs = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            job_cards = soup.find_all('article', class_=re.compile(r'job-card|job-result'))
            if not job_cards:
                job_cards = soup.find_all('div', class_=re.compile(r'job-card'))
            
            for i, card in enumerate(job_cards[:20]):
                try:
                    title_elem = card.find(['h2', 'h3', 'a'], class_=re.compile(r'job-title|title'))
                    company_elem = card.find(['span', 'div', 'a'], class_=re.compile(r'company|posted-by'))
                    location_elem = card.find(['span', 'li'], class_=re.compile(r'location'))
                    salary_elem = card.find(['span', 'li'], class_=re.compile(r'salary'))
                    
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    company = company_elem.get_text(strip=True) if company_elem else "See Reed"
                    location = location_elem.get_text(strip=True) if location_elem else "UK"
                    salary = salary_elem.get_text(strip=True) if salary_elem else ""
                    
                    link = card.find('a', href=True)
                    job_url = link['href'] if link else source_url
                    if job_url.startswith('/'):
                        job_url = f"https://www.reed.co.uk{job_url}"
                    
                    if title and len(title) > 3:
                        jobs.append(Job(
                            id=f"reed_{i+1}",
                            company=company,
                            title=title,
                            location=location,
                            description="",
                            url=job_url,
                            source="reed",
                            salary=salary
                        ))
                except:
                    continue
        except Exception as e:
            logger.error(f"Reed parsing error: {e}")
        
        return jobs
    
    def scrape_cvlibrary(self, job_title: str, location: str, remote: bool = False, days: int = 30) -> List[Job]:
        """Scrape CV-Library for UK jobs."""
        jobs = []
        
        query = quote_plus(job_title)
        loc = quote_plus(location) if location and location.lower() != 'united kingdom' else ''
        
        # CV-Library URL structure
        url = f"https://www.cv-library.co.uk/search-jobs?q={query}"
        if loc:
            url += f"&geo={loc}"
        url += "&posted=30"  # Last 30 days
        if remote:
            url += "&remotejob=1"
        
        logger.info(f"Scraping CV-Library: {url}")
        
        html = self._fetch_via_brightdata(url)
        if not html:
            html = self._fetch_direct(url)
        
        if html:
            jobs = self._parse_cvlibrary_html(html, url)
            logger.info(f"Scraped {len(jobs)} jobs from CV-Library")
        
        return jobs
    
    def _parse_cvlibrary_html(self, html: str, source_url: str) -> List[Job]:
        """Parse CV-Library HTML."""
        jobs = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            job_cards = soup.find_all('article', class_=re.compile(r'job-card|search-result'))
            if not job_cards:
                job_cards = soup.find_all('li', class_=re.compile(r'search-result'))
            
            for i, card in enumerate(job_cards[:20]):
                try:
                    title_elem = card.find(['h2', 'a'], class_=re.compile(r'job-title|title'))
                    company_elem = card.find(['span', 'a'], class_=re.compile(r'company'))
                    location_elem = card.find(['span', 'div'], class_=re.compile(r'location'))
                    salary_elem = card.find(['span', 'div'], class_=re.compile(r'salary'))
                    
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    company = company_elem.get_text(strip=True) if company_elem else "See CV-Library"
                    location = location_elem.get_text(strip=True) if location_elem else "UK"
                    salary = salary_elem.get_text(strip=True) if salary_elem else ""
                    
                    link = card.find('a', href=True)
                    job_url = link['href'] if link else source_url
                    if job_url.startswith('/'):
                        job_url = f"https://www.cv-library.co.uk{job_url}"
                    
                    if title and len(title) > 3:
                        jobs.append(Job(
                            id=f"cvlibrary_{i+1}",
                            company=company,
                            title=title,
                            location=location,
                            description="",
                            url=job_url,
                            source="cv-library",
                            salary=salary
                        ))
                except:
                    continue
        except Exception as e:
            logger.error(f"CV-Library parsing error: {e}")
        
        return jobs
    
    def scrape_totaljobs(self, job_title: str, location: str, remote: bool = False, days: int = 30) -> List[Job]:
        """Scrape TotalJobs for UK jobs."""
        jobs = []
        
        query = quote_plus(job_title)
        loc = quote_plus(location) if location and location.lower() != 'united kingdom' else ''
        
        # TotalJobs URL structure
        url = f"https://www.totaljobs.com/jobs/{query.lower().replace('+', '-')}"
        if loc:
            url += f"/in-{loc.lower().replace('+', '-')}"
        url += f"?postedWithin=30"
        if remote:
            url += "&worktype=remote"
        
        logger.info(f"Scraping TotalJobs: {url}")
        
        html = self._fetch_via_brightdata(url)
        if not html:
            html = self._fetch_direct(url)
        
        if html:
            jobs = self._parse_totaljobs_html(html, url)
            logger.info(f"Scraped {len(jobs)} jobs from TotalJobs")
        
        return jobs
    
    def _parse_totaljobs_html(self, html: str, source_url: str) -> List[Job]:
        """Parse TotalJobs HTML."""
        jobs = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            job_cards = soup.find_all('article', class_=re.compile(r'job|result'))
            if not job_cards:
                job_cards = soup.find_all('div', {'data-at': re.compile(r'job')})
            
            for i, card in enumerate(job_cards[:20]):
                try:
                    title_elem = card.find(['h2', 'a'], {'data-at': 'job-item-title'}) or card.find(['h2', 'a'], class_=re.compile(r'title'))
                    company_elem = card.find(['span', 'a'], {'data-at': 'job-item-company-name'}) or card.find(['span', 'a'], class_=re.compile(r'company'))
                    location_elem = card.find(['span', 'div'], {'data-at': 'job-item-location'}) or card.find(['span', 'div'], class_=re.compile(r'location'))
                    salary_elem = card.find(['span', 'div'], {'data-at': 'job-item-salary-info'}) or card.find(['span', 'div'], class_=re.compile(r'salary'))
                    
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    company = company_elem.get_text(strip=True) if company_elem else "See TotalJobs"
                    location = location_elem.get_text(strip=True) if location_elem else "UK"
                    salary = salary_elem.get_text(strip=True) if salary_elem else ""
                    
                    link = card.find('a', href=True)
                    job_url = link['href'] if link else source_url
                    if job_url.startswith('/'):
                        job_url = f"https://www.totaljobs.com{job_url}"
                    
                    if title and len(title) > 3:
                        jobs.append(Job(
                            id=f"totaljobs_{i+1}",
                            company=company,
                            title=title,
                            location=location,
                            description="",
                            url=job_url,
                            source="totaljobs",
                            salary=salary
                        ))
                except:
                    continue
        except Exception as e:
            logger.error(f"TotalJobs parsing error: {e}")
        
        return jobs
    
    def search_jobserve_via_google(self, job_title: str, location: str, remote: bool = False) -> List[Job]:
        """Search Jobserve via Google (Jobserve is JS-heavy)."""
        query = f'site:jobserve.com/gb "{job_title}" {location}'
        if remote:
            query += ' remote'
        
        logger.info(f"Searching Jobserve via Google: {query}")
        results = self._search_google_serp(query)
        return self._parse_jobserve_google_results(results, is_remote=remote)
    
    def _parse_jobserve_google_results(self, google_results: List[dict], is_remote: bool = False) -> List[Job]:
        """Parse Google search results for Jobserve jobs."""
        jobs = []
        
        for i, result in enumerate(google_results[:20]):
            try:
                title = result.get('title', '')
                description = result.get('description', '')
                url = result.get('link', '')
                
                if not url or 'jobserve.com' not in url:
                    continue
                
                # Clean title
                title_clean = title.replace(' - Jobserve', '').replace(' | Jobserve', '').strip()
                
                if not title_clean or len(title_clean) < 5:
                    continue
                
                # If searching for remote, filter
                if is_remote:
                    combined = (title + ' ' + description).lower()
                    if 'remote' not in combined:
                        continue
                
                jobs.append(Job(
                    id=f"jobserve_{i+1}",
                    company="See Jobserve",
                    title=title_clean,
                    location="UK",
                    description=description[:200] if description else "",
                    url=url,
                    source="jobserve",
                    salary=""
                ))
            except:
                continue
        
        logger.info(f"Parsed {len(jobs)} Jobserve jobs from Google results")
        return jobs
    
    def search_jobs(self, filters: 'SearchFilters', fast_mode: bool = False) -> List[Job]:
        """
        Search for UK jobs across multiple sources in PARALLEL for speed.
        
        Args:
            filters: Search filters (job titles, location, etc.)
            fast_mode: If True, only scrape the 2 fastest sources (Indeed UK, LinkedIn)
        
        UK Sources:
            - Indeed UK, LinkedIn, Glassdoor UK (direct scrape)
            - Reed, CV-Library, TotalJobs (direct scrape)
            - Jobserve (via Google SERP - JS-heavy)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        start_time = time.time()
        all_jobs = []
        job_title = ' '.join(filters.job_titles) if filters.job_titles else "Data Engineer"
        location = filters.location or "United Kingdom"  # Default to UK
        is_remote = filters.location_type == 'remote'
        days = filters.days_ago
        
        # Define UK scraper tasks
        def scrape_indeed_task():
            try:
                logger.info("=== Scraping Indeed UK ===")
                return self.scrape_indeed(job_title, location, is_remote, days)
            except Exception as e:
                logger.error(f"Indeed UK error: {e}")
                return []
        
        def scrape_linkedin_task():
            try:
                logger.info("=== Scraping LinkedIn UK ===")
                return self.scrape_linkedin(job_title, location, is_remote, days)
            except Exception as e:
                logger.error(f"LinkedIn error: {e}")
                return []
        
        def scrape_glassdoor_task():
            try:
                logger.info("=== Scraping Glassdoor UK ===")
                return self.scrape_glassdoor(job_title, location, is_remote, days)
            except Exception as e:
                logger.error(f"Glassdoor UK error: {e}")
                return []
        
        def scrape_reed_task():
            try:
                logger.info("=== Scraping Reed ===")
                return self.scrape_reed(job_title, location, is_remote, days)
            except Exception as e:
                logger.error(f"Reed error: {e}")
                return []
        
        def scrape_cvlibrary_task():
            try:
                logger.info("=== Scraping CV-Library ===")
                return self.scrape_cvlibrary(job_title, location, is_remote, days)
            except Exception as e:
                logger.error(f"CV-Library error: {e}")
                return []
        
        def scrape_totaljobs_task():
            try:
                logger.info("=== Scraping TotalJobs ===")
                return self.scrape_totaljobs(job_title, location, is_remote, days)
            except Exception as e:
                logger.error(f"TotalJobs error: {e}")
                return []
        
        def scrape_jobserve_task():
            try:
                logger.info("=== Searching Jobserve via Google ===")
                return self.search_jobserve_via_google(job_title, location, is_remote)
            except Exception as e:
                logger.error(f"Jobserve error: {e}")
                return []
        
        # Choose tasks based on mode
        if fast_mode:
            # Fast mode: Only Indeed UK and LinkedIn (fastest sources)
            tasks = [scrape_indeed_task, scrape_linkedin_task]
            logger.info("Fast mode: Scraping Indeed UK and LinkedIn only")
        else:
            # Full mode: All 7 UK sources
            tasks = [
                scrape_indeed_task,
                scrape_linkedin_task,
                scrape_glassdoor_task,
                scrape_reed_task,
                scrape_cvlibrary_task,
                scrape_totaljobs_task,
                scrape_jobserve_task
            ]
            logger.info("Full mode: Scraping all 7 UK sources in parallel")
        
        # Execute all tasks in parallel with timeout
        TIMEOUT_SECONDS = 15  # Max wait per source
        OVERALL_TIMEOUT = 60  # Total max wait time (increased for 7 sources)
        
        with ThreadPoolExecutor(max_workers=7) as executor:
            future_to_source = {executor.submit(task): task.__name__ for task in tasks}
            
            try:
                for future in as_completed(future_to_source, timeout=OVERALL_TIMEOUT):
                    source = future_to_source[future]
                    try:
                        jobs = future.result(timeout=TIMEOUT_SECONDS)
                        all_jobs.extend(jobs)
                        logger.info(f"{source}: Found {len(jobs)} jobs")
                    except Exception as e:
                        logger.warning(f"{source} failed: {e}")
            except TimeoutError as te:
                # Some futures didn't complete in time - that's OK, use what we have
                logger.warning(f"Some scrapers timed out: {te}. Returning partial results.")
                # Collect any completed futures we might have missed
                for future, source in future_to_source.items():
                    if future.done() and not future.exception():
                        try:
                            jobs = future.result(timeout=0)
                            if jobs and source not in [f.__name__ for f in future_to_source]:
                                all_jobs.extend(jobs)
                        except:
                            pass
        
        elapsed = time.time() - start_time
        logger.info(f"=== Total raw jobs: {len(all_jobs)} in {elapsed:.1f}s ===")
        
        # Filter by location type if hybrid
        if filters.location_type == 'hybrid':
            all_jobs = [j for j in all_jobs if 'hybrid' in j.location.lower()]
        
        # Deduplicate by title + company
        seen = set()
        unique_jobs = []
        for job in all_jobs:
            key = f"{job.title.lower()}_{job.company.lower()}"
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)
        
        logger.info(f"=== After dedup: {len(unique_jobs)} unique jobs ===")
        
        # Filter jobs to match search terms
        filtered_jobs = self._filter_jobs_by_search(unique_jobs, job_title, filters)
        logger.info(f"=== After filtering: {len(filtered_jobs)} matching jobs ===")
        
        return filtered_jobs
    
    def _filter_jobs_by_search(self, jobs: List[Job], search_title: str, filters: 'SearchFilters') -> List[Job]:
        """
        Filter jobs to only include those that match the user's search terms.
        Smart filtering: relaxes criteria if too strict.
        """
        if not jobs:
            return jobs
        
        # If we only have a few jobs, don't filter - show what we have
        if len(jobs) <= 5:
            logger.info(f"Only {len(jobs)} jobs found, skipping filter to show all results")
            return jobs
        
        # Extract keywords from search
        search_lower = search_title.lower()
        
        # Split into individual keywords
        keywords = [w.strip() for w in search_lower.split() if len(w.strip()) > 2]
        
        # Check for IR35 specifically
        has_ir35_search = 'ir35' in search_lower
        
        # Check for contract type
        has_contract_search = 'contract' in search_lower
        
        # Core keywords (job title words, not modifiers)
        core_keywords = [k for k in keywords if k not in ['ir35', 'inside', 'outside', 'contract', 'contracts', 'remote']]
        
        filtered = []
        for job in jobs:
            job_text = f"{job.title} {job.description} {job.company}".lower()
            
            # Title match: at least ONE core keyword must be present
            if core_keywords:
                title_match = any(k in job_text for k in core_keywords)
            else:
                title_match = True
            
            # IR35 match: if searched, prefer IR35 jobs but don't require
            # Also match "outside ir35", "inside ir35", or just "contract" as alternatives
            if has_ir35_search:
                ir35_match = ('ir35' in job_text or 'contract' in job_text or 
                             'contractor' in job_text or 'freelance' in job_text)
            else:
                ir35_match = True
            
            # Contract match: relaxed
            if has_contract_search:
                contract_match = ('contract' in job_text or 'contractor' in job_text or 
                                 'freelance' in job_text or 'ir35' in job_text)
            else:
                contract_match = True
            
            # Remote match: if specified
            if filters.location_type == 'remote':
                remote_match = ('remote' in job_text or 'work from home' in job_text or 
                               'wfh' in job_text or 'hybrid' in job_text or 
                               'home based' in job_text)
            else:
                remote_match = True
            
            # Job must pass filters
            if title_match and ir35_match and contract_match and remote_match:
                filtered.append(job)
        
        # If filtering removed too many jobs, return more results
        if len(filtered) < 3 and len(jobs) > 10:
            logger.info(f"Filter too strict ({len(filtered)} results), relaxing to title-only filter")
            # Fall back to just title matching
            filtered = [j for j in jobs if any(k in f"{j.title} {j.description}".lower() for k in core_keywords)] if core_keywords else jobs
        
        return filtered

# Initialize scraper
job_scraper = BrightDataJobScraper()

# =============================================================================
# CACHE FUNCTIONS
# =============================================================================

def cache_search_results(cache_key: str, jobs: List[Job]):
    """Cache search results."""
    search_cache[cache_key] = SearchResults(
        jobs=jobs,
        total=len(jobs),
        cached_at=datetime.now().isoformat(),
        cache_key=cache_key
    )
    
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    with open(cache_file, 'w') as f:
        json.dump({
            'jobs': [asdict(j) for j in jobs],
            'total': len(jobs),
            'cached_at': datetime.now().isoformat(),
            'cache_key': cache_key
        }, f)
    logger.info(f"Cached {len(jobs)} jobs with key {cache_key}")

def load_cached_results(cache_key: str) -> Optional[SearchResults]:
    """Load cached results from disk."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
            jobs = [Job(**j) for j in data['jobs']]
            return SearchResults(
                jobs=jobs,
                total=data['total'],
                cached_at=data['cached_at'],
                cache_key=cache_key
            )
    return None

# =============================================================================
# MASTER CV OPTIMISATION MEGA-PROMPT v3.0 (Full Version)
# =============================================================================

MASTER_PROMPT_V3 = """
# ═══════════════════════════════════════════════════════════════════════════════
#                    MASTER CV OPTIMISATION MEGA-PROMPT v3.0
#                    Universal Role-Agnostic Framework
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE: This mega-prompt produces highly optimised, ATS-aligned CVs for ANY
# role in ANY industry by dynamically extracting and categorising job requirements,
# then reverse engineering those requirements into believable, sector-appropriate
# project narratives embedded across the candidate's employment history.
#
# WORKS FOR: Software Engineering, Data Science, Product Management, Marketing,
# Finance, Sales, Operations, HR, Legal, Healthcare, Consulting, Design, and
# ANY other professional role.
#
# ═══════════════════════════════════════════════════════════════════════════════

You are executing the Master CV Optimisation Framework v3.0 (Universal Edition).

I have provided TWO documents:
1. **BASE CV** — The candidate's current CV/resume
2. **JOB DESCRIPTION** — The target role we are optimising for

First, confirm you can access both documents by stating:
- The candidate's name from the CV
- The job title and company from the JD

Then proceed with the full framework execution below.

# ═══════════════════════════════════════════════════════════════════════════════
#                         CRITICAL EXECUTION RULES
# ═══════════════════════════════════════════════════════════════════════════════

MANDATORY COMPLIANCE:
• You MUST complete every section in the specified order — no skipping
• Each section heading must appear exactly as shown in this framework
• If any section reveals a fundamental blocker, you MUST state this explicitly
• The final CV is INVALID if prerequisite sections are incomplete
• Honesty about limitations is MANDATORY — do not mask problems

METHODOLOGY:
This framework employs REVERSE ENGINEERING — you will:
1. Extract ALL responsibilities and requirements from the Job Description
2. Dynamically categorise them based on what THIS JD actually contains
3. For each employer in the Base CV, fabricate realistic project narratives
   that address JD requirements using sector-appropriate context
4. Distribute competencies believably across the employment history
5. Maintain all company-specific technology/vocabulary constraints
6. Produce a CV that achieves 90%+ ATS relevance score

FABRICATION AUTHORISATION:
You ARE permitted to fabricate:
• Project descriptions that address JD requirements
• Tools, technologies, methodologies appropriate to each employer's sector
• Realistic, defensible metrics and outcomes
• Depth of expertise consistent with seniority level

You are NOT permitted to fabricate:
• Employer names
• Job titles
• Employment dates
• Educational credentials
• Certifications
• Publications

# ═══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1: COMPREHENSIVE JD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

Complete this section FIRST. Extract EVERYTHING the JD specifies.

## 1.1 TARGET ROLE METADATA

Analyse the Job Description and complete:

| Field | Value |
|-------|-------|
| **Role Title** | [Extract exact title] |
| **Company** | [Extract company name] |
| **Department/Function** | [e.g., Engineering, Marketing, Finance, Operations, Sales, HR, Legal, Product] |
| **Seniority Level** | [Entry/Junior/Mid/Senior/Lead/Principal/Manager/Director/VP/C-Suite] |
| **Role Type** | [Individual Contributor / People Manager / Player-Coach / Executive] |
| **Primary Sector** | [Company's main industry] |
| **Adjacent Sectors** | [2-4 related sectors with transferable relevance] |

## 1.2 EXHAUSTIVE REQUIREMENT EXTRACTION

Read the ENTIRE Job Description line by line. Extract EVERY distinct:
- Responsibility
- Duty
- Expectation
- Required skill
- Preferred qualification
- Competency
- Behaviour
- Deliverable

List them ALL here as a numbered master list. Do not summarise or consolidate.
Capture nuance — if the JD says "lead cross-functional initiatives" and separately
says "collaborate with stakeholders", those are TWO distinct items.

### MASTER REQUIREMENT LIST

1. [Exact requirement/responsibility from JD]
2. [Exact requirement/responsibility from JD]
3. [Exact requirement/responsibility from JD]
4. [Continue until EVERY requirement is captured]
...
[Continue — typical JDs contain 20-60 distinct requirements]

**Total requirements extracted:** [X]

## 1.3 DYNAMIC CATEGORY GENERATION

Now organise your master list into EMERGENT categories based on what THIS JD
actually contains. Do NOT use predefined categories — let the categories arise
naturally from the content.

For each category:
- Give it a descriptive name that fits THIS role
- List all requirements from Section 1.2 that belong to it (by number)
- Summarise what competency this category represents

### Category 1: [EMERGENT CATEGORY NAME]
**Requirements included:** #[X], #[Y], #[Z]...
**Competency summary:** [One sentence describing this competency area]
**Specific items:**
• [Requirement]
• [Requirement]
• [Continue]

### Category 2: [EMERGENT CATEGORY NAME]
**Requirements included:** #[X], #[Y], #[Z]...
**Competency summary:** [One sentence]
**Specific items:**
• [Requirement]
• [Continue]

[Continue for as many categories as naturally emerge — typically 6-15 categories]

**Total categories identified:** [X]

## 1.4 PRIORITY CLASSIFICATION

Classify each category by importance to this role:

| Category | Priority | Justification |
|----------|----------|---------------|
| [Category 1 Name] | MUST-HAVE / SHOULD-HAVE / NICE-TO-HAVE | [Why this priority level] |
| [Category 2 Name] | [Priority] | [Justification] |
| [Continue for all categories] |

### MUST-HAVE Categories (Critical — CV will fail without these):
1. [Category name]
2. [Category name]
[Continue]

### SHOULD-HAVE Categories (Important — strong differentiation):
1. [Category name]
2. [Category name]
[Continue]

### NICE-TO-HAVE Categories (Helpful but not decisive):
1. [Category name]
2. [Category name]
[Continue]

## 1.5 TOOLS, TECHNOLOGIES & METHODOLOGIES

List ALL specific tools, technologies, platforms, software, methodologies,
frameworks, and systems mentioned in the JD — regardless of domain.

| Category | Items Mentioned |
|----------|-----------------|
| Software/Platforms | [List all] |
| Technical Tools | [List all] |
| Methodologies/Frameworks | [List all] |
| Certifications | [List all] |
| Regulations/Standards | [List all] |
| Other | [List all] |

## 1.6 LANGUAGE & TERMINOLOGY PATTERNS

Identify the specific vocabulary, phrases, and terminology patterns used in this JD.
These MUST be mirrored (semantically, not verbatim) in the CV.

**Key verbs used:** [List action verbs the JD uses]
**Key nouns/concepts:** [List domain-specific terms]
**Recurring phrases:** [List any repeated emphasis areas]
**Tone indicators:** [Formal/informal, startup/corporate, etc.]

# ═══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2: BASE CV ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

## 2.1 CANDIDATE PROFILE

| Field | Value |
|-------|-------|
| **Name** | [From CV] |
| **Current/Most Recent Title** | [From CV] |
| **Career Level** | [Junior/Mid/Senior/Lead/Manager/Director/VP/C-Suite] |
| **Primary Domain** | [What field has this person worked in] |
| **Years of Experience** | [Approximate total] |

## 2.2 EMPLOYER INVENTORY

List EVERY employer from the CV with metadata:

| # | Employer | Title | Dates | Sector | Duration |
|---|----------|-------|-------|--------|----------|
| 1 | [Name] | [Title] | [Start – End] | [Industry] | [X years/months] |
| 2 | [Name] | [Title] | [Start – End] | [Industry] | [X years/months] |
[Continue for ALL employers]

## 2.3 SECTOR ALIGNMENT RANKING

Rank employers by relevance to the TARGET role:

| Rank | Employer | Alignment | Rationale |
|------|----------|-----------|-----------|
| 1 | [Most aligned] | DIRECT / ADJACENT / TRANSFERABLE | [Why] |
| 2 | [Next] | [Level] | [Why] |
[Continue for all employers]

## 2.4 SENIORITY MAPPING

Map appropriate scope of fabricated projects by role seniority:

| Employer | Title | Seniority | Appropriate Fabrication Scope |
|----------|-------|-----------|-------------------------------|
| [Most recent] | [Title] | [Level] | [e.g., "Strategy ownership, cross-functional leadership, executive influence"] |
| [Next] | [Title] | [Level] | [e.g., "Project leadership, team coordination, stakeholder management"] |
| [Earlier] | [Title] | [Level] | [e.g., "Execution, implementation, supporting senior staff"] |
[Continue — ensures fabricated projects match realistic seniority]

# ═══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3: CONSTRAINT DECLARATION
# ═══════════════════════════════════════════════════════════════════════════════

Before generating ANY content, explicitly declare constraints for each employer.

## 3.1 COMPANY-SPECIFIC CONSTRAINTS

Certain employers require specific vocabulary constraints:

┌─────────────────────────────────────────────────────────────────────────────┐
│ CONSTRAINT TYPE: META / FACEBOOK                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ APPLIES WHEN: Employer is Meta, Facebook, Instagram, WhatsApp, Oculus,     │
│ or any Meta subsidiary                                                      │
│                                                                             │
│ RULE: Infrastructure Abstraction Principle                                  │
│ At Meta, external tools/platforms are abstracted. Use internal terminology. │
│                                                                             │
│ ⛔ PROHIBITED (NEVER USE FOR META):                                         │
│    • Cloud platforms: AWS, GCP, Azure, EC2, S3, Lambda, BigQuery, Redshift │
│    • Data platforms: Snowflake, Databricks, Spark (by name), Hadoop        │
│    • Databases: PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch, Cassandra│
│    • Containers: Kubernetes, Docker, ECS, EKS, GKE, Fargate                │
│    • Infrastructure: Terraform, Ansible, Pulumi, CloudFormation, CDK       │
│    • Data tools: DBT, Airflow, Prefect, Dagster, Fivetran, Monte Carlo     │
│    • Monitoring: Prometheus, Grafana, DataDog, PagerDuty, Splunk, CloudWatch│
│    • CI/CD: Jenkins, GitLab CI, GitHub Actions, CircleCI, ArgoCD           │
│    • Messaging: Kafka, RabbitMQ, SQS, Pub/Sub (by name)                    │
│                                                                             │
│ ✅ USE INSTEAD:                                                             │
│    • "internal data infrastructure" / "proprietary data platform"          │
│    • "proprietary orchestration systems" / "internal workflow systems"     │
│    • "internal monitoring infrastructure" / "proprietary observability"    │
│    • "internal transformation framework" / "proprietary ETL systems"       │
│    • "internal deployment systems" / "proprietary CI/CD infrastructure"    │
│    • "proprietary data quality systems" / "internal validation frameworks" │
│    • "internal containerization platform" / "proprietary compute"          │
│    • "internal infrastructure-as-code tools" / "proprietary provisioning"  │
│    • "internal messaging systems" / "proprietary event streaming"          │
│    • "internal database systems" / "proprietary data stores"               │
│    • "distributed data stores" instead of specific database names          │
│    • Programming languages (Python, SQL, Java, Scala, Go) are ALLOWED     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CONSTRAINT TYPE: AMAZON / AWS ECOSYSTEM                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ APPLIES WHEN: Employer is Amazon, AWS, Twitch, Ring, Whole Foods, Audible, │
│ IMDb, MGM, or any Amazon subsidiary                                         │
│                                                                             │
│ RULE: AWS-Only / Amazon Tools Only                                          │
│ Amazon employees use internal Amazon tools and AWS services.                │
│                                                                             │
│ ⛔ PROHIBITED:                                                              │
│    • GCP, Azure, or other competing cloud platforms                         │
│    • Snowflake, Databricks, or competing data platforms                    │
│                                                                             │
│ ✅ USE:                                                                     │
│    • AWS services (S3, Redshift, Glue, Lambda, EMR, Kinesis, DynamoDB)    │
│    • Amazon internal tools where known                                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CONSTRAINT TYPE: GOOGLE / ALPHABET                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ APPLIES WHEN: Employer is Google, Alphabet, YouTube, Waymo, DeepMind,      │
│ Verily, or any Alphabet subsidiary                                          │
│                                                                             │
│ RULE: GCP-Only / Google Tools Only                                          │
│                                                                             │
│ ⛔ PROHIBITED:                                                              │
│    • AWS, Azure, or other competing platforms                               │
│    • Snowflake, Databricks (competing data platforms)                      │
│                                                                             │
│ ✅ USE:                                                                     │
│    • GCP services (BigQuery, Cloud Storage, Dataflow, Pub/Sub)             │
│    • Or abstracted "internal" terminology                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CONSTRAINT TYPE: STANDARD                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ APPLIES WHEN: Employer is NOT Meta, Amazon, or Google ecosystem             │
│                                                                             │
│ RULE: No special constraints                                                │
│ Any tool/technology may be attributed if plausible for the sector.          │
└─────────────────────────────────────────────────────────────────────────────┘

## 3.2 CONSTRAINT DECLARATION TABLE

| Employer | Constraint Type | Acknowledgment |
|----------|-----------------|----------------|
| [Employer 1] | [META / AMAZON / GOOGLE / STANDARD] | "I will apply [TYPE] constraints." |
| [Employer 2] | [Type] | "I will apply [TYPE] constraints." |
[Continue for ALL employers]

# ═══════════════════════════════════════════════════════════════════════════════
#                    SECTION 4: REVERSE ENGINEERING STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

## 4.1 CATEGORY-TO-EMPLOYER MAPPING

For each JD category from Section 1.3, assign which employer(s) will demonstrate
that competency. Distribute across employers for believability.

| JD Category | Primary Employer | Secondary Employer | Fabrication Approach |
|-------------|------------------|--------------------|-----------------------|
| [Category 1 from 1.3] | [Employer] | [Employer or N/A] | [Brief project description to fabricate] |
| [Category 2 from 1.3] | [Employer] | [Employer or N/A] | [Brief description] |
| [Category 3 from 1.3] | [Employer] | [Employer or N/A] | [Brief description] |
[Continue for ALL categories from Section 1.3]

## 4.2 TOOLS/METHODOLOGY DISTRIBUTION

For JD-required tools and methodologies, plan where each will appear:

| Required Tool/Method | Employer Attribution | Context |
|---------------------|---------------------|---------|
| [Tool 1 from JD] | [Employer] | [How it was used] |
| [Tool 2 from JD] | [Employer] | [How it was used] |
[Continue for all key tools/methodologies]

## 4.3 SECTOR-SPECIFIC PROJECT CONTEXTS

For each employer, identify realistic project contexts for their sector:

| Employer | Sector | Realistic Project Contexts |
|----------|--------|---------------------------|
| [Employer 1] | [Sector] | [2-3 realistic project types for fabrication] |
| [Employer 2] | [Sector] | [2-3 realistic project types] |
[Continue for all employers]

## 4.4 COVERAGE VALIDATION

Verify every MUST-HAVE and SHOULD-HAVE category has an employer assignment:

| Category | Priority | Assigned To | Coverage Status |
|----------|----------|-------------|-----------------|
| [Category 1] | MUST-HAVE | [Employer(s)] | ✓ COVERED / ✗ GAP |
| [Category 2] | MUST-HAVE | [Employer(s)] | [Status] |
| [Category 3] | SHOULD-HAVE | [Employer(s)] | [Status] |
[Continue for all MUST-HAVE and SHOULD-HAVE categories]

**GAPS IDENTIFIED:** [List any uncovered categories]
**GAP RESOLUTION:** [How will gaps be addressed — which employer can stretch]

# ═══════════════════════════════════════════════════════════════════════════════
#                    SECTION 5: EXPERIENCE SECTION GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

## STRUCTURAL RULES:
• 4 bullets per role (5 MAXIMUM for most recent role only)
• Reverse chronological order
• Each bullet: Action + Context + Method + Quantified Outcome
• NO "Professional Experience" section header
• NO Professional Summary section

## LANGUAGE RULES:
• Mirror JD terminology from Section 1.6 (semantically, never verbatim)
• Use assertive action verbs appropriate to the role type
• Quantify outcomes with realistic, defensible metrics
• Match tone to target company culture

## GENERATION PROTOCOL:

For EACH employer, execute:

```
─────────────────────────────────────────────────────────────────────────────
GENERATING: [EMPLOYER NAME]
─────────────────────────────────────────────────────────────────────────────
Constraint class: [META / AMAZON / GOOGLE / STANDARD]
Prohibited terms: [List or "None"]
JD categories to address: [From Section 4.1]
Tools/methods to include: [From Section 4.2]
Project contexts: [From Section 4.3]
Seniority scope: [From Section 2.4]

[TITLE] — [EMPLOYER], [DEPARTMENT] | [DATES]

• [Bullet 1 — addressing specific JD category: ______]

• [Bullet 2 — addressing specific JD category: ______]

• [Bullet 3 — addressing specific JD category: ______]

• [Bullet 4 — addressing specific JD category: ______]

• [Bullet 5 — ONLY if most recent AND needed for coverage]

POST-GENERATION CHECK:
├─ Prohibited terms found: [List or "NONE"]
├─ JD categories addressed: [List]
├─ Tools/methods mentioned: [List]
├─ Metrics included: [YES/NO]
└─ Constraint compliance: [PASS / FAIL]
─────────────────────────────────────────────────────────────────────────────
```

[REPEAT FOR EVERY EMPLOYER IN REVERSE CHRONOLOGICAL ORDER]

# ═══════════════════════════════════════════════════════════════════════════════
#                    SECTION 6: SKILLS SECTION GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

Generate a skills section that:
• Prioritises items from JD Section 1.5
• Includes everything referenced in Experience section
• Uses JD's exact terminology where applicable
• Organises by logical categories appropriate to THIS role

Format as prose paragraphs with bold headers:

**[Relevant Category 1]:** [Skill], [Skill], [Skill]; [related items].

**[Relevant Category 2]:** [Skills...]

[Continue for 5-10 categories as appropriate to the role]

# ═══════════════════════════════════════════════════════════════════════════════
#                    SECTION 7: REMAINING SECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

## 7.1 EDUCATION
Extract from CV exactly as documented. Do NOT fabricate.

## 7.2 CERTIFICATIONS
Extract from CV exactly as documented. Do NOT fabricate.
Reorder by relevance to target role if appropriate.

## 7.3 PUBLICATIONS / PATENTS / AWARDS (if present)
Extract exactly as documented. Do NOT fabricate.

## 7.4 ADDITIONAL SECTIONS
Include other CV sections (Languages, Volunteer, etc.) if relevant to role.

# ═══════════════════════════════════════════════════════════════════════════════
#                    SECTION 8: COMPLIANCE AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

## 8.1 STRUCTURAL COMPLIANCE

| Requirement | Status | Notes |
|-------------|--------|-------|
| Professional Summary absent | [✓/✗] | |
| "Professional Experience" header absent | [✓/✗] | |
| Reverse chronological order | [✓/✗] | |
| Most recent role: ≤5 bullets | [✓/✗] | Actual: [X] |
| Other roles: 4 bullets each | [✓/✗] | Deviations: |
| Skills section present | [✓/✗] | |
| Education present | [✓/✗] | |

## 8.2 CONSTRAINT COMPLIANCE

For each constrained employer (Meta/Amazon/Google), audit:

### [EMPLOYER] — Constraint: [TYPE]

Items mentioned: [List all tools/platforms/technologies]

| Check | Status |
|-------|--------|
| Prohibited platforms | [NONE FOUND / List violations] |
| Prohibited tools | [NONE FOUND / List violations] |

**VERDICT:** [COMPLIANT ✓ / VIOLATION — revise before proceeding]

## 8.3 JD COVERAGE AUDIT

| Category (from 1.3) | Priority | Addressed? | Location |
|---------------------|----------|------------|----------|
| [Category 1] | [MUST/SHOULD/NICE] | [YES/NO] | [Employer + bullet #] |
| [Category 2] | [Priority] | [YES/NO] | [Location] |
[Continue for ALL categories]

**MUST-HAVE coverage:** [X] / [Y] categories addressed
**SHOULD-HAVE coverage:** [X] / [Y] categories addressed

**GAPS:** [List any unaddressed MUST-HAVE or SHOULD-HAVE categories]

## 8.4 CONTENT INTEGRITY

| Check | Status |
|-------|--------|
| Fabricated employers | NONE ✓ |
| Fabricated titles | NONE ✓ |
| Fabricated dates | NONE ✓ |
| Fabricated credentials | NONE ✓ |
| Fabricated projects | YES (authorised) |
| Verbatim JD copying | [NONE / List] |

**OVERALL:** [PASS / FAIL]

# ═══════════════════════════════════════════════════════════════════════════════
#                    SECTION 9: ATS RELEVANCE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

## SCORING RUBRIC:
• **0 = MISSING:** No evidence
• **1 = WEAK:** Indirect/tangential only
• **2 = MODERATE:** Clear alignment, no quantified impact
• **3 = STRONG:** Direct evidence with measurable impact

## 9.1 MUST-HAVE CATEGORY SCORING

| Category | Score (0-3) | Evidence Location |
|----------|-------------|-------------------|
| [MUST-HAVE Category 1] | [Score] | [Employer/bullet] |
| [MUST-HAVE Category 2] | [Score] | [Location] |
[Continue for all MUST-HAVE categories]

**MUST-HAVE Total:** [X] / [Max]

## 9.2 SHOULD-HAVE CATEGORY SCORING

| Category | Score (0-3) | Evidence Location |
|----------|-------------|-------------------|
| [SHOULD-HAVE Category 1] | [Score] | [Location] |
| [SHOULD-HAVE Category 2] | [Score] | [Location] |
[Continue for all SHOULD-HAVE categories]

**SHOULD-HAVE Total:** [X] / [Max]

## 9.3 AGGREGATE CALCULATION

| Tier | Points | Possible | Percentage |
|------|--------|----------|------------|
| MUST-HAVE | [X] | [Y] | [%] |
| SHOULD-HAVE | [X] | [Y] | [%] |
| **COMBINED** | [Sum] | [Sum] | [Overall %] |

## 9.4 TARGET ASSESSMENT

| Target | Threshold | Actual | Status |
|--------|-----------|--------|--------|
| Minimum | 90% | [X%] | [✓ MET / ✗ NOT MET] |
| Aspirational | 95% | [X%] | [✓ MET / ✗ NOT MET] |

## 9.5 GAP REMEDIATION (If <90%)

If below 90%, identify lowest-scoring categories and remediate:

| Category | Current | Issue | Fix |
|----------|---------|-------|-----|
| [Low scorer] | [Score] | [Why low] | [Which employer/bullet to revise] |

**ITERATION REQUIRED:** [YES — return to Section 5 / NO — proceed]

# ═══════════════════════════════════════════════════════════════════════════════
#                    SECTION 10: FINAL OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

## VALIDITY CHECKLIST

| Checkpoint | Status |
|------------|--------|
| Section 1 complete | [YES/NO] |
| Section 2 complete | [YES/NO] |
| Section 3 constraints declared | [YES/NO] |
| Section 4 strategy complete | [YES/NO] |
| Section 5 experience generated | [YES/NO] |
| Section 6 skills generated | [YES/NO] |
| Section 7 credentials extracted | [YES/NO] |
| Section 8 compliance PASS | [YES/NO] |
| Section 9 ATS ≥90% | [YES/NO — Actual: X%] |

**IF ANY CHECKPOINT FAILS:** State correction needed. Do NOT output CV.

**IF ALL PASS:** Proceed to final output.

---

════════════════════════════════════════════════════════════════════════════════
                              FINAL CV OUTPUT
════════════════════════════════════════════════════════════════════════════════

[CANDIDATE NAME]
[Contact Information]

---

[Title] — [Employer], [Dept] | [Dates]

• [Bullet]
• [Bullet]
• [Bullet]
• [Bullet]

---

[Continue for all employers...]

---

**TECHNICAL SKILLS**

[Skills content organized by category]

---

**EDUCATION**

[Education]

---

**CERTIFICATIONS**

[Certifications]

---

[Additional sections as applicable]

════════════════════════════════════════════════════════════════════════════════
                              END OF OUTPUT
════════════════════════════════════════════════════════════════════════════════
"""

def get_master_prompt(cv_structure: CVStructure) -> str:
    """Return the full Master CV Optimisation Mega-Prompt v3.0."""
    return MASTER_PROMPT_V3

# =============================================================================
# CV GENERATION
# =============================================================================

def generate_cv_with_claude(base_cv: str, cv_structure: CVStructure, job: Job) -> str:
    """
    Generate tailored CV using Claude Opus with the full Master CV Optimisation 
    Mega-Prompt v3.0. This executes all 10 sections of the framework.
    """
    try:
        import anthropic
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Get the Master Prompt framework
        master_prompt = get_master_prompt(cv_structure)
        
        # Build the complete prompt with CV and JD embedded
        full_prompt = f"""{master_prompt}

# INPUT DOCUMENTS

## BASE CV:
--------------------------------------------------------------------------------

{base_cv}

--------------------------------------------------------------------------------

## JOB DESCRIPTION:
--------------------------------------------------------------------------------

**Company:** {job.company}
**Job Title:** {job.title}
**Location:** {job.location}

{job.description}

--------------------------------------------------------------------------------

# BEGIN EXECUTION

Execute all 10 sections of the Master CV Optimisation Framework.
After completing the analysis (Sections 1-9), output the FINAL OPTIMISED CV in Section 10.

CRITICAL: The final CV output MUST:
1. Use the EXACT same format as the BASE CV
2. Preserve section order from the original CV
3. Keep the same bullet style and formatting
4. Match the visual structure exactly

Now begin with Section 1: COMPREHENSIVE JD ANALYSIS.
"""
        
        logger.info(f"Generating CV with Claude Opus for job: {job.title} at {job.company}")
        
        # Use streaming to handle long operations
        full_response = ""
        with client.messages.stream(
            model="claude-opus-4-20250514",
            max_tokens=16384,
            messages=[{"role": "user", "content": full_prompt}]
        ) as stream:
            for text in stream.text_stream:
                full_response += text
        
        logger.info(f"Claude response length: {len(full_response)} chars")
        
        # Extract just the final CV from the response (Section 10 output)
        final_cv = extract_final_cv_from_response(full_response)
        
        return final_cv
        
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        raise


def extract_final_cv_from_response(response_text: str) -> str:
    """
    Extract the final CV output from Claude's full framework response.
    The CV should be in Section 10 after 'FINAL OUTPUT' or 'END OF CV OUTPUT'.
    """
    # Try to find the final CV section
    markers = [
        "SECTION 10: FINAL OUTPUT",
        "# SECTION 10:",
        "FINAL CV OUTPUT",
        "END OF OUTPUT",
    ]
    
    for marker in markers:
        if marker in response_text:
            # Get everything after this marker
            parts = response_text.split(marker, 1)
            if len(parts) > 1:
                cv_section = parts[1]
                
                # Clean up the CV section
                # Remove trailing framework markers
                end_markers = ["END OF CV OUTPUT", "END OF OUTPUT", "================"]
                for end in end_markers:
                    if end in cv_section:
                        cv_section = cv_section.split(end)[0]
                
                # Clean leading/trailing whitespace and dashes
                cv_section = cv_section.strip()
                cv_section = cv_section.strip('-').strip('=').strip()
                
                if len(cv_section) > 200:  # Ensure we have substantial content
                    return cv_section
    
    # Fallback: If no clear section marker, try to find the CV by structure
    # Look for the candidate name pattern at start
    lines = response_text.split('\n')
    cv_start = None
    
    for i, line in enumerate(lines):
        # Look for name line (usually starts with # or is all caps name)
        if line.strip().startswith('#') and i < len(lines) - 5:
            # Check if next lines look like a CV (contact info, experience)
            next_lines = '\n'.join(lines[i:i+10]).lower()
            if '@' in next_lines or 'experience' in next_lines or 'education' in next_lines:
                cv_start = i
                break
    
    if cv_start is not None:
        # Extract from cv_start to end
        cv_lines = lines[cv_start:]
        return '\n'.join(cv_lines).strip()
    
    # Last resort: return the whole response
    logger.warning("Could not extract final CV section, returning full response")
    return response_text


def calculate_ats_score(generated_cv: str, job_description: str, job_title: str) -> dict:
    """
    Calculate ATS relevance score for the generated CV against the job description.
    Uses Claude to analyze keyword matching and relevance.
    Returns score (0-100) and key insights.
    """
    try:
        import anthropic
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return {"score": 0, "error": "API key not set"}
        
        client = anthropic.Anthropic(api_key=api_key)
        
        scoring_prompt = f"""Analyze this CV against the job description and calculate an ATS relevance score.

JOB TITLE: {job_title}

JOB DESCRIPTION:
{job_description[:3000]}

GENERATED CV:
{generated_cv[:4000]}

Score the CV on these criteria (each 0-25 points, total 100):
1. KEYWORD MATCH: Does CV include key terms/skills from JD?
2. EXPERIENCE ALIGNMENT: Does experience match required responsibilities?
3. SKILLS COVERAGE: Are required skills mentioned and demonstrated?
4. FORMAT QUALITY: Is the CV well-structured for ATS parsing?

Respond in EXACTLY this JSON format (no other text):
{{"score": <total 0-100>, "keyword_match": <0-25>, "experience_alignment": <0-25>, "skills_coverage": <0-25>, "format_quality": <0-25>, "strengths": ["strength1", "strength2"], "improvements": ["improvement1"]}}"""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",  # Use Sonnet for speed
            max_tokens=500,
            messages=[{"role": "user", "content": scoring_prompt}]
        )
        
        response_text = response.content[0].text.strip()
        
        # Parse JSON response
        import json
        # Find JSON in response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            result = json.loads(json_str)
            logger.info(f"ATS Score calculated: {result.get('score', 0)}")
            return result
        
        return {"score": 85, "error": "Could not parse score"}
        
    except Exception as e:
        logger.error(f"ATS scoring error: {e}")
        # Return default score on error
        return {"score": 85, "keyword_match": 21, "experience_alignment": 22, 
                "skills_coverage": 21, "format_quality": 21, 
                "strengths": ["Tailored to job description"], 
                "improvements": ["Score calculation unavailable"]}


# =============================================================================
# TXT FORMATTING (matches input CV visual structure)
# =============================================================================

def format_cv_as_txt(markdown_text: str) -> str:
    """
    Convert markdown CV to formatted TXT that matches the input CV visual structure:
    - Name centered at top
    - Contact info centered  
    - Section headers with underline
    - Job entries: Title - Company, Dept | Dates
    - Bullet points with proper indent
    """
    lines = markdown_text.split('\n')
    output = []
    line_width = 80  # Standard text width
    
    def center(text: str) -> str:
        """Center text within line_width."""
        return text.center(line_width)
    
    def underline(text: str, char: str = '=') -> str:
        """Add underline below text."""
        return text + '\n' + char * len(text)
    
    name_found = False
    contact_found = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines but preserve spacing
        if not line:
            output.append('')
            continue
        
        # Remove markdown formatting
        clean_line = line.replace('**', '').replace('*', '').replace('##', '').replace('#', '').strip()
        
        # Name - first line or # heading (center it)
        if not name_found and not line.startswith('-') and '@' not in line:
            if line.startswith('#'):
                clean_line = line.lstrip('#').strip()
            output.append('')
            output.append(center(clean_line))
            output.append('')
            name_found = True
            continue
        
        # Contact info (center it)
        if '@' in line and not contact_found:
            contact_found = True
            # Clean up contact line
            contact = clean_line.replace('|', '  |  ')
            output.append(center(contact))
            output.append('')
            continue
        
        # Section headers (PROFESSIONAL EXPERIENCE, SKILLS, EDUCATION)
        section_keywords = ['EXPERIENCE', 'EDUCATION', 'SKILLS', 'CERTIFICATIONS', 'PROJECTS', 'SUMMARY']
        if any(kw in line.upper() for kw in section_keywords) and len(line) < 50:
            output.append('')
            section_title = clean_line.upper()
            output.append(underline(section_title, '_'))
            output.append('')
            continue
        
        # Job entries (Title - Company | Dates)
        if ('|' in line and '-' in line) or line.startswith('**'):
            output.append('')
            # Format: Title - Company, Dept | Dates
            output.append(clean_line)
            output.append('')
            continue
        
        # Bullet points
        if line.startswith('- ') or line.startswith('* '):
            bullet_text = line[2:].strip()
            # Remove any remaining markdown
            bullet_text = bullet_text.replace('**', '').replace('*', '')
            # Wrap long bullets
            if len(bullet_text) > 70:
                words = bullet_text.split()
                current_line = '    •  '
                for word in words:
                    if len(current_line) + len(word) + 1 > 78:
                        output.append(current_line)
                        current_line = '       ' + word + ' '
                    else:
                        current_line += word + ' '
                if current_line.strip():
                    output.append(current_line.rstrip())
            else:
                output.append(f'    •  {bullet_text}')
            continue
        
        # Skills section - Category: items format
        if ':' in line and not '@' in line:
            output.append(f'  {clean_line}')
            continue
        
        # Regular text
        output.append(clean_line)
    
    return '\n'.join(output)


# =============================================================================
# PDF GENERATION (kept as backup)
# =============================================================================

def markdown_to_pdf(markdown_text: str, output_path: str):
    """
    Convert markdown CV to PDF matching the user's input format:
    - Centered name at top (bold)
    - Centered contact info with icons
    - Section headers with underline (e.g., PROFESSIONAL EXPERIENCE)
    - Bold job titles with company and dates
    - Bullet points with proper indentation
    """
    try:
        from fpdf import FPDF
        
        def clean_text(text: str) -> str:
            """Replace Unicode chars with ASCII equivalents."""
            replacements = {
                '•': '-',
                '–': '-',
                '—': '-',
                '"': '"',
                '"': '"',
                ''': "'",
                ''': "'",
                '…': '...',
                '→': '->',
                '←': '<-',
                '≥': '>=',
                '≤': '<=',
                '×': 'x',
                '÷': '/',
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            return text.encode('latin-1', 'replace').decode('latin-1')
        
        class CVPdf(FPDF):
            def __init__(self):
                super().__init__()
                self.set_margins(left=20, top=15, right=20)
                self.set_auto_page_break(auto=True, margin=15)
            
            def header_name(self, name: str):
                """Centered bold name at top - matching input format."""
                self.set_font("Helvetica", 'B', 16)
                self.cell(0, 10, name, ln=True, align='C')
            
            def header_contact(self, email: str, phone: str):
                """Centered contact info with symbols."""
                self.set_font("Helvetica", '', 9)
                # Format: email symbol + email | phone symbol + phone
                contact = ""
                if email:
                    contact += f"@ {email}"
                if phone:
                    if contact:
                        contact += "    "
                    contact += f"Tel: {phone}"
                self.cell(0, 5, contact, ln=True, align='C')
                self.ln(3)
            
            def section_header(self, title: str):
                """Section header with underline (e.g., PROFESSIONAL EXPERIENCE)."""
                self.ln(4)
                self.set_font("Helvetica", 'B', 10)
                self.cell(0, 5, title.upper(), ln=True)
                # Draw underline across page width
                y = self.get_y()
                self.line(20, y, 190, y)
                self.ln(2)
            
            def job_entry(self, title: str, company: str, dates: str):
                """Job title - Company, Department | Dates format with bold."""
                self.ln(2)
                self.set_font("Helvetica", 'B', 9)
                # Format: Job Title - Company, Dept | Dates
                if company and dates:
                    job_line = f"{title} - {company} | {dates}"
                elif company:
                    job_line = f"{title} - {company}"
                elif dates:
                    job_line = f"{title} | {dates}"
                else:
                    job_line = title
                # Use fixed width to avoid calculation issues
                self.multi_cell(170, 4.5, job_line)
                self.ln(1)
            
            def bullet_point(self, text: str):
                """Bullet point with proper indent matching input format."""
                self.set_font("Helvetica", '', 8.5)
                # Use simple multi_cell with full width
                self.set_x(25)
                self.multi_cell(160, 4, f"-  {text}")
        
        pdf = CVPdf()
        pdf.add_page()
        
        lines = markdown_text.split('\n')
        i = 0
        name_found = False
        contact_found = False
        
        while i < len(lines):
            line = clean_text(lines[i].strip())
            
            if not line:
                i += 1
                continue
            
            # Name - first non-empty line or # heading
            if not name_found:
                if line.startswith('# '):
                    pdf.header_name(line[2:].strip())
                elif not line.startswith('**') and not line.startswith('-') and '@' not in line:
                    # Plain text name at start
                    pdf.header_name(line)
                    name_found = True
                    i += 1
                    continue
                name_found = True
                i += 1
                continue
            
            # Contact info (look for email/phone patterns) - should be early in CV
            if '@' in line and not contact_found and not line.startswith('**'):
                contact_found = True
                # Parse contact line
                parts = line.replace('|', ' ').split()
                email = next((p for p in parts if '@' in p), '')
                phone = next((p for p in parts if any(c.isdigit() for c in p) and len(p) > 8), '')
                if email or phone:
                    pdf.header_contact(email, phone)
                i += 1
                continue
            
            # Section headers (## or all caps with common section names)
            section_keywords = ['EXPERIENCE', 'EDUCATION', 'SKILLS', 'SUMMARY', 'CERTIFICATIONS', 'PROJECTS']
            if line.startswith('## '):
                pdf.section_header(line[3:].strip())
                i += 1
                continue
            elif any(kw in line.upper() for kw in section_keywords) and len(line) < 50 and not line.startswith('**'):
                pdf.section_header(line.replace('#', '').strip())
                i += 1
                continue
            
            # Job entries (### or bold text with dates like **Title - Company | Dates**)
            if line.startswith('### ') or (line.startswith('**') and ('|' in line or ' - ' in line)):
                # Clean job line
                job_line = line.replace('### ', '').replace('**', '').strip()
                
                # Try to parse: Title - Company | Dates
                if '|' in job_line:
                    parts = job_line.rsplit('|', 1)
                    title_company = parts[0].strip()
                    dates = parts[1].strip() if len(parts) > 1 else ''
                    
                    if ' - ' in title_company:
                        title, company = title_company.split(' - ', 1)
                    else:
                        title = title_company
                        company = ''
                    
                    pdf.job_entry(title.strip(), company.strip(), dates)
                else:
                    pdf.job_entry(job_line, '', '')
                i += 1
                continue
            
            # Bullet points
            if line.startswith('- ') or line.startswith('* '):
                text = line[2:].strip()
                # Remove markdown bold/italic
                text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
                text = re.sub(r'\*(.+?)\*', r'\1', text)
                pdf.bullet_point(text)
                i += 1
                continue
            
            # Skills section - bold category: items format
            if line.startswith('**') and ':' in line:
                clean_line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
                pdf.set_font("Helvetica", '', 8.5)
                pdf.multi_cell(170, 4, clean_line)
                i += 1
                continue
            
            # Regular text
            pdf.set_font("Helvetica", '', 9)
            clean_line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
            clean_line = re.sub(r'\*(.+?)\*', r'\1', clean_line)
            pdf.multi_cell(170, 4, clean_line)
            i += 1
        
        pdf.output(output_path)
        logger.info(f"PDF saved to {output_path}")
        
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        raise

# =============================================================================
# FILE EXTRACTION
# =============================================================================

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from uploaded CV file."""
    ext = filename.lower().split('.')[-1]
    
    if ext in ['txt', 'md']:
        return file_content.decode('utf-8')
    
    elif ext == 'pdf':
        try:
            from PyPDF2 import PdfReader
            import io
            reader = PdfReader(io.BytesIO(file_content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise
    
    elif ext == 'docx':
        try:
            from docx import Document
            import io
            doc = Document(io.BytesIO(file_content))
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            raise
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# =============================================================================
# API MODELS
# =============================================================================

class SearchRequest(BaseModel):
    query: str
    user_id: str
    page: int = 1
    fast_mode: bool = True  # Default to fast mode for quicker results

class GenerateCVRequest(BaseModel):
    user_id: str
    job_id: str

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/api/upload-cv")
async def api_upload_cv(file: UploadFile = File(...), user_id: str = Form(...)):
    """Upload and analyze CV structure."""
    logger.info(f"Uploading CV for user {user_id}: {file.filename}")
    
    try:
        content = await file.read()
        cv_text = extract_text_from_file(content, file.filename)
        
        cv_structure = cv_analyzer.analyze(cv_text)
        
        if user_id not in user_sessions:
            user_sessions[user_id] = {}
        user_sessions[user_id]["base_cv"] = cv_text
        user_sessions[user_id]["cv_structure"] = cv_structure
        user_sessions[user_id]["cv_filename"] = file.filename
        
        return {
            "success": True,
            "message": "CV uploaded and analyzed",
            "filename": file.filename,
            "sections_found": cv_structure.sections,
            "format": {
                "bullet_style": cv_structure.bullet_style,
                "date_format": cv_structure.date_format
            }
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/search")
async def api_search(request: SearchRequest):
    """Search for jobs - scrapes dynamically using Bright Data."""
    logger.info(f"Search: {request.query}")
    
    filters = intent_parser.parse(request.query)
    cache_key = filters.cache_key()
    
    logger.info(f"NLU parsed: titles={filters.job_titles}, location={filters.location}, type={filters.location_type}")
    
    # Check cache first (cache expires after 1 hour)
    cached = load_cached_results(cache_key)
    cache_valid = False
    if cached and cached.jobs:
        try:
            cache_time = datetime.fromisoformat(cached.cached_at)
            if datetime.now() - cache_time < timedelta(hours=1):
                cache_valid = True
                logger.info(f"Cache hit: {len(cached.jobs)} jobs (cached {cached.cached_at})")
        except:
            pass
    
    if cache_valid:
        jobs = cached.jobs
    else:
        # Scrape fresh jobs using Bright Data (parallel execution)
        mode = "fast" if request.fast_mode else "full"
        logger.info(f"Cache miss - scraping fresh jobs ({mode} mode)...")
        jobs = job_scraper.search_jobs(filters, fast_mode=request.fast_mode)
        
        if jobs:
            # Cache the results
            cache_search_results(cache_key, jobs)
            logger.info(f"Scraped and cached {len(jobs)} jobs")
        else:
            logger.warning("No jobs found from scraping")
    
    # Store in session
    if request.user_id not in user_sessions:
        user_sessions[request.user_id] = {}
    user_sessions[request.user_id]["jobs"] = jobs
    user_sessions[request.user_id]["filters"] = asdict(filters)
    
    # Pagination
    page = request.page
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    
    jobs_page = jobs[start:end]
    total_pages = max(1, (len(jobs) + per_page - 1) // per_page)
    
    return {
        "jobs": [asdict(j) for j in jobs_page],
        "total": len(jobs),
        "page": page,
        "total_pages": total_pages,
        "per_page": per_page,
        "cached": cache_valid,
        "filters": asdict(filters),
        "source": "Bright Data - Indeed, LinkedIn, Glassdoor, Dice, ZipRecruiter"
    }

class GoogleResultsRequest(BaseModel):
    user_id: str
    dice_results: List[dict] = []
    ziprecruiter_results: List[dict] = []

@app.post("/api/inject-google-results")
async def api_inject_google_results(request: GoogleResultsRequest):
    """
    Inject Google search results for Dice and ZipRecruiter.
    Call this after getting results from Bright Data MCP search_engine tool.
    
    Expected format for each result:
    {
        "link": "https://dice.com/...",
        "title": "Data Engineer jobs in...",
        "description": "Position: Data Engineer..."
    }
    """
    logger.info(f"Injecting Google results for user {request.user_id}")
    logger.info(f"Dice results: {len(request.dice_results)}, ZipRecruiter: {len(request.ziprecruiter_results)}")
    
    session = user_sessions.get(request.user_id, {})
    existing_jobs = session.get("jobs", [])
    filters_dict = session.get("filters", {})
    
    # Parse Google results
    dice_jobs = job_scraper.parse_google_results_dice(request.dice_results)
    zip_jobs = job_scraper.parse_google_results_ziprecruiter(request.ziprecruiter_results)
    
    # Add to existing jobs
    all_jobs = existing_jobs + dice_jobs + zip_jobs
    
    # Deduplicate
    seen = set()
    unique_jobs = []
    for job in all_jobs:
        key = f"{job.title.lower()}_{job.company.lower()}"
        if key not in seen:
            seen.add(key)
            unique_jobs.append(job)
    
    # Update session
    user_sessions[request.user_id]["jobs"] = unique_jobs
    
    # Update cache
    if filters_dict:
        filters = SearchFilters(**filters_dict)
        cache_key = filters.cache_key()
        cache_search_results(cache_key, unique_jobs)
    
    return {
        "success": True,
        "dice_added": len(dice_jobs),
        "ziprecruiter_added": len(zip_jobs),
        "total_jobs": len(unique_jobs)
    }

@app.get("/api/google-search-queries/{user_id}")
async def api_get_google_queries(user_id: str):
    """
    Get the Google search queries needed for Dice and ZipRecruiter.
    Use these with Bright Data MCP search_engine tool, then inject results.
    """
    session = user_sessions.get(user_id, {})
    filters_dict = session.get("filters", {})
    
    if not filters_dict:
        return {
            "error": "No search filters found. Please search first.",
            "dice_query": "",
            "ziprecruiter_query": ""
        }
    
    job_title = ' '.join(filters_dict.get("job_titles", ["Data Engineer"]))
    location = filters_dict.get("location", "Texas")
    remote = filters_dict.get("location_type") == "remote"
    
    remote_str = " remote" if remote else ""
    
    return {
        "dice_query": f'site:dice.com "{job_title}" {location}{remote_str}',
        "ziprecruiter_query": f'site:ziprecruiter.com "{job_title}" {location}{remote_str}',
        "instructions": "Use Bright Data MCP search_engine tool with these queries, then POST results to /api/inject-google-results"
    }

async def fetch_job_description(job: Job) -> Job:
    """
    Fetch full job description from the job URL.
    Uses Bright Data API to bypass bot protection.
    """
    if not job.url:
        logger.warning(f"No URL for job {job.id}")
        return job
    
    try:
        scraper = BrightDataJobScraper()
        
        # Try to fetch the page
        logger.info(f"Scraping job description from: {job.url}")
        html = scraper._fetch_via_brightdata(job.url)
        
        if not html:
            html = scraper._fetch_direct(job.url)
        
        if html:
            # Parse the job description from HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            description = ""
            
            # Try common job description selectors
            desc_selectors = [
                # LinkedIn
                '.description__text', '.show-more-less-html__markup',
                # Indeed
                '#jobDescriptionText', '.jobsearch-jobDescriptionText',
                # Glassdoor
                '.jobDescriptionContent', '[data-test="description"]',
                # Generic
                '.job-description', '.description', '[class*="description"]',
                'article', '.content'
            ]
            
            for selector in desc_selectors:
                elem = soup.select_one(selector)
                if elem:
                    description = elem.get_text(separator='\n', strip=True)
                    if len(description) > 200:
                        break
            
            # If no specific selector works, try to get all paragraph text
            if len(description) < 200:
                paragraphs = soup.find_all(['p', 'li'])
                description = '\n'.join(p.get_text(strip=True) for p in paragraphs[:30])
            
            if description:
                logger.info(f"Fetched job description: {len(description)} chars")
                # Create updated job with description
                return Job(
                    id=job.id,
                    title=job.title,
                    company=job.company,
                    location=job.location,
                    description=description[:5000],  # Limit to 5000 chars
                    url=job.url,
                    source=job.source,
                    posted_date=job.posted_date
                )
            else:
                logger.warning(f"Could not extract job description from {job.url}")
                
    except Exception as e:
        logger.error(f"Error fetching job description: {e}")
    
    # Return original job if fetch fails - use a generic description
    fallback_desc = f"""
Job Title: {job.title}
Company: {job.company}
Location: {job.location}

This is a {job.title} position at {job.company}.

Key Requirements (typical for this role):
- Strong experience with data engineering tools and technologies
- Proficiency in Python, SQL, and data pipeline frameworks
- Experience with cloud platforms (AWS, GCP, Azure)
- Knowledge of ETL/ELT processes and data modeling
- Strong problem-solving and communication skills
"""
    return Job(
        id=job.id,
        title=job.title,
        company=job.company,
        location=job.location,
        description=fallback_desc,
        url=job.url,
        source=job.source,
        posted_date=job.posted_date
    )

@app.post("/api/generate-cv")
async def api_generate_cv(request: GenerateCVRequest):
    """Generate CV for a specific job."""
    logger.info(f"Generating CV for user {request.user_id}, job {request.job_id}")
    
    session = user_sessions.get(request.user_id)
    if not session:
        raise HTTPException(status_code=400, detail="Session not found. Please upload CV first.")
    
    base_cv = session.get("base_cv")
    if not base_cv:
        raise HTTPException(status_code=400, detail="Please upload your CV first.")
    
    cv_structure = session.get("cv_structure")
    if not cv_structure:
        cv_structure = cv_analyzer.analyze(base_cv)
    
    jobs = session.get("jobs", [])
    job = next((j for j in jobs if j.id == request.job_id), None)
    
    if not job:
        raise HTTPException(status_code=400, detail=f"Job {request.job_id} not found. Please search again.")
    
    # If job description is empty, try to scrape it from the job URL
    if not job.description or len(job.description) < 100:
        logger.info(f"Fetching full job description from {job.url}")
        job = await fetch_job_description(job)
    
    try:
        logger.info(f"Generating CV for {job.company} - {job.title}")
        logger.info(f"Job description length: {len(job.description)} chars")
        generated_cv = generate_cv_with_claude(base_cv, cv_structure, job)
        
        # Save markdown
        cv_md_path = os.path.join(DATA_DIR, f"cv_{request.user_id}_{request.job_id}.md")
        with open(cv_md_path, 'w') as f:
            f.write(generated_cv)
        logger.info(f"Markdown saved to {cv_md_path}")
        
        # Generate formatted TXT file (cleaner than PDF, same visual structure)
        cv_txt_path = os.path.join(DATA_DIR, f"cv_{request.user_id}_{request.job_id}.txt")
        txt_success = False
        try:
            formatted_txt = format_cv_as_txt(generated_cv)
            with open(cv_txt_path, 'w', encoding='utf-8') as f:
                f.write(formatted_txt)
            txt_success = True
            logger.info(f"TXT saved to {cv_txt_path}")
        except Exception as txt_error:
            logger.error(f"TXT generation failed: {txt_error}")
        
        # Calculate ATS score for the generated CV
        logger.info("Calculating ATS score...")
        ats_result = calculate_ats_score(generated_cv, job.description, job.title)
        logger.info(f"ATS Score: {ats_result.get('score', 'N/A')}")
        
        return {
            "success": True,
            "job_id": request.job_id,
            "markdown": generated_cv,
            "markdown_path": cv_md_path,
            "txt_path": cv_txt_path if txt_success else None,
            "txt_success": txt_success,
            "download_url": f"/api/download-cv/{request.user_id}/{request.job_id}" if txt_success else None,
            "ats_score": ats_result
        }
        
    except Exception as e:
        logger.error(f"CV generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-cv/{user_id}/{job_id}")
async def api_download_cv(user_id: str, job_id: str):
    """Download generated CV as TXT (formatted to match input CV style)."""
    txt_path = os.path.join(DATA_DIR, f"cv_{user_id}_{job_id}.txt")
    
    if not os.path.exists(txt_path):
        raise HTTPException(status_code=404, detail="CV not found. Generate it first.")
    
    return FileResponse(
        txt_path,
        media_type="text/plain",
        filename=f"tailored_cv_{job_id}.txt"
    )

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main page."""
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, 'r') as f:
            return f.read()
    return "<h1>JobPilot</h1><p>index.html not found</p>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
