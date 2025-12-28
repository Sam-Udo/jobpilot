"""
Discovery Agent - Layer 1

Finds jobs matching user preferences using:
1. Bright Data MCP for protected sites (Indeed, LinkedIn, Workday)
2. Direct APIs for Greenhouse, Lever
3. Reverse search strategy for efficiency

Integrates with existing scrapers in /scrapers directory.
"""

import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import json

from jobpilot.core.schemas import (
    JobSearchPreferences, JobListing, JobListingBatch,
    LocationType, ATSType
)
from jobpilot.core.config import get_settings

# Import existing scrapers
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

logger = logging.getLogger(__name__)


class DiscoveryAgent:
    """
    Agent responsible for finding relevant jobs.
    
    Uses a tiered approach:
    1. Direct APIs (Greenhouse, Lever) - fastest, most reliable
    2. Bright Data MCP - for protected sites
    3. Reverse search - for aggregator sites
    """
    
    def __init__(self, config_path: str = "data/company_career_urls.json"):
        """
        Initialize the Discovery Agent.
        
        Args:
            config_path: Path to company configuration JSON
        """
        self.settings = get_settings()
        self.config = self._load_config(config_path)
        self.companies = self.config.get("companies", {})
        self.job_titles = self.config.get("job_titles", [])
        
        # Track seen jobs for deduplication
        self._seen_hashes: Set[str] = set()
        
        logger.info(f"Discovery Agent initialized with {len(self.companies)} companies")
    
    def _load_config(self, path: str) -> dict:
        """Load company configuration."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {path}")
            return {"companies": {}, "job_titles": []}
    
    def _hash_job(self, job: JobListing) -> str:
        """Create hash for deduplication."""
        content = f"{job.company}|{job.title}|{job.location}|{job.description or ''}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _is_duplicate(self, job: JobListing) -> bool:
        """Check if job is a duplicate."""
        job_hash = self._hash_job(job)
        if job_hash in self._seen_hashes:
            return True
        self._seen_hashes.add(job_hash)
        return False
    
    # =========================================================================
    # Preference Matching
    # =========================================================================
    
    def _matches_preferences(self, job: JobListing, prefs: JobSearchPreferences) -> bool:
        """Check if a job matches user preferences."""
        
        # Company filter
        if prefs.companies:
            company_lower = job.company.lower()
            if not any(c.lower() in company_lower or company_lower in c.lower() 
                      for c in prefs.companies):
                return False
        
        # Location type filter
        if prefs.location_type != LocationType.ANY:
            if job.location_type != prefs.location_type:
                # Allow remote jobs when hybrid is requested
                if not (prefs.location_type == LocationType.HYBRID and 
                       job.location_type == LocationType.REMOTE):
                    return False
        
        # Country filter (basic)
        if prefs.countries:
            location_lower = job.location.lower()
            if not any(c.lower() in location_lower for c in prefs.countries):
                # Check for US indicators
                if "united states" in prefs.countries or "us" in prefs.countries:
                    if not self._is_us_location(job.location):
                        return False
        
        return True
    
    def _is_us_location(self, location: str) -> bool:
        """Check if location is in the US."""
        if not location:
            return False
        
        location_lower = location.lower()
        
        # Explicit US indicators
        us_indicators = ['usa', 'united states', 'us-', '-us', 'u.s.', 
                        'remote - us', 'us remote']
        if any(ind in location_lower for ind in us_indicators):
            return True
        
        # Non-US exclusions
        non_us = ['canada', 'uk', 'united kingdom', 'london', 'india', 
                 'germany', 'ireland', 'singapore', 'australia']
        if any(loc in location_lower for loc in non_us):
            return False
        
        # US state names and abbreviations
        us_states = ['california', 'ca', 'new york', 'ny', 'texas', 'tx', 
                    'washington', 'wa', 'colorado', 'co', 'florida', 'fl']
        if any(state in location_lower for state in us_states):
            return True
        
        return False
    
    def _score_relevance(self, job: JobListing, prefs: JobSearchPreferences) -> float:
        """
        Score job relevance to preferences (0-100).
        
        Higher score = better match.
        """
        score = 50.0  # Base score
        
        # Title match bonus
        title_lower = job.title.lower()
        for pref_title in prefs.job_titles:
            if pref_title.lower() in title_lower:
                score += 20
                break
        
        # Company match bonus
        if prefs.companies:
            company_lower = job.company.lower()
            for pref_company in prefs.companies:
                if pref_company.lower() == company_lower:
                    score += 15  # Exact match
                    break
                elif pref_company.lower() in company_lower:
                    score += 10  # Partial match
                    break
        
        # Location type match bonus
        if prefs.location_type != LocationType.ANY:
            if job.location_type == prefs.location_type:
                score += 10
            elif job.location_type == LocationType.REMOTE:
                score += 5  # Remote is always good
        
        # City match bonus
        if prefs.cities:
            location_lower = job.location.lower()
            if any(city.lower() in location_lower for city in prefs.cities):
                score += 5
        
        return min(100.0, score)
    
    # =========================================================================
    # Direct API Scrapers
    # =========================================================================
    
    def scrape_greenhouse_api(self, company_name: str, board_token: str, 
                               prefs: JobSearchPreferences) -> List[JobListing]:
        """
        Scrape jobs from Greenhouse public API.
        
        API: https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true
        """
        import requests
        
        jobs = []
        api_url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true"
        
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for job_data in data.get('jobs', []):
                title = job_data.get('title', '')
                
                # Check if title matches preferences
                if prefs.job_titles:
                    title_lower = title.lower()
                    if not any(t.lower() in title_lower for t in prefs.job_titles):
                        continue
                
                location = job_data.get('location', {}).get('name', '')
                
                # Create job listing
                job = JobListing(
                    external_id=str(job_data.get('id', '')),
                    company=company_name,
                    title=title,
                    location=location or 'USA',
                    location_type=self._detect_location_type(location),
                    job_url=job_data.get('absolute_url', ''),
                    apply_url=job_data.get('absolute_url', ''),
                    description=job_data.get('content', ''),
                    posted_date=self._parse_date(job_data.get('updated_at')),
                    source='Greenhouse API',
                    ats_type=ATSType.GREENHOUSE
                )
                
                # Apply filters
                if self._matches_preferences(job, prefs) and not self._is_duplicate(job):
                    job.relevance_score = self._score_relevance(job, prefs)
                    jobs.append(job)
            
            logger.info(f"Greenhouse [{company_name}]: Found {len(jobs)} matching jobs")
            
        except Exception as e:
            logger.error(f"Greenhouse [{company_name}]: Error - {e}")
        
        return jobs
    
    def scrape_lever_api(self, company_name: str, company_id: str,
                         prefs: JobSearchPreferences) -> List[JobListing]:
        """
        Scrape jobs from Lever public API.
        
        API: https://api.lever.co/v0/postings/{company}?mode=json
        """
        import requests
        
        jobs = []
        api_url = f"https://api.lever.co/v0/postings/{company_id}?mode=json"
        
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for job_data in data:
                title = job_data.get('text', '')
                
                # Check if title matches preferences
                if prefs.job_titles:
                    title_lower = title.lower()
                    if not any(t.lower() in title_lower for t in prefs.job_titles):
                        continue
                
                location = job_data.get('categories', {}).get('location', '')
                
                job = JobListing(
                    external_id=job_data.get('id', ''),
                    company=company_name,
                    title=title,
                    location=location or 'USA',
                    location_type=self._detect_location_type(location),
                    job_url=job_data.get('hostedUrl', ''),
                    apply_url=job_data.get('applyUrl', job_data.get('hostedUrl', '')),
                    description=job_data.get('descriptionPlain', ''),
                    source='Lever API',
                    ats_type=ATSType.LEVER
                )
                
                if self._matches_preferences(job, prefs) and not self._is_duplicate(job):
                    job.relevance_score = self._score_relevance(job, prefs)
                    jobs.append(job)
            
            logger.info(f"Lever [{company_name}]: Found {len(jobs)} matching jobs")
            
        except Exception as e:
            logger.error(f"Lever [{company_name}]: Error - {e}")
        
        return jobs
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _detect_location_type(self, location: str) -> LocationType:
        """Detect Remote/Hybrid/Onsite from location string."""
        if not location:
            return LocationType.ONSITE
        
        location_lower = location.lower()
        
        if any(kw in location_lower for kw in ['remote', 'anywhere', 'wfh']):
            return LocationType.REMOTE
        if any(kw in location_lower for kw in ['hybrid', 'flexible']):
            return LocationType.HYBRID
        
        return LocationType.ONSITE
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            return None
    
    # =========================================================================
    # Main Discovery Methods
    # =========================================================================
    
    def discover_api_jobs(self, prefs: JobSearchPreferences) -> List[JobListing]:
        """
        Discover jobs from direct API sources (Greenhouse, Lever).
        
        This is fast and reliable - no proxies needed.
        """
        all_jobs = []
        import re
        
        # Filter companies by preferences if specified
        target_companies = prefs.companies if prefs.companies else list(self.companies.keys())
        
        for company_name in target_companies:
            if company_name not in self.companies:
                continue
            
            info = self.companies[company_name]
            ats = info.get('ats', '')
            api_url = info.get('api_url', '')
            
            if ats == 'greenhouse' and api_url:
                # Extract board token
                match = re.search(r'/boards/([^/]+)/jobs', api_url)
                if match:
                    board_token = match.group(1)
                    jobs = self.scrape_greenhouse_api(company_name, board_token, prefs)
                    all_jobs.extend(jobs)
            
            elif ats == 'lever' and api_url:
                # Extract company id
                match = re.search(r'/postings/([^?]+)', api_url)
                if match:
                    company_id = match.group(1)
                    jobs = self.scrape_lever_api(company_name, company_id, prefs)
                    all_jobs.extend(jobs)
        
        logger.info(f"API Discovery: Found {len(all_jobs)} total jobs from direct APIs")
        return all_jobs
    
    def get_mcp_scrape_urls(self, prefs: JobSearchPreferences) -> Dict[str, List[Dict]]:
        """
        Generate URLs for Bright Data MCP scraping.
        
        Returns URLs grouped by ATS type for batch processing.
        """
        urls = {
            'workday': [],
            'custom': [],
            'aggregator': []
        }
        
        target_companies = prefs.companies if prefs.companies else list(self.companies.keys())
        
        for company_name in target_companies:
            if company_name not in self.companies:
                continue
            
            info = self.companies[company_name]
            ats = info.get('ats', '')
            career_url = info.get('career_url', '')
            
            if not career_url:
                continue
            
            # Skip API-based companies (already handled)
            if ats in ['greenhouse', 'lever'] and info.get('api_url'):
                continue
            
            # Generate search URLs for each job title
            for title in (prefs.job_titles or self.job_titles[:5]):
                url = career_url.replace('{query}', title.replace(' ', '+'))
                
                if ats == 'workday':
                    urls['workday'].append({
                        'company': company_name,
                        'title': title,
                        'url': url,
                        'ats': ats
                    })
                else:
                    urls['custom'].append({
                        'company': company_name,
                        'title': title,
                        'url': url,
                        'ats': ats
                    })
        
        # Add aggregator URLs for reverse search
        for title in (prefs.job_titles or self.job_titles[:5]):
            query = title.replace(' ', '+')
            
            urls['aggregator'].extend([
                {
                    'source': 'Indeed',
                    'title': title,
                    'url': f'https://www.indeed.com/jobs?q={query}&l=United+States&limit=50'
                },
                {
                    'source': 'LinkedIn',
                    'title': title,
                    'url': f'https://www.linkedin.com/jobs/search?keywords={query}&location=United%20States'
                }
            ])
        
        logger.info(f"MCP URLs: {len(urls['workday'])} Workday, "
                   f"{len(urls['custom'])} Custom, {len(urls['aggregator'])} Aggregator")
        
        return urls
    
    def search(self, prefs: JobSearchPreferences) -> JobListingBatch:
        """
        Main search method. Discovers jobs from all sources.
        
        Args:
            prefs: User's job search preferences
            
        Returns:
            Batch of discovered job listings
        """
        logger.info(f"Starting job discovery with preferences: "
                   f"companies={len(prefs.companies)}, titles={len(prefs.job_titles)}")
        
        # Phase 1: Direct APIs (fast, reliable)
        api_jobs = self.discover_api_jobs(prefs)
        
        # Phase 2: Generate MCP URLs for further scraping
        # Note: Actual MCP calls are made by the orchestrator
        mcp_urls = self.get_mcp_scrape_urls(prefs)
        
        # Store URLs for orchestrator to process
        self._pending_mcp_urls = mcp_urls
        
        # Sort by relevance score
        api_jobs.sort(key=lambda j: j.relevance_score or 0, reverse=True)
        
        return JobListingBatch(
            jobs=api_jobs,
            total_found=len(api_jobs),
            search_query=json.dumps(prefs.dict(), default=str)
        )
    
    def get_pending_mcp_urls(self) -> Dict[str, List[Dict]]:
        """Get URLs that need MCP scraping."""
        return getattr(self, '_pending_mcp_urls', {})


# ============================================================================
# Job Ranker (LLM-based relevance scoring)
# ============================================================================

class JobRanker:
    """
    Uses LLM to rank jobs by relevance to user preferences.
    
    More sophisticated than rule-based scoring.
    """
    
    def __init__(self, llm_client=None):
        """Initialize with optional LLM client."""
        self.llm_client = llm_client
    
    def rank_batch(self, jobs: List[JobListing], prefs: JobSearchPreferences,
                   user_cv_summary: Optional[str] = None) -> List[JobListing]:
        """
        Rank a batch of jobs using LLM.
        
        Args:
            jobs: List of jobs to rank
            prefs: User preferences
            user_cv_summary: Optional CV summary for better matching
            
        Returns:
            Jobs sorted by relevance with updated scores
        """
        if not self.llm_client:
            # Fallback to rule-based ranking
            logger.warning("No LLM client, using rule-based ranking")
            return sorted(jobs, key=lambda j: j.relevance_score or 0, reverse=True)
        
        # Build prompt for batch ranking
        prompt = self._build_ranking_prompt(jobs, prefs, user_cv_summary)
        
        try:
            # Call LLM for ranking
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            rankings = json.loads(response.choices[0].message.content)
            
            # Update job scores
            for job in jobs:
                job_key = f"{job.company}|{job.title}"
                if job_key in rankings:
                    job.relevance_score = rankings[job_key].get('score', 50)
            
        except Exception as e:
            logger.error(f"LLM ranking failed: {e}")
        
        return sorted(jobs, key=lambda j: j.relevance_score or 0, reverse=True)
    
    def _build_ranking_prompt(self, jobs: List[JobListing], 
                               prefs: JobSearchPreferences,
                               cv_summary: Optional[str]) -> str:
        """Build prompt for LLM ranking."""
        jobs_list = "\n".join([
            f"- {j.company} | {j.title} | {j.location}" 
            for j in jobs[:20]  # Limit to 20 for context window
        ])
        
        return f"""Score these jobs from 0-100 based on match to candidate preferences.

PREFERENCES:
- Target roles: {', '.join(prefs.job_titles)}
- Target companies: {', '.join(prefs.companies) if prefs.companies else 'Any'}
- Location: {prefs.location_type.value}

{'CANDIDATE SUMMARY: ' + cv_summary if cv_summary else ''}

JOBS:
{jobs_list}

Return JSON with job key (company|title) and score:
{{"Company|Title": {{"score": 85, "reason": "Good match because..."}}, ...}}
"""

