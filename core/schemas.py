"""
Pydantic schemas for data validation and LLM structured outputs.

These schemas ensure:
1. User input is validated
2. LLM outputs conform to expected structure
3. API responses are consistent
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator


# ============================================================================
# Enums for State Management
# ============================================================================

class WorkflowState(str, Enum):
    """States for the job application workflow."""
    
    INIT = "init"
    SEARCHING = "searching"
    JOBS_FOUND = "jobs_found"
    JOBS_SELECTED = "jobs_selected"
    CV_GENERATING = "cv_generating"
    CV_READY = "cv_ready"
    AWAITING_USER_APPROVAL = "awaiting_user_approval"
    CV_APPROVED = "cv_approved"
    APPLYING = "applying"
    NEEDS_INPUT = "needs_input"
    INPUT_RECEIVED = "input_received"
    SUBMITTED = "submitted"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LocationType(str, Enum):
    """Job location type."""
    
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    ANY = "any"


class ApplicationMode(str, Enum):
    """Application mode - supervised requires human approval."""
    
    SUPERVISED = "supervised"
    AUTONOMOUS = "autonomous"


class ATSType(str, Enum):
    """Applicant Tracking System types."""
    
    WORKDAY = "workday"
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ICIMS = "icims"
    TALEO = "taleo"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


# ============================================================================
# User Preferences (Input from Chat)
# ============================================================================

class JobSearchPreferences(BaseModel):
    """User preferences for job search. Collected via chat."""
    
    # Target companies (optional - if empty, searches all)
    companies: List[str] = Field(
        default_factory=list,
        description="List of specific companies to target"
    )
    
    # Job titles to search
    job_titles: List[str] = Field(
        default_factory=list,
        description="Job titles to search for"
    )
    
    # Location preferences
    countries: List[str] = Field(
        default=["United States"],
        description="Countries to search in"
    )
    cities: List[str] = Field(
        default_factory=list,
        description="Specific cities (optional)"
    )
    location_type: LocationType = Field(
        default=LocationType.ANY,
        description="Remote, Hybrid, Onsite, or Any"
    )
    
    # Experience level
    min_years_experience: Optional[int] = Field(
        default=None,
        description="Minimum years of experience"
    )
    max_years_experience: Optional[int] = Field(
        default=None,
        description="Maximum years of experience"
    )
    
    # Salary expectations (optional)
    min_salary: Optional[int] = Field(
        default=None,
        description="Minimum salary expectation"
    )
    salary_currency: str = Field(
        default="USD",
        description="Salary currency"
    )
    
    # Mode
    application_mode: ApplicationMode = Field(
        default=ApplicationMode.SUPERVISED,
        description="Supervised (requires approval) or Autonomous"
    )


class UserProfile(BaseModel):
    """User profile with personal information for applications."""
    
    # Basic info
    full_name: str
    email: str
    phone: Optional[str] = None
    
    # Location
    city: Optional[str] = None
    state: Optional[str] = None
    country: str = "United States"
    
    # Work authorization
    work_authorized: bool = True
    requires_sponsorship: bool = False
    
    # Voluntary disclosures (for EEO questions)
    veteran_status: Optional[str] = None  # "veteran", "non-veteran", "prefer_not_to_say"
    disability_status: Optional[str] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    
    # Links
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None


# ============================================================================
# Job Data Schemas
# ============================================================================

class JobListing(BaseModel):
    """Schema for a scraped job listing."""
    
    # Identifiers
    job_id: Optional[str] = None
    external_id: Optional[str] = None  # ID from the job board
    
    # Core info
    company: str
    title: str
    location: str
    location_type: LocationType = LocationType.ONSITE
    
    # URLs
    job_url: str
    apply_url: Optional[str] = None
    
    # Description
    description: Optional[str] = None
    requirements: Optional[List[str]] = None
    
    # Metadata
    posted_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    salary_range: Optional[str] = None
    
    # Source info
    source: str  # "Indeed", "LinkedIn", "Greenhouse API", etc.
    ats_type: ATSType = ATSType.UNKNOWN
    
    # Ranking
    relevance_score: Optional[float] = None  # 0-100 match to user preferences
    
    # Timestamps
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class JobListingBatch(BaseModel):
    """Batch of job listings from a scrape session."""
    
    jobs: List[JobListing]
    total_found: int
    search_query: str
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# CV Generation Schemas (For LLM Structured Output)
# ============================================================================

class JDRequirement(BaseModel):
    """Single requirement extracted from a job description."""
    
    text: str = Field(description="The requirement text")
    category: str = Field(description="Category like 'Technical', 'Leadership', etc.")
    priority: str = Field(description="MUST-HAVE, SHOULD-HAVE, or NICE-TO-HAVE")
    keywords: List[str] = Field(default_factory=list, description="Key terms to mirror")


class JDAnalysis(BaseModel):
    """Structured analysis of a job description. LLM output schema."""
    
    # Role metadata
    role_title: str
    company: str
    department: Optional[str] = None
    seniority_level: str  # Entry/Mid/Senior/Lead/etc.
    
    # Requirements
    requirements: List[JDRequirement]
    total_requirements: int
    
    # Tools and skills
    tools_mentioned: List[str] = Field(default_factory=list)
    methodologies: List[str] = Field(default_factory=list)
    certifications_mentioned: List[str] = Field(default_factory=list)
    
    # Language patterns
    key_verbs: List[str] = Field(default_factory=list)
    key_terms: List[str] = Field(default_factory=list)


class CVBulletPoint(BaseModel):
    """Single bullet point in a CV experience section."""
    
    text: str = Field(description="The bullet point text")
    addresses_requirement: Optional[str] = Field(
        default=None,
        description="Which JD requirement this addresses"
    )
    metrics_included: bool = Field(default=False)


class CVExperience(BaseModel):
    """Single work experience entry."""
    
    company: str
    title: str
    department: Optional[str] = None
    start_date: str
    end_date: Optional[str] = None  # None = "Present"
    location: Optional[str] = None
    bullets: List[CVBulletPoint]


class GeneratedCV(BaseModel):
    """Complete generated CV. LLM output schema."""
    
    # Target info
    target_company: str
    target_role: str
    
    # Candidate info (from original CV)
    candidate_name: str
    contact_info: Dict[str, str] = Field(default_factory=dict)
    
    # Generated content
    experiences: List[CVExperience]
    skills_section: str
    education: List[str]
    certifications: List[str] = Field(default_factory=list)
    
    # Validation
    ats_score: float = Field(description="Estimated ATS match score 0-100")
    requirements_covered: int
    total_requirements: int
    
    # Audit trail
    fabricated_content: List[str] = Field(
        default_factory=list,
        description="List of fabricated project descriptions"
    )
    preserved_facts: List[str] = Field(
        default_factory=list,
        description="Facts preserved from original CV"
    )


# ============================================================================
# Form Filling Schemas
# ============================================================================

class FormField(BaseModel):
    """Single form field detected on application page."""
    
    field_id: str
    field_type: str  # text, select, checkbox, radio, textarea, file
    label: str
    required: bool = False
    options: Optional[List[str]] = None  # For select/radio fields
    current_value: Optional[str] = None
    
    # Agent analysis
    confidence: float = Field(
        default=0.0,
        description="Agent's confidence in knowing the answer (0-1)"
    )
    suggested_value: Optional[str] = None
    needs_human_input: bool = False


class ApplicationForm(BaseModel):
    """Detected application form structure."""
    
    job_id: str
    page_url: str
    form_fields: List[FormField]
    total_fields: int
    fields_filled: int = 0
    fields_needing_input: int = 0
    
    # State
    screenshot_url: Optional[str] = None  # S3 URL for snapshot
    dom_snapshot: Optional[str] = None  # Serialized DOM


class HumanInputRequest(BaseModel):
    """Request for human input when agent is stuck."""
    
    workflow_id: str
    job_id: str
    company: str
    question: str
    field_type: str
    options: Optional[List[str]] = None
    context: Optional[str] = None
    screenshot_url: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    timeout_at: Optional[datetime] = None


# ============================================================================
# Workflow Schemas
# ============================================================================

class WorkflowStatus(BaseModel):
    """Current status of a job application workflow."""
    
    workflow_id: str
    user_id: str
    state: WorkflowState
    
    # Progress
    jobs_found: int = 0
    jobs_selected: int = 0
    cvs_generated: int = 0
    cvs_approved: int = 0
    applications_submitted: int = 0
    applications_pending_input: int = 0
    
    # Current action
    current_job: Optional[str] = None
    current_action: Optional[str] = None
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    # Errors
    last_error: Optional[str] = None


# ============================================================================
# API Response Schemas
# ============================================================================

class APIResponse(BaseModel):
    """Standard API response wrapper."""
    
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None


class ChatMessage(BaseModel):
    """Chat message between user and agent."""
    
    role: str  # "user", "agent", "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # For agent messages
    action_type: Optional[str] = None  # "question", "update", "confirmation", etc.
    requires_response: bool = False
    options: Optional[List[str]] = None  # Quick reply options

