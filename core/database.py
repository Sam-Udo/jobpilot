"""
Database models for JobPilot.

Uses SQLAlchemy for ORM with PostgreSQL.
Includes pgvector for knowledge base embeddings.
"""

from datetime import datetime
from typing import Optional, List
import uuid
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    Text, JSON, ForeignKey, Enum as SQLEnum, Index
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

from .schemas import WorkflowState, LocationType, ApplicationMode, ATSType


Base = declarative_base()


def generate_uuid():
    """Generate a new UUID string."""
    return str(uuid.uuid4())


# ============================================================================
# User & Authentication
# ============================================================================

class User(Base):
    """User account."""
    
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    
    # Profile info
    full_name = Column(String(255))
    phone = Column(String(50))
    city = Column(String(100))
    state = Column(String(100))
    country = Column(String(100), default="United States")
    
    # Work authorization
    work_authorized = Column(Boolean, default=True)
    requires_sponsorship = Column(Boolean, default=False)
    
    # Voluntary disclosures (encrypted at app level)
    veteran_status = Column(String(50))
    disability_status = Column(String(50))
    gender = Column(String(50))
    ethnicity = Column(String(100))
    
    # Links
    linkedin_url = Column(String(500))
    github_url = Column(String(500))
    portfolio_url = Column(String(500))
    
    # Settings
    default_mode = Column(String(20), default="supervised")
    notification_email = Column(Boolean, default=True)
    notification_push = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    workflows = relationship("Workflow", back_populates="user")
    cvs = relationship("UserCV", back_populates="user")
    credentials = relationship("UserCredential", back_populates="user")
    knowledge_entries = relationship("KnowledgeEntry", back_populates="user")


# ============================================================================
# Credentials Vault (Encrypted)
# ============================================================================

class UserCredential(Base):
    """Encrypted credentials for job portals.
    
    The password field is encrypted using AES-256 at the application layer.
    Never store plain text passwords.
    """
    
    __tablename__ = "user_credentials"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    
    # Portal info
    portal_name = Column(String(100), nullable=False)  # "LinkedIn", "Workday-Microsoft"
    portal_url = Column(String(500))
    
    # Credentials (encrypted)
    username = Column(String(255), nullable=False)  # Usually email
    password_encrypted = Column(Text, nullable=False)  # AES-256 encrypted
    
    # Session data (for avoiding re-login)
    cookies_json = Column(Text)  # Encrypted session cookies
    local_storage_json = Column(Text)  # Encrypted localStorage
    
    # Metadata
    last_login_at = Column(DateTime)
    login_success_count = Column(Integer, default=0)
    login_failure_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="credentials")
    
    __table_args__ = (
        Index("idx_credentials_user_portal", "user_id", "portal_name"),
    )


# ============================================================================
# Knowledge Base (RAG for form answers)
# ============================================================================

class KnowledgeEntry(Base):
    """Knowledge base entry for answering form questions.
    
    Uses vector embeddings for semantic search.
    Example: "Years of Python experience: 5 years"
    """
    
    __tablename__ = "knowledge_entries"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    
    # Content
    question_pattern = Column(Text, nullable=False)  # Common question patterns
    answer = Column(Text, nullable=False)
    
    # Categorization
    category = Column(String(100))  # "experience", "skills", "personal", "legal"
    
    # For semantic search (pgvector)
    # Note: In production, use pgvector's VECTOR type
    # embedding = Column(VECTOR(1536))  # Embedding dimension (if using vector search)
    embedding_json = Column(JSONB)  # Fallback: store as JSON array
    
    # Metadata
    source = Column(String(50))  # "user_input", "cv_extracted", "form_learned"
    confidence = Column(Float, default=1.0)
    times_used = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="knowledge_entries")
    
    __table_args__ = (
        Index("idx_knowledge_user_category", "user_id", "category"),
    )


# ============================================================================
# User CVs
# ============================================================================

class UserCV(Base):
    """User's base CV and generated tailored CVs."""
    
    __tablename__ = "user_cvs"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    
    # CV type
    is_base_cv = Column(Boolean, default=False)  # True for the original CV
    
    # For tailored CVs
    target_job_id = Column(String(36), ForeignKey("jobs.id"))
    target_company = Column(String(255))
    target_role = Column(String(255))
    
    # Content
    cv_text = Column(Text)  # Plain text version
    cv_markdown = Column(Text)  # Markdown version
    cv_json = Column(JSONB)  # Structured version (GeneratedCV schema)
    
    # Files
    cv_pdf_url = Column(String(500))  # S3 URL for PDF
    cv_docx_url = Column(String(500))  # S3 URL for DOCX
    
    # Validation
    ats_score = Column(Float)
    requirements_covered = Column(Integer)
    total_requirements = Column(Integer)
    
    # Approval
    status = Column(String(50), default="draft")  # draft, pending_approval, approved, rejected
    approved_at = Column(DateTime)
    rejection_reason = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="cvs")
    target_job = relationship("Job", back_populates="tailored_cvs")


# ============================================================================
# Jobs
# ============================================================================

class Job(Base):
    """Scraped job listing."""
    
    __tablename__ = "jobs"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    external_id = Column(String(255))  # ID from the job board
    
    # Core info
    company = Column(String(255), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    location = Column(String(255))
    location_type = Column(String(20), default="onsite")
    
    # URLs
    job_url = Column(String(1000), nullable=False)
    apply_url = Column(String(1000))
    
    # Description (store full text for JD analysis)
    description = Column(Text)
    requirements_json = Column(JSONB)  # Parsed requirements
    
    # Metadata
    posted_date = Column(DateTime)
    expiration_date = Column(DateTime)
    salary_range = Column(String(100))
    
    # Source
    source = Column(String(100))  # "Indeed", "LinkedIn", "Greenhouse API"
    ats_type = Column(String(50), default="unknown")
    
    # Deduplication
    description_hash = Column(String(64), index=True)  # SHA-256 of description
    
    # Timestamps
    scraped_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    tailored_cvs = relationship("UserCV", back_populates="target_job")
    applications = relationship("Application", back_populates="job")
    workflow_jobs = relationship("WorkflowJob", back_populates="job")
    
    __table_args__ = (
        Index("idx_jobs_company_title", "company", "title"),
        Index("idx_jobs_source", "source"),
    )


# ============================================================================
# Workflows
# ============================================================================

class Workflow(Base):
    """Job application workflow tracking.
    
    Tracks the state of a user's job search session.
    """
    
    __tablename__ = "workflows"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    
    # State
    state = Column(String(50), default="init", index=True)
    
    # Search preferences (snapshot)
    preferences_json = Column(JSONB)  # JobSearchPreferences
    
    # Mode
    application_mode = Column(String(20), default="supervised")
    
    # Progress counters
    jobs_found = Column(Integer, default=0)
    jobs_selected = Column(Integer, default=0)
    cvs_generated = Column(Integer, default=0)
    cvs_approved = Column(Integer, default=0)
    applications_submitted = Column(Integer, default=0)
    applications_failed = Column(Integer, default=0)
    
    # Current action
    current_job_id = Column(String(36))
    current_action = Column(String(100))
    
    # Error tracking
    last_error = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="workflows")
    workflow_jobs = relationship("WorkflowJob", back_populates="workflow")
    
    __table_args__ = (
        Index("idx_workflows_user_state", "user_id", "state"),
    )


class WorkflowJob(Base):
    """Junction table linking workflows to selected jobs."""
    
    __tablename__ = "workflow_jobs"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    workflow_id = Column(String(36), ForeignKey("workflows.id"), nullable=False)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    
    # Job-specific state within workflow
    status = Column(String(50), default="selected")  
    # selected, cv_generating, cv_ready, applying, needs_input, submitted, failed
    
    # Relevance
    relevance_score = Column(Float)  # 0-100 match score
    user_priority = Column(Integer)  # User's ranking
    
    # CV
    tailored_cv_id = Column(String(36), ForeignKey("user_cvs.id"))
    
    # Application
    application_id = Column(String(36), ForeignKey("applications.id"))
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    workflow = relationship("Workflow", back_populates="workflow_jobs")
    job = relationship("Job", back_populates="workflow_jobs")


# ============================================================================
# Applications
# ============================================================================

class Application(Base):
    """Job application tracking."""
    
    __tablename__ = "applications"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    workflow_id = Column(String(36), ForeignKey("workflows.id"))
    cv_id = Column(String(36), ForeignKey("user_cvs.id"))
    
    # Status
    status = Column(String(50), default="pending")
    # pending, in_progress, needs_input, submitted, confirmed, rejected, withdrawn
    
    # Form data
    form_data_json = Column(JSONB)  # Captured form fields and values
    
    # Questions needing human input
    pending_questions_json = Column(JSONB)  # List of HumanInputRequest
    
    # Submission
    submitted_at = Column(DateTime)
    confirmation_number = Column(String(255))
    confirmation_screenshot_url = Column(String(500))
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    job = relationship("Job", back_populates="applications")


# ============================================================================
# Interaction Logs (Audit Trail)
# ============================================================================

class InteractionLog(Base):
    """Logs every browser action for debugging and audit.
    
    Every click, type, navigation is logged with a screenshot.
    """
    
    __tablename__ = "interaction_logs"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    workflow_id = Column(String(36), ForeignKey("workflows.id"))
    application_id = Column(String(36), ForeignKey("applications.id"))
    
    # Action
    action_type = Column(String(50))  # navigate, click, type, select, submit
    target_selector = Column(String(500))
    target_text = Column(Text)
    input_value = Column(Text)
    
    # Result
    success = Column(Boolean)
    error_message = Column(Text)
    
    # Evidence
    screenshot_url = Column(String(500))  # S3 URL
    page_url = Column(String(1000))
    dom_snapshot_url = Column(String(500))  # Compressed DOM
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    duration_ms = Column(Integer)  # Action duration
    
    __table_args__ = (
        Index("idx_logs_workflow", "workflow_id"),
        Index("idx_logs_application", "application_id"),
    )

