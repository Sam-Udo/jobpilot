"""
Configuration management for JobPilot.

Loads settings from environment variables with sensible defaults.
Uses pydantic-settings for validation.
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class DatabaseSettings(BaseSettings):
    """Database connection settings."""
    
    postgres_url: str = Field(
        default="postgresql://localhost:5432/jobpilot",
        description="PostgreSQL connection URL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for Celery and caching"
    )
    
    class Config:
        env_prefix = "DB_"


class LLMSettings(BaseSettings):
    """LLM API settings - Claude (Anthropic) as primary."""
    
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (backup/optional)"
    )
    default_provider: str = Field(
        default="claude",
        description="Primary LLM provider: claude (recommended)"
    )
    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default Claude model - claude-sonnet-4-20250514 (best balance), claude-opus-4-20250514 (most capable), claude-3-5-haiku-20241022 (fastest)"
    )
    temperature: float = Field(
        default=0.3,
        description="LLM temperature for deterministic outputs"
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens in LLM response"
    )
    
    class Config:
        env_prefix = "LLM_"


class GoogleSheetsSettings(BaseSettings):
    """Google Sheets integration settings."""
    
    credentials_file: str = Field(
        default="google_credentials.json",
        description="Path to Google service account credentials"
    )
    default_sheet_id: Optional[str] = Field(
        default=None,
        description="Default Google Sheet ID for job listings"
    )
    
    class Config:
        env_prefix = "GOOGLE_"


class ScraperSettings(BaseSettings):
    """Scraper configuration."""
    
    # Rate limiting
    request_delay: float = Field(
        default=2.0,
        description="Seconds between requests"
    )
    max_concurrent: int = Field(
        default=5,
        description="Maximum concurrent scraping tasks"
    )
    
    # Bright Data MCP batch limits
    mcp_batch_size: int = Field(
        default=10,
        description="Max URLs per MCP batch request"
    )
    
    class Config:
        env_prefix = "SCRAPER_"


class WorkflowSettings(BaseSettings):
    """Workflow and automation settings."""
    
    # Modes
    default_mode: str = Field(
        default="supervised",
        description="Default mode: supervised or autonomous"
    )
    
    # Thresholds
    ats_score_threshold: float = Field(
        default=0.85,
        description="Minimum ATS score to auto-approve CV in autonomous mode"
    )
    confidence_threshold: float = Field(
        default=0.80,
        description="Confidence level below which agent asks for human input"
    )
    
    # Timeouts
    user_response_timeout_hours: int = Field(
        default=48,
        description="Hours to wait for user response before timing out"
    )
    
    class Config:
        env_prefix = "WORKFLOW_"


class Settings(BaseSettings):
    """Main application settings aggregating all sub-settings."""
    
    # Application
    app_name: str = "JobPilot"
    debug: bool = Field(default=False)
    environment: str = Field(default="development")
    
    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    google_sheets: GoogleSheetsSettings = Field(default_factory=GoogleSheetsSettings)
    scraper: ScraperSettings = Field(default_factory=ScraperSettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    
    class Config:
        env_prefix = "JOBPILOT_"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings. Useful for dependency injection."""
    return settings

