"""
CV Architect Agent - Layer 2

Generates tailored CVs using Claude (Anthropic):
1. JD Parser - Extracts requirements from job descriptions
2. CV Generator - Creates tailored CV using Master CV Prompt v3.0
3. Critic Agent - Validates CV for hallucinations and ATS compliance
4. PDF Renderer - Converts approved CV to PDF

This implements a Creator -> Critic -> User Approval loop.
"""

import logging
import json
from typing import Optional, List, Dict, Tuple
from datetime import datetime

from jobpilot.core.schemas import (
    JobListing, JDAnalysis, JDRequirement, GeneratedCV,
    CVExperience, CVBulletPoint
)
from jobpilot.core.config import get_settings
from jobpilot.core.llm_client import ClaudeLLMClient, get_llm_client

logger = logging.getLogger(__name__)


# ============================================================================
# JD Parser - Extracts structured requirements from job descriptions
# ============================================================================

class JDParser:
    """
    Parses job descriptions into structured requirements.
    
    Uses Claude to extract:
    - Role metadata (title, seniority, department)
    - All requirements categorized by type
    - Tools, technologies, methodologies
    - Language patterns to mirror
    """
    
    SYSTEM_PROMPT = """You are an expert at analyzing job descriptions.
Your task is to extract ALL requirements and details from a job description.
Be thorough - capture every distinct requirement, responsibility, and qualification.
Do not summarize or consolidate similar items.
Return your analysis as valid JSON only."""
    
    PARSE_PROMPT = """Analyze this job description and extract ALL requirements.

JOB DESCRIPTION:
{jd_text}

Return a JSON object with this exact structure:
{{
    "role_title": "exact title from JD",
    "company": "company name",
    "department": "department/function",
    "seniority_level": "Entry/Mid/Senior/Lead/Manager/Director",
    "requirements": [
        {{
            "text": "exact requirement text",
            "category": "Technical/Leadership/Communication/Domain/etc",
            "priority": "MUST-HAVE/SHOULD-HAVE/NICE-TO-HAVE",
            "keywords": ["key", "terms", "to", "mirror"]
        }}
    ],
    "total_requirements": number,
    "tools_mentioned": ["tool1", "tool2"],
    "methodologies": ["Agile", "Scrum", etc],
    "certifications_mentioned": ["cert1", "cert2"],
    "key_verbs": ["led", "developed", "managed"],
    "key_terms": ["domain", "specific", "terms"]
}}

Extract EVERY distinct requirement. Do not summarize. Be thorough."""
    
    def __init__(self, llm_client: ClaudeLLMClient = None):
        """Initialize with Claude LLM client."""
        self.llm_client = llm_client
        self.settings = get_settings()
    
    def parse(self, job: JobListing) -> JDAnalysis:
        """
        Parse a job listing into structured analysis.
        
        Args:
            job: Job listing with description
            
        Returns:
            Structured JD analysis
        """
        if not job.description:
            logger.warning(f"No description for {job.company} - {job.title}")
            return self._create_minimal_analysis(job)
        
        if not self.llm_client:
            logger.warning("No LLM client, returning minimal analysis")
            return self._create_minimal_analysis(job)
        
        prompt = self.PARSE_PROMPT.format(jd_text=job.description[:8000])
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.1,
                json_output=True
            )
            
            data = response.to_json()
            if data is None:
                logger.error("Failed to parse JSON from JD analysis")
                return self._create_minimal_analysis(job)
            
            # Convert to JDAnalysis
            requirements = [
                JDRequirement(**req) for req in data.get('requirements', [])
            ]
            
            return JDAnalysis(
                role_title=data.get('role_title', job.title),
                company=data.get('company', job.company),
                department=data.get('department'),
                seniority_level=data.get('seniority_level', 'Mid'),
                requirements=requirements,
                total_requirements=data.get('total_requirements', len(requirements)),
                tools_mentioned=data.get('tools_mentioned', []),
                methodologies=data.get('methodologies', []),
                certifications_mentioned=data.get('certifications_mentioned', []),
                key_verbs=data.get('key_verbs', []),
                key_terms=data.get('key_terms', [])
            )
            
        except Exception as e:
            logger.error(f"JD parsing failed: {e}")
            return self._create_minimal_analysis(job)
    
    def _create_minimal_analysis(self, job: JobListing) -> JDAnalysis:
        """Create minimal analysis when LLM is unavailable."""
        return JDAnalysis(
            role_title=job.title,
            company=job.company,
            seniority_level="Mid",
            requirements=[],
            total_requirements=0
        )


# ============================================================================
# CV Generator - Creates tailored CVs using Master CV Prompt
# ============================================================================

class CVGenerator:
    """
    Generates tailored CVs using the Master CV Optimization Framework.
    
    Key principles:
    1. Reverse engineer JD requirements into CV content
    2. Fabricate realistic project narratives (authorized)
    3. Preserve real facts (employers, titles, dates, education)
    4. Apply company-specific constraints (Meta, Amazon, Google)
    """
    
    SYSTEM_PROMPT = """You are an expert CV writer executing the Master CV Optimization Framework v3.0.

CONSTRAINTS:
- Do NOT fabricate: employer names, job titles, dates, education, certifications
- You MAY fabricate: project descriptions, metrics, tools used (within constraints)
- Company-specific rules:
  - Meta/Facebook: Use "internal systems" not AWS/GCP
  - Amazon: Use AWS services only  
  - Google: Use GCP services only

Your goal is to generate a CV optimized for ATS systems with 90%+ relevance score.
Return your output as valid JSON only."""
    
    GENERATION_PROMPT = """Generate a tailored CV for this role.

BASE CV (candidate's actual background):
{base_cv}

JOB DESCRIPTION ANALYSIS:
{jd_analysis}

Generate a tailored CV optimized for this role. Target 90%+ ATS relevance.

Return JSON with this exact structure:
{{
    "candidate_name": "from CV",
    "contact_info": {{"email": "", "phone": "", "linkedin": ""}},
    "experiences": [
        {{
            "company": "real company from CV",
            "title": "real title from CV",
            "start_date": "YYYY-MM",
            "end_date": "YYYY-MM or Present",
            "location": "location",
            "bullets": [
                {{
                    "text": "Action + Context + Method + Quantified Outcome",
                    "addresses_requirement": "which JD requirement this addresses",
                    "metrics_included": true
                }}
            ]
        }}
    ],
    "skills_section": "Formatted skills text with JD keywords",
    "education": ["degree details"],
    "certifications": ["cert1", "cert2"],
    "ats_score": 92.5,
    "requirements_covered": 15,
    "total_requirements": 18,
    "fabricated_content": ["list of fabricated project descriptions"],
    "preserved_facts": ["employers", "titles", "dates preserved"]
}}

Use 4 bullets per role (5 max for most recent).
Mirror JD terminology and key verbs.
Include realistic, defensible metrics."""
    
    def __init__(self, llm_client: ClaudeLLMClient = None):
        """Initialize with Claude LLM client."""
        self.llm_client = llm_client
        self.settings = get_settings()
    
    def generate(self, base_cv: str, jd_analysis: JDAnalysis, 
                 job: JobListing) -> GeneratedCV:
        """
        Generate tailored CV for a specific job.
        
        Args:
            base_cv: User's original CV text
            jd_analysis: Parsed JD requirements
            job: Target job listing
            
        Returns:
            Generated CV structure
        """
        if not self.llm_client:
            raise ValueError("LLM client required for CV generation")
        
        # Format JD analysis for prompt
        jd_analysis_text = self._format_jd_analysis(jd_analysis)
        
        prompt = self.GENERATION_PROMPT.format(
            base_cv=base_cv[:10000],  # Limit context
            jd_analysis=jd_analysis_text
        )
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=self.settings.llm.temperature,
                json_output=True
            )
            
            data = response.to_json()
            if data is None:
                raise ValueError(f"Failed to parse JSON from CV generation: {response.content[:200]}")
            
            # Convert to GeneratedCV
            experiences = []
            for exp in data.get('experiences', []):
                bullets = [
                    CVBulletPoint(
                        text=b.get('text', ''),
                        addresses_requirement=b.get('addresses_requirement'),
                        metrics_included=b.get('metrics_included', False)
                    )
                    for b in exp.get('bullets', [])
                ]
                
                experiences.append(CVExperience(
                    company=exp.get('company', ''),
                    title=exp.get('title', ''),
                    department=exp.get('department'),
                    start_date=exp.get('start_date', ''),
                    end_date=exp.get('end_date'),
                    location=exp.get('location'),
                    bullets=bullets
                ))
            
            return GeneratedCV(
                target_company=job.company,
                target_role=job.title,
                candidate_name=data.get('candidate_name', ''),
                contact_info=data.get('contact_info', {}),
                experiences=experiences,
                skills_section=data.get('skills_section', ''),
                education=data.get('education', []),
                certifications=data.get('certifications', []),
                ats_score=data.get('ats_score', 0),
                requirements_covered=data.get('requirements_covered', 0),
                total_requirements=data.get('total_requirements', 0),
                fabricated_content=data.get('fabricated_content', []),
                preserved_facts=data.get('preserved_facts', [])
            )
            
        except Exception as e:
            logger.error(f"CV generation failed: {e}")
            raise
    
    def _format_jd_analysis(self, analysis: JDAnalysis) -> str:
        """Format JD analysis for inclusion in prompt."""
        lines = [
            f"Role: {analysis.role_title} at {analysis.company}",
            f"Seniority: {analysis.seniority_level}",
            f"Department: {analysis.department or 'Not specified'}",
            "",
            "REQUIREMENTS:"
        ]
        
        for i, req in enumerate(analysis.requirements, 1):
            lines.append(f"{i}. [{req.priority}] {req.text}")
            if req.keywords:
                lines.append(f"   Keywords: {', '.join(req.keywords)}")
        
        lines.extend([
            "",
            f"Tools mentioned: {', '.join(analysis.tools_mentioned)}",
            f"Methodologies: {', '.join(analysis.methodologies)}",
            f"Key verbs to use: {', '.join(analysis.key_verbs)}",
            f"Key terms to mirror: {', '.join(analysis.key_terms)}"
        ])
        
        return "\n".join(lines)


# ============================================================================
# CV Critic - Validates generated CV for quality and compliance
# ============================================================================

class CVCritic:
    """
    Validates generated CVs for:
    1. Hallucination detection (invented facts)
    2. ATS compliance (keyword coverage)
    3. Constraint compliance (company-specific rules)
    4. Quality standards (metrics, action verbs)
    """
    
    SYSTEM_PROMPT = """You are a CV auditor checking for issues.
Your job is to compare a generated CV against the original and flag any problems.
Be strict about detecting fabricated employers, titles, dates, or education.
Return your analysis as valid JSON only."""
    
    CRITIC_PROMPT = """You are auditing a generated CV for quality issues.

ORIGINAL CV (source of truth):
{original_cv}

GENERATED CV:
{generated_cv}

JOB REQUIREMENTS:
{requirements}

Check for these issues:
1. HALLUCINATIONS: Did the generated CV invent employers, titles, dates, or education not in the original?
2. ATS COVERAGE: What percentage of MUST-HAVE requirements are addressed?
3. CONSTRAINT VIOLATIONS: For Meta/Amazon/Google employers, are there prohibited tool mentions?
4. QUALITY: Do bullets have metrics? Do they use action verbs?

Return JSON with this exact structure:
{{
    "passes_validation": true/false,
    "hallucinations_found": ["list any invented facts"],
    "constraint_violations": ["list any prohibited terms"],
    "ats_score": 85.5,
    "must_have_coverage": 0.90,
    "should_have_coverage": 0.75,
    "quality_issues": ["bullets without metrics", "weak verbs"],
    "improvement_suggestions": ["specific suggestions"],
    "overall_verdict": "PASS/FAIL/NEEDS_REVISION"
}}"""
    
    def __init__(self, llm_client: ClaudeLLMClient = None):
        """Initialize with Claude LLM client."""
        self.llm_client = llm_client
        self.settings = get_settings()
    
    def validate(self, original_cv: str, generated_cv: GeneratedCV,
                 jd_analysis: JDAnalysis) -> Tuple[bool, Dict]:
        """
        Validate a generated CV.
        
        Args:
            original_cv: Original CV text
            generated_cv: Generated CV structure
            jd_analysis: JD requirements
            
        Returns:
            Tuple of (passes_validation, validation_report)
        """
        if not self.llm_client:
            logger.warning("No LLM client, skipping validation")
            return True, {"verdict": "SKIPPED", "reason": "No LLM client"}
        
        # Format generated CV for comparison
        generated_text = self._format_generated_cv(generated_cv)
        
        # Format requirements
        requirements_text = "\n".join([
            f"[{req.priority}] {req.text}"
            for req in jd_analysis.requirements
        ])
        
        prompt = self.CRITIC_PROMPT.format(
            original_cv=original_cv[:6000],
            generated_cv=generated_text[:6000],
            requirements=requirements_text[:2000]
        )
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.1,
                json_output=True
            )
            
            report = response.to_json()
            if report is None:
                return False, {"verdict": "ERROR", "error": "Failed to parse validation JSON"}
            
            passes = report.get('passes_validation', False)
            
            # Auto-fail on hallucinations
            if report.get('hallucinations_found'):
                passes = False
                report['overall_verdict'] = 'FAIL - HALLUCINATIONS'
            
            # Auto-fail on low MUST-HAVE coverage
            if report.get('must_have_coverage', 0) < 0.80:
                passes = False
                report['overall_verdict'] = 'FAIL - LOW COVERAGE'
            
            logger.info(f"CV validation: {report.get('overall_verdict')}")
            return passes, report
            
        except Exception as e:
            logger.error(f"CV validation failed: {e}")
            return False, {"verdict": "ERROR", "error": str(e)}
    
    def _format_generated_cv(self, cv: GeneratedCV) -> str:
        """Format generated CV for validation."""
        lines = [
            cv.candidate_name,
            json.dumps(cv.contact_info),
            ""
        ]
        
        for exp in cv.experiences:
            lines.extend([
                f"{exp.title} - {exp.company} | {exp.start_date} - {exp.end_date or 'Present'}",
            ])
            for bullet in exp.bullets:
                lines.append(f"* {bullet.text}")
            lines.append("")
        
        lines.extend([
            "SKILLS:",
            cv.skills_section,
            "",
            "EDUCATION:",
            "\n".join(cv.education),
            "",
            "CERTIFICATIONS:",
            "\n".join(cv.certifications)
        ])
        
        return "\n".join(lines)


# ============================================================================
# Main CV Architect Agent
# ============================================================================

class CVArchitectAgent:
    """
    Main agent that orchestrates CV generation with validation.
    
    Implements: Generate -> Critique -> Fix -> (User Approval)
    Uses Claude (Anthropic) for all LLM operations.
    """
    
    def __init__(self, llm_client: ClaudeLLMClient = None, use_mock: bool = False):
        """
        Initialize with Claude LLM client.
        
        Args:
            llm_client: Pre-configured Claude client (optional)
            use_mock: If True, use mock client for testing without API calls
        """
        if llm_client is None and not use_mock:
            try:
                llm_client = get_llm_client(provider="claude")
            except ValueError as e:
                logger.warning(f"Could not initialize Claude client: {e}")
                llm_client = get_llm_client(use_mock=True)
        elif use_mock:
            llm_client = get_llm_client(use_mock=True)
        
        self.llm_client = llm_client
        self.settings = get_settings()
        
        # Initialize components with the same client
        self.jd_parser = JDParser(llm_client)
        self.generator = CVGenerator(llm_client)
        self.critic = CVCritic(llm_client)
        
        logger.info(f"CVArchitectAgent initialized with: {type(llm_client).__name__}")
    
    def create_tailored_cv(self, base_cv: str, job: JobListing,
                           max_iterations: int = 3) -> Tuple[GeneratedCV, Dict]:
        """
        Create a tailored CV with validation loop.
        
        Args:
            base_cv: User's original CV text
            job: Target job listing
            max_iterations: Max attempts to pass validation
            
        Returns:
            Tuple of (generated_cv, validation_report)
        """
        logger.info(f"Creating tailored CV for {job.company} - {job.title}")
        
        # Step 1: Parse JD
        logger.info("Parsing job description...")
        jd_analysis = self.jd_parser.parse(job)
        logger.info(f"Extracted {jd_analysis.total_requirements} requirements")
        
        # Step 2: Generate -> Validate loop
        generated_cv = None
        report = {}
        
        for iteration in range(max_iterations):
            logger.info(f"Generation attempt {iteration + 1}/{max_iterations}")
            
            # Generate CV
            generated_cv = self.generator.generate(base_cv, jd_analysis, job)
            logger.info(f"Generated CV with ATS score: {generated_cv.ats_score}")
            
            # Validate
            passes, report = self.critic.validate(base_cv, generated_cv, jd_analysis)
            
            if passes:
                logger.info("CV passed validation")
                return generated_cv, report
            
            logger.warning(f"CV failed validation: {report.get('overall_verdict')}")
            
            # Could use improvement_suggestions for next iteration
            # For now, just retry with fresh generation
        
        # Return best effort after max iterations
        logger.warning(f"CV did not pass after {max_iterations} attempts")
        return generated_cv, report
    
    def should_auto_approve(self, generated_cv: GeneratedCV, 
                            validation_report: Dict) -> bool:
        """
        Determine if CV can be auto-approved in autonomous mode.
        
        Args:
            generated_cv: The generated CV
            validation_report: Validation results
            
        Returns:
            True if CV meets auto-approval threshold
        """
        threshold = self.settings.workflow.ats_score_threshold
        
        # Must pass validation
        if validation_report.get('overall_verdict') not in ['PASS', 'PASS - MINOR ISSUES']:
            return False
        
        # Must meet ATS threshold
        if generated_cv.ats_score < threshold * 100:
            return False
        
        # Must have no hallucinations
        if validation_report.get('hallucinations_found'):
            return False
        
        return True


# ============================================================================
# PDF Renderer (Placeholder - use WeasyPrint or LaTeX in production)
# ============================================================================

class CVRenderer:
    """
    Renders generated CV to PDF.
    
    Uses deterministic rendering (not LLM) to ensure consistent formatting.
    """
    
    def render_markdown(self, cv: GeneratedCV) -> str:
        """Render CV as Markdown."""
        lines = [
            f"# {cv.candidate_name}",
            ""
        ]
        
        # Contact info
        contact = cv.contact_info
        if contact:
            lines.append(" | ".join([
                contact.get('email', ''),
                contact.get('phone', ''),
                contact.get('linkedin', '')
            ]))
            lines.append("")
        
        # Experience
        for exp in cv.experiences:
            lines.extend([
                f"## {exp.title}",
                f"**{exp.company}** | {exp.start_date} - {exp.end_date or 'Present'}",
                ""
            ])
            for bullet in exp.bullets:
                lines.append(f"- {bullet.text}")
            lines.append("")
        
        # Skills
        lines.extend([
            "## Skills",
            cv.skills_section,
            ""
        ])
        
        # Education
        lines.extend([
            "## Education",
            *cv.education,
            ""
        ])
        
        # Certifications
        if cv.certifications:
            lines.extend([
                "## Certifications",
                *cv.certifications
            ])
        
        return "\n".join(lines)
    
    def render_pdf(self, cv: GeneratedCV, output_path: str) -> str:
        """
        Render CV as PDF.
        
        Note: In production, use WeasyPrint or LaTeX for professional output.
        """
        # Placeholder - would use WeasyPrint here
        markdown = self.render_markdown(cv)
        
        # For now, just save markdown
        md_path = output_path.replace('.pdf', '.md')
        with open(md_path, 'w') as f:
            f.write(markdown)
        
        logger.info(f"Saved CV markdown to {md_path}")
        return md_path
