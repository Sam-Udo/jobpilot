"""
Test the JobPilot workflow components.

Run with: python -m pytest jobpilot/tests/test_workflow.py -v
Or directly: python jobpilot/tests/test_workflow.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def test_discovery_agent():
    """Test the Discovery Agent with your existing scraper config."""
    from jobpilot.agents.discovery.discovery_agent import DiscoveryAgent
    from jobpilot.core.schemas import JobSearchPreferences, LocationType
    
    logger.info("=" * 60)
    logger.info("TEST: Discovery Agent")
    logger.info("=" * 60)
    
    # Initialize agent (uses your existing company_career_urls.json)
    agent = DiscoveryAgent(config_path="data/company_career_urls.json")
    
    # Create preferences
    prefs = JobSearchPreferences(
        companies=["Blue Origin", "SpaceX"],  # Companies with Greenhouse
        job_titles=["Data Engineer", "Machine Learning Engineer"],
        location_type=LocationType.ANY,
        countries=["United States"]
    )
    
    # Search (API only - no MCP needed for Greenhouse/Lever)
    logger.info(f"Searching for jobs: {prefs.job_titles}")
    batch = agent.search(prefs)
    
    logger.info(f"Found {len(batch.jobs)} jobs from direct APIs")
    for job in batch.jobs[:5]:
        logger.info(f"  - {job.company}: {job.title} ({job.location})")
    
    # Get URLs for MCP scraping
    mcp_urls = agent.get_pending_mcp_urls()
    logger.info(f"\nURLs pending for MCP scraping:")
    logger.info(f"  Workday: {len(mcp_urls.get('workday', []))} URLs")
    logger.info(f"  Custom: {len(mcp_urls.get('custom', []))} URLs")
    logger.info(f"  Aggregator: {len(mcp_urls.get('aggregator', []))} URLs")
    
    return batch


def test_vault():
    """Test the credential vault and knowledge base."""
    from jobpilot.agents.vault.vault import (
        EncryptionManager, CredentialVault, KnowledgeBase
    )
    
    logger.info("=" * 60)
    logger.info("TEST: Vault & Knowledge Base")
    logger.info("=" * 60)
    
    # Test encryption
    encryption = EncryptionManager(master_key="test_key_12345")
    
    encrypted = encryption.encrypt("my_password")
    decrypted = encryption.decrypt(encrypted)
    
    assert decrypted == "my_password", "Encryption roundtrip failed"
    logger.info("Encryption: PASS")
    
    # Test knowledge base
    kb = KnowledgeBase(storage_path=".test_knowledge")
    
    # Add entries
    kb.add_entry(
        user_id="test_user",
        question_pattern="years of python experience",
        answer="5 years",
        category="experience_years"
    )
    
    kb.add_entry(
        user_id="test_user",
        question_pattern="authorized to work in the united states",
        answer="Yes",
        category="work_authorization"
    )
    
    # Test lookup
    result = kb.find_answer("test_user", "How many years of Python experience do you have?")
    assert result is not None, "Knowledge lookup failed"
    logger.info(f"Knowledge lookup: Found '{result['answer']}' with confidence {result['confidence']:.2f}")
    
    result = kb.find_answer("test_user", "Are you authorized to work in the US?")
    assert result is not None, "Knowledge lookup failed"
    logger.info(f"Knowledge lookup: Found '{result['answer']}' with confidence {result['confidence']:.2f}")
    
    # Cleanup test file
    if os.path.exists(".test_knowledge"):
        os.remove(".test_knowledge")
    
    logger.info("Knowledge Base: PASS")
    return True


def test_cv_schemas():
    """Test the CV-related schemas."""
    from jobpilot.core.schemas import (
        JobListing, JDAnalysis, JDRequirement, 
        GeneratedCV, CVExperience, CVBulletPoint,
        LocationType, ATSType
    )
    
    logger.info("=" * 60)
    logger.info("TEST: CV Schemas")
    logger.info("=" * 60)
    
    # Create a job listing
    job = JobListing(
        company="Acme Corp",
        title="Senior Data Engineer",
        location="San Francisco, CA",
        location_type=LocationType.HYBRID,
        job_url="https://acme.com/jobs/123",
        description="We are looking for a Senior Data Engineer...",
        source="Greenhouse API",
        ats_type=ATSType.GREENHOUSE,
        relevance_score=85.0
    )
    
    logger.info(f"Job: {job.company} - {job.title}")
    logger.info(f"Location: {job.location} ({job.location_type.value})")
    
    # Create JD requirements
    req = JDRequirement(
        text="5+ years experience with Python",
        category="Technical",
        priority="MUST-HAVE",
        keywords=["Python", "experience"]
    )
    
    jd = JDAnalysis(
        role_title="Senior Data Engineer",
        company="Acme Corp",
        seniority_level="Senior",
        requirements=[req],
        total_requirements=1,
        tools_mentioned=["Python", "Spark", "AWS"],
        key_verbs=["build", "design", "lead"]
    )
    
    logger.info(f"JD Analysis: {jd.total_requirements} requirements, "
               f"{len(jd.tools_mentioned)} tools mentioned")
    
    # Create generated CV
    bullet = CVBulletPoint(
        text="Led migration of 50TB data warehouse to Snowflake, reducing query times by 70%",
        addresses_requirement="Data warehouse experience",
        metrics_included=True
    )
    
    exp = CVExperience(
        company="Previous Corp",
        title="Data Engineer",
        start_date="2020-01",
        end_date="2023-12",
        bullets=[bullet]
    )
    
    cv = GeneratedCV(
        target_company="Acme Corp",
        target_role="Senior Data Engineer",
        candidate_name="John Doe",
        experiences=[exp],
        skills_section="Python, SQL, Spark, AWS, Snowflake",
        education=["BS Computer Science, Stanford"],
        ats_score=92.5,
        requirements_covered=15,
        total_requirements=18
    )
    
    logger.info(f"Generated CV: ATS Score {cv.ats_score}%, "
               f"{cv.requirements_covered}/{cv.total_requirements} requirements covered")
    
    logger.info("CV Schemas: PASS")
    return True


def test_workflow_state_machine():
    """Test the workflow state machine."""
    from jobpilot.services.orchestrator import (
        WorkflowContext, WorkflowStateMachine
    )
    from jobpilot.core.schemas import (
        WorkflowState, JobSearchPreferences, ApplicationMode
    )
    
    logger.info("=" * 60)
    logger.info("TEST: Workflow State Machine")
    logger.info("=" * 60)
    
    # Create context
    prefs = JobSearchPreferences(
        job_titles=["Data Engineer"],
        application_mode=ApplicationMode.SUPERVISED
    )
    
    ctx = WorkflowContext(
        workflow_id="test-123",
        user_id="user-456",
        preferences=prefs
    )
    
    # Test valid transitions
    assert ctx.current_state == WorkflowState.INIT
    
    result = WorkflowStateMachine.transition(ctx, WorkflowState.SEARCHING)
    assert result == True
    assert ctx.current_state == WorkflowState.SEARCHING
    
    result = WorkflowStateMachine.transition(ctx, WorkflowState.JOBS_FOUND)
    assert result == True
    
    result = WorkflowStateMachine.transition(ctx, WorkflowState.JOBS_SELECTED)
    assert result == True
    
    # Test invalid transition
    result = WorkflowStateMachine.transition(ctx, WorkflowState.COMPLETE)
    assert result == False  # Can't go from JOBS_SELECTED to COMPLETE
    
    logger.info("State machine transitions: PASS")
    
    # Test status generation
    status = ctx.to_status()
    logger.info(f"Workflow status: {status.state.value}")
    
    return True


def test_form_filler_field_detection():
    """Test form field detection."""
    from jobpilot.agents.form_filler.form_filler import DOMAnalyzer, FieldType
    
    logger.info("=" * 60)
    logger.info("TEST: Form Field Detection")
    logger.info("=" * 60)
    
    # Sample HTML form
    html = '''
    <form>
        <label for="fname">First Name *</label>
        <input type="text" id="fname" name="first_name" required>
        
        <label for="email">Email Address</label>
        <input type="email" id="email" name="email">
        
        <label for="exp">Years of Experience</label>
        <select id="exp" name="experience">
            <option value="0-2">0-2 years</option>
            <option value="3-5">3-5 years</option>
            <option value="5+">5+ years</option>
        </select>
        
        <label for="auth">Are you authorized to work in the US?</label>
        <input type="radio" name="work_auth" value="yes"> Yes
        <input type="radio" name="work_auth" value="no"> No
    </form>
    '''
    
    analyzer = DOMAnalyzer()
    fields = analyzer.analyze_form(html, "https://example.com/apply")
    
    logger.info(f"Detected {len(fields)} form fields:")
    for field in fields:
        logger.info(f"  - {field.label}: {field.field_type.value} "
                   f"(required={field.required}, category={field.question_category})")
    
    assert len(fields) >= 3, f"Expected at least 3 fields, got {len(fields)}"
    logger.info("Form field detection: PASS")
    
    return True


def run_all_tests():
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("JOBPILOT TEST SUITE")
    logger.info("=" * 60 + "\n")
    
    tests = [
        ("Vault & Knowledge Base", test_vault),
        ("CV Schemas", test_cv_schemas),
        ("Workflow State Machine", test_workflow_state_machine),
        ("Form Field Detection", test_form_filler_field_detection),
        ("Discovery Agent", test_discovery_agent),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "PASS"))
        except Exception as e:
            logger.error(f"Test {name} FAILED: {e}")
            results.append((name, f"FAIL: {e}"))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for name, result in results:
        status = "PASS" if result == "PASS" else "FAIL"
        logger.info(f"  [{status}] {name}")
    
    passed = sum(1 for _, r in results if r == "PASS")
    logger.info(f"\nTotal: {passed}/{len(results)} tests passed")
    
    return all(r == "PASS" for _, r in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

