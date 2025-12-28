"""
JobPilot CLI Demo

Interactive command-line interface demonstrating the full workflow.
Run with: python -m jobpilot.examples.cli_demo

This shows how all components work together:
1. Chat-based preference collection
2. Job discovery using existing scrapers
3. CV generation workflow (mocked without LLM)
4. Form filling with interrupt pattern
"""

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import logging
from datetime import datetime

from jobpilot.api.chat_handler import ChatHandler
from jobpilot.services.orchestrator import WorkflowOrchestrator
from jobpilot.core.schemas import ChatMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_message(msg: ChatMessage):
    """Pretty print a chat message."""
    
    role_colors = {
        'user': '\033[94m',  # Blue
        'agent': '\033[92m',  # Green
        'system': '\033[93m',  # Yellow
    }
    
    reset = '\033[0m'
    bold = '\033[1m'
    
    color = role_colors.get(msg.role, '')
    role_display = msg.role.upper()
    
    print(f"\n{color}{bold}[{role_display}]{reset}")
    print(msg.content)
    
    if msg.options:
        print(f"\n{color}Options: {', '.join(msg.options)}{reset}")


def run_interactive_demo():
    """Run an interactive CLI demo."""
    
    print("\n" + "=" * 60)
    print("           JOBPILOT - Interactive Demo")
    print("=" * 60)
    print("\nThis demo shows the chat-based job application workflow.")
    print("Type 'quit' or 'exit' at any time to stop.\n")
    
    # Initialize
    orchestrator = WorkflowOrchestrator()
    chat_handler = ChatHandler(orchestrator)
    
    user_id = "demo_user"
    
    # Start conversation
    responses = chat_handler.process_message(user_id, "hi")
    for msg in responses:
        print_message(msg)
    
    # Main loop
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            # Process message
            responses = chat_handler.process_message(user_id, user_input)
            
            for msg in responses:
                print_message(msg)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nError: {e}")


def run_automated_demo():
    """Run an automated demo showing the full workflow."""
    
    print("\n" + "=" * 60)
    print("           JOBPILOT - Automated Demo")
    print("=" * 60)
    print("\nThis demo simulates a complete job application workflow.\n")
    
    # Initialize
    orchestrator = WorkflowOrchestrator()
    chat_handler = ChatHandler(orchestrator)
    
    user_id = "demo_user"
    
    # Simulated conversation
    messages = [
        "hi",
        "Google, Meta, Amazon",  # Companies
        "Data Engineer, ML Engineer",  # Titles
        "remote",  # Location
        "supervised",  # Mode
        "yes",  # Confirm
        "1, 2",  # Select jobs (if any found)
        "yes",  # Approve CV
    ]
    
    for i, user_msg in enumerate(messages):
        print(f"\n{'=' * 40}")
        print(f"Step {i + 1}: Sending '{user_msg}'")
        print('=' * 40)
        
        # Show user message
        print_message(ChatMessage(role="user", content=user_msg))
        
        # Process
        responses = chat_handler.process_message(user_id, user_msg)
        
        for msg in responses:
            print_message(msg)
        
        # Check if workflow is complete
        ctx = chat_handler.get_conversation(user_id)
        if ctx.workflow_id:
            workflow = orchestrator.get_workflow(ctx.workflow_id)
            if workflow and workflow.current_state.value in ['complete', 'failed']:
                print("\n\nWorkflow finished!")
                break
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def run_discovery_only_demo():
    """Demo showing just the discovery agent with your existing scrapers."""
    
    print("\n" + "=" * 60)
    print("           DISCOVERY AGENT DEMO")
    print("=" * 60)
    print("\nThis demo uses your existing scraper infrastructure.\n")
    
    from jobpilot.agents.discovery.discovery_agent import DiscoveryAgent
    from jobpilot.core.schemas import JobSearchPreferences, LocationType
    
    # Initialize agent
    print("Initializing Discovery Agent...")
    agent = DiscoveryAgent(config_path="data/company_career_urls.json")
    print(f"Loaded {len(agent.companies)} companies from config.\n")
    
    # Create preferences
    prefs = JobSearchPreferences(
        companies=[],  # Search all
        job_titles=["Data Engineer", "Machine Learning Engineer"],
        location_type=LocationType.REMOTE,
        countries=["United States"]
    )
    
    print(f"Searching for: {prefs.job_titles}")
    print(f"Location: {prefs.location_type.value}")
    print(f"Companies: {'All' if not prefs.companies else prefs.companies}\n")
    
    # Run search
    print("Running API-based search (Greenhouse, Lever)...")
    batch = agent.search(prefs)
    
    print(f"\n{'=' * 40}")
    print(f"RESULTS: Found {len(batch.jobs)} jobs from direct APIs")
    print('=' * 40)
    
    if batch.jobs:
        print("\nTop matches:")
        for i, job in enumerate(batch.jobs[:10], 1):
            print(f"\n{i}. {job.company}")
            print(f"   Title: {job.title}")
            print(f"   Location: {job.location} ({job.location_type.value})")
            print(f"   Source: {job.source}")
            print(f"   URL: {job.job_url[:60]}...")
    
    # Show MCP URLs
    mcp_urls = agent.get_pending_mcp_urls()
    
    print(f"\n{'=' * 40}")
    print("URLs pending for Bright Data MCP scraping:")
    print('=' * 40)
    print(f"Workday companies: {len(mcp_urls.get('workday', []))}")
    print(f"Custom ATS: {len(mcp_urls.get('custom', []))}")
    print(f"Aggregators (Indeed, LinkedIn): {len(mcp_urls.get('aggregator', []))}")
    
    if mcp_urls.get('aggregator'):
        print("\nSample aggregator URLs:")
        for url_info in mcp_urls['aggregator'][:3]:
            print(f"  - {url_info['source']}: {url_info['url'][:60]}...")
    
    print("\n" + "=" * 60)
    print("To scrape these URLs, use Bright Data MCP tools:")
    print("  mcp_Bright_Data_scrape_batch(urls)")
    print("=" * 60)


def run_vault_demo():
    """Demo showing the vault and knowledge base."""
    
    print("\n" + "=" * 60)
    print("           VAULT & KNOWLEDGE BASE DEMO")
    print("=" * 60)
    
    from jobpilot.agents.vault.vault import (
        EncryptionManager, CredentialVault, KnowledgeBase
    )
    
    # Encryption demo
    print("\n1. ENCRYPTION")
    print("-" * 40)
    
    encryption = EncryptionManager(master_key="demo_key_12345")
    
    test_password = "my_secret_password"
    encrypted = encryption.encrypt(test_password)
    decrypted = encryption.decrypt(encrypted)
    
    print(f"Original: {test_password}")
    print(f"Encrypted: {encrypted[:50]}...")
    print(f"Decrypted: {decrypted}")
    print(f"Match: {test_password == decrypted}")
    
    # Knowledge Base demo
    print("\n2. KNOWLEDGE BASE")
    print("-" * 40)
    
    kb = KnowledgeBase(storage_path=".demo_knowledge")
    
    # Add some entries
    entries = [
        ("years of python experience", "5 years"),
        ("authorized to work in the united states", "Yes"),
        ("require visa sponsorship", "No"),
        ("notice period", "2 weeks"),
        ("expected salary", "$150,000 - $180,000"),
    ]
    
    print("Adding knowledge entries...")
    for question, answer in entries:
        kb.add_entry("demo_user", question, answer, source="demo")
        print(f"  Added: '{question}' -> '{answer}'")
    
    # Test lookups
    print("\nTesting lookups:")
    
    test_questions = [
        "How many years of Python experience do you have?",
        "Are you authorized to work in the US?",
        "Do you need visa sponsorship?",
        "What is your notice period?",
        "What are your salary expectations?",
        "What is your favorite color?"  # Unknown
    ]
    
    for question in test_questions:
        result = kb.find_answer("demo_user", question)
        if result:
            print(f"\n  Q: {question}")
            print(f"  A: {result['answer']} (confidence: {result['confidence']:.2f})")
        else:
            print(f"\n  Q: {question}")
            print(f"  A: [No answer found - would ask user]")
    
    # Cleanup
    if os.path.exists(".demo_knowledge"):
        os.remove(".demo_knowledge")
    
    print("\n" + "=" * 60)


def run_form_analysis_demo():
    """Demo showing form field detection."""
    
    print("\n" + "=" * 60)
    print("           FORM ANALYSIS DEMO")
    print("=" * 60)
    
    from jobpilot.agents.form_filler.form_filler import DOMAnalyzer
    
    analyzer = DOMAnalyzer()
    
    # Sample Workday-like form
    sample_html = """
    <form id="job-application">
        <div class="form-group">
            <label for="firstName">First Name <span class="required">*</span></label>
            <input type="text" id="firstName" name="first_name" required>
        </div>
        
        <div class="form-group">
            <label for="lastName">Last Name <span class="required">*</span></label>
            <input type="text" id="lastName" name="last_name" required>
        </div>
        
        <div class="form-group">
            <label for="email">Email Address <span class="required">*</span></label>
            <input type="email" id="email" name="email" required>
        </div>
        
        <div class="form-group">
            <label for="phone">Phone Number</label>
            <input type="tel" id="phone" name="phone">
        </div>
        
        <div class="form-group">
            <label for="linkedin">LinkedIn Profile</label>
            <input type="url" id="linkedin" name="linkedin" placeholder="https://linkedin.com/in/...">
        </div>
        
        <div class="form-group">
            <label for="experience">Years of Experience <span class="required">*</span></label>
            <select id="experience" name="experience" required>
                <option value="">Select...</option>
                <option value="0-2">0-2 years</option>
                <option value="3-5">3-5 years</option>
                <option value="5-10">5-10 years</option>
                <option value="10+">10+ years</option>
            </select>
        </div>
        
        <div class="form-group">
            <label>Are you authorized to work in the United States? <span class="required">*</span></label>
            <input type="radio" name="work_auth" value="yes" required> Yes
            <input type="radio" name="work_auth" value="no"> No
        </div>
        
        <div class="form-group">
            <label>Will you now or in the future require sponsorship?</label>
            <input type="radio" name="sponsorship" value="yes"> Yes
            <input type="radio" name="sponsorship" value="no"> No
        </div>
        
        <div class="form-group">
            <label for="resume">Upload Resume <span class="required">*</span></label>
            <input type="file" id="resume" name="resume" accept=".pdf,.doc,.docx" required>
        </div>
        
        <div class="form-group">
            <label for="coverLetter">Cover Letter (Optional)</label>
            <textarea id="coverLetter" name="cover_letter" rows="5"></textarea>
        </div>
    </form>
    """
    
    print("Analyzing sample job application form...")
    print("-" * 40)
    
    fields = analyzer.analyze_form(sample_html, "https://example.com/apply")
    
    print(f"\nDetected {len(fields)} form fields:\n")
    
    for i, field in enumerate(fields, 1):
        required_marker = " *" if field.required else ""
        
        print(f"{i}. {field.label}{required_marker}")
        print(f"   Type: {field.field_type.value}")
        print(f"   Selector: {field.selector}")
        
        if field.question_category:
            print(f"   Category: {field.question_category}")
        
        if field.options:
            print(f"   Options: {field.options[:3]}...")
        
        print()
    
    # Show which fields can be auto-filled
    print("-" * 40)
    print("AUTO-FILL ANALYSIS:")
    print("-" * 40)
    
    auto_fillable = ['first_name', 'last_name', 'email', 'phone', 'linkedin',
                    'work_authorization', 'sponsorship']
    
    for field in fields:
        if field.question_category in auto_fillable:
            print(f"  [AUTO] {field.label} -> From profile/knowledge base")
        elif field.field_type.value == 'file':
            print(f"  [FILE] {field.label} -> Will upload CV")
        elif field.question_category:
            print(f"  [KB?]  {field.label} -> Check knowledge base")
        else:
            print(f"  [ASK]  {field.label} -> May need user input")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    
    print("\n" + "=" * 60)
    print("           JOBPILOT DEMO MENU")
    print("=" * 60)
    print("""
Choose a demo:

1. Interactive Chat Demo (full workflow)
2. Automated Workflow Demo
3. Discovery Agent Demo (uses your scrapers)
4. Vault & Knowledge Base Demo
5. Form Analysis Demo

Enter 'q' to quit.
""")
    
    demos = {
        '1': run_interactive_demo,
        '2': run_automated_demo,
        '3': run_discovery_only_demo,
        '4': run_vault_demo,
        '5': run_form_analysis_demo,
    }
    
    while True:
        choice = input("Select demo (1-5): ").strip()
        
        if choice.lower() in ['q', 'quit', 'exit']:
            break
        
        if choice in demos:
            demos[choice]()
            print("\n" + "-" * 60)
            print("Demo complete. Select another or 'q' to quit.\n")
        else:
            print("Invalid choice. Please enter 1-5 or 'q'.")


if __name__ == "__main__":
    main()

