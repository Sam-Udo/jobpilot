"""
Chat Handler for JobPilot

Handles conversational interactions with users:
1. Collecting job search preferences
2. Presenting job listings for selection
3. Showing generated CVs for approval
4. Requesting answers to form questions
5. Status updates and notifications

This can be used with any chat interface (CLI, web, Slack, etc.)
"""

import logging
import json
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from jobpilot.core.schemas import (
    JobSearchPreferences, JobListing, GeneratedCV,
    WorkflowState, ApplicationMode, LocationType,
    ChatMessage, HumanInputRequest
)
from jobpilot.services.orchestrator import (
    WorkflowOrchestrator, WorkflowContext, WorkflowStateMachine
)

logger = logging.getLogger(__name__)


# ============================================================================
# Conversation States
# ============================================================================

class ConversationState(str, Enum):
    """States for the conversation flow."""
    
    IDLE = "idle"
    COLLECTING_COMPANIES = "collecting_companies"
    COLLECTING_TITLES = "collecting_titles"
    COLLECTING_LOCATION = "collecting_location"
    COLLECTING_MODE = "collecting_mode"
    CONFIRMING_PREFERENCES = "confirming_preferences"
    AWAITING_JOB_SELECTION = "awaiting_job_selection"
    AWAITING_CV_APPROVAL = "awaiting_cv_approval"
    AWAITING_FORM_ANSWERS = "awaiting_form_answers"
    RUNNING = "running"


@dataclass
class ConversationContext:
    """Tracks the current conversation state and data being collected."""
    
    user_id: str
    state: ConversationState = ConversationState.IDLE
    
    # Partial preferences being collected
    partial_preferences: Dict = field(default_factory=dict)
    
    # Active workflow
    workflow_id: Optional[str] = None
    
    # Pending questions
    pending_questions: List[HumanInputRequest] = field(default_factory=list)
    
    # Current job being processed
    current_job_id: Optional[str] = None
    
    # Message history
    messages: List[ChatMessage] = field(default_factory=list)


# ============================================================================
# Response Templates
# ============================================================================

class ResponseTemplates:
    """Templates for agent responses."""
    
    WELCOME = """Hi! I'm JobPilot, your AI job application assistant.

I can help you:
- Search for jobs at specific companies
- Generate tailored CVs for each application
- Fill out application forms automatically

To get started, tell me:
1. Which companies would you like to work for? (or say "any" for all)
2. What job titles are you looking for?

Example: "I want to apply for Data Engineer roles at Google, Meta, and Amazon"
"""
    
    ASK_COMPANIES = """Which companies would you like to target?

You can:
- List specific companies: "Google, Meta, Amazon"
- Say "any" to search all 243 companies in our database
- Say "tech giants" for FAANG companies
"""
    
    ASK_TITLES = """What job titles are you looking for?

Examples:
- Data Engineer
- Machine Learning Engineer
- DevOps Engineer
- Data Scientist

You can list multiple: "Data Engineer, ML Engineer, Platform Engineer"
"""
    
    ASK_LOCATION = """What's your location preference?

- "remote" - Remote only
- "hybrid" - Hybrid positions
- "onsite" - Office-based
- "any" - All types
"""
    
    ASK_MODE = """How would you like me to handle applications?

**Supervised** (recommended): I'll ask for your approval before each CV and application

**Autonomous**: I'll automatically apply if the CV scores 85%+ on ATS matching (you can still set limits)

Which mode? (supervised/autonomous)
"""
    
    CONFIRM_PREFERENCES = """Here's what I understood:

**Companies**: {companies}
**Job Titles**: {titles}
**Location**: {location}
**Mode**: {mode}

Should I start searching? (yes/no)
"""
    
    SEARCHING = "Searching for jobs... This may take a moment."
    
    JOBS_FOUND = """Found {count} matching jobs!

Here are the top matches:
{job_list}

Which jobs would you like to apply to?
- Enter numbers: "1, 3, 5"
- Or say "all" to apply to all
- Or "top 5" for the first 5
"""
    
    NO_JOBS_FOUND = """I couldn't find any jobs matching your criteria from direct APIs.

I have URLs ready for deeper scraping:
- {workday_count} Workday company pages
- {custom_count} Custom career sites
- {aggregator_count} Indeed/LinkedIn searches

Would you like me to continue with MCP scraping? (yes/no)
"""
    
    CV_READY = """I've generated a tailored CV for:
**{company}** - {title}

**ATS Score**: {score}%
**Requirements Covered**: {covered}/{total}

{cv_preview}

Do you approve this CV? (yes/no/edit)
"""
    
    FORM_QUESTION = """The application for {company} has a question I need help with:

**Question**: {question}
{options_text}

Please provide your answer:
"""
    
    APPLICATION_SUBMITTED = """Application submitted for {company} - {title}!

Confirmation: {confirmation}

{next_action}
"""
    
    WORKFLOW_COMPLETE = """All done! Here's your summary:

**Applications Submitted**: {submitted}
**Jobs Processed**: {total}

Good luck with your applications!
"""


# ============================================================================
# Chat Handler
# ============================================================================

class ChatHandler:
    """
    Handles chat-based interactions with users.
    
    Manages conversation flow and translates between
    natural language and system actions.
    """
    
    def __init__(self, orchestrator: WorkflowOrchestrator = None):
        """
        Initialize chat handler.
        
        Args:
            orchestrator: Workflow orchestrator instance
        """
        self.orchestrator = orchestrator or WorkflowOrchestrator()
        self.templates = ResponseTemplates()
        
        # Active conversations
        self._conversations: Dict[str, ConversationContext] = {}
    
    def get_conversation(self, user_id: str) -> ConversationContext:
        """Get or create conversation context for user."""
        if user_id not in self._conversations:
            self._conversations[user_id] = ConversationContext(user_id=user_id)
        return self._conversations[user_id]
    
    def process_message(self, user_id: str, message: str, 
                       attachments: List[Dict] = None) -> List[ChatMessage]:
        """
        Process a user message and return response(s).
        
        Args:
            user_id: User identifier
            message: User's message text
            attachments: Optional file attachments (CV, etc.)
            
        Returns:
            List of response messages
        """
        ctx = self.get_conversation(user_id)
        message_lower = message.lower().strip()
        
        # Log user message
        ctx.messages.append(ChatMessage(
            role="user",
            content=message
        ))
        
        responses = []
        
        # Handle based on conversation state
        if ctx.state == ConversationState.IDLE:
            responses = self._handle_idle(ctx, message_lower, attachments)
        
        elif ctx.state == ConversationState.COLLECTING_COMPANIES:
            responses = self._handle_companies(ctx, message)
        
        elif ctx.state == ConversationState.COLLECTING_TITLES:
            responses = self._handle_titles(ctx, message)
        
        elif ctx.state == ConversationState.COLLECTING_LOCATION:
            responses = self._handle_location(ctx, message_lower)
        
        elif ctx.state == ConversationState.COLLECTING_MODE:
            responses = self._handle_mode(ctx, message_lower)
        
        elif ctx.state == ConversationState.CONFIRMING_PREFERENCES:
            responses = self._handle_confirmation(ctx, message_lower)
        
        elif ctx.state == ConversationState.AWAITING_JOB_SELECTION:
            responses = self._handle_job_selection(ctx, message)
        
        elif ctx.state == ConversationState.AWAITING_CV_APPROVAL:
            responses = self._handle_cv_approval(ctx, message_lower)
        
        elif ctx.state == ConversationState.AWAITING_FORM_ANSWERS:
            responses = self._handle_form_answer(ctx, message)
        
        elif ctx.state == ConversationState.RUNNING:
            responses = self._handle_running(ctx, message_lower)
        
        # Log responses
        for resp in responses:
            ctx.messages.append(resp)
        
        return responses
    
    # =========================================================================
    # State Handlers
    # =========================================================================
    
    def _handle_idle(self, ctx: ConversationContext, message: str,
                    attachments: List[Dict] = None) -> List[ChatMessage]:
        """Handle messages when idle."""
        
        # Check for greetings or start commands
        if any(word in message for word in ['hi', 'hello', 'start', 'help']):
            ctx.state = ConversationState.COLLECTING_COMPANIES
            return [
                ChatMessage(role="agent", content=self.templates.WELCOME),
                ChatMessage(role="agent", content=self.templates.ASK_COMPANIES,
                          action_type="question", requires_response=True)
            ]
        
        # Try to parse a complete request
        parsed = self._try_parse_full_request(message)
        if parsed:
            ctx.partial_preferences = parsed
            ctx.state = ConversationState.CONFIRMING_PREFERENCES
            return [self._create_confirmation_message(ctx)]
        
        # Default welcome
        ctx.state = ConversationState.COLLECTING_COMPANIES
        return [
            ChatMessage(role="agent", content=self.templates.WELCOME),
            ChatMessage(role="agent", content=self.templates.ASK_COMPANIES,
                      action_type="question", requires_response=True)
        ]
    
    def _handle_companies(self, ctx: ConversationContext, message: str) -> List[ChatMessage]:
        """Handle company selection."""
        
        companies = self._parse_company_list(message)
        ctx.partial_preferences['companies'] = companies
        
        ctx.state = ConversationState.COLLECTING_TITLES
        return [
            ChatMessage(
                role="agent",
                content=f"Got it! Targeting {len(companies)} companies." if companies else "Searching all companies.",
                action_type="confirmation"
            ),
            ChatMessage(
                role="agent",
                content=self.templates.ASK_TITLES,
                action_type="question",
                requires_response=True
            )
        ]
    
    def _handle_titles(self, ctx: ConversationContext, message: str) -> List[ChatMessage]:
        """Handle job title selection."""
        
        titles = self._parse_title_list(message)
        if not titles:
            return [ChatMessage(
                role="agent",
                content="I need at least one job title. Try: 'Data Engineer' or 'ML Engineer, Data Scientist'",
                action_type="error",
                requires_response=True
            )]
        
        ctx.partial_preferences['titles'] = titles
        ctx.state = ConversationState.COLLECTING_LOCATION
        
        return [
            ChatMessage(
                role="agent",
                content=f"Looking for: {', '.join(titles)}",
                action_type="confirmation"
            ),
            ChatMessage(
                role="agent",
                content=self.templates.ASK_LOCATION,
                action_type="question",
                requires_response=True,
                options=["remote", "hybrid", "onsite", "any"]
            )
        ]
    
    def _handle_location(self, ctx: ConversationContext, message: str) -> List[ChatMessage]:
        """Handle location preference."""
        
        location_map = {
            'remote': LocationType.REMOTE,
            'hybrid': LocationType.HYBRID,
            'onsite': LocationType.ONSITE,
            'on-site': LocationType.ONSITE,
            'office': LocationType.ONSITE,
            'any': LocationType.ANY,
            'all': LocationType.ANY,
        }
        
        location = location_map.get(message, LocationType.ANY)
        ctx.partial_preferences['location'] = location.value
        
        ctx.state = ConversationState.COLLECTING_MODE
        return [
            ChatMessage(
                role="agent",
                content=f"Location preference: {location.value}",
                action_type="confirmation"
            ),
            ChatMessage(
                role="agent",
                content=self.templates.ASK_MODE,
                action_type="question",
                requires_response=True,
                options=["supervised", "autonomous"]
            )
        ]
    
    def _handle_mode(self, ctx: ConversationContext, message: str) -> List[ChatMessage]:
        """Handle application mode selection."""
        
        if 'auto' in message:
            mode = ApplicationMode.AUTONOMOUS
        else:
            mode = ApplicationMode.SUPERVISED
        
        ctx.partial_preferences['mode'] = mode.value
        ctx.state = ConversationState.CONFIRMING_PREFERENCES
        
        return [self._create_confirmation_message(ctx)]
    
    def _handle_confirmation(self, ctx: ConversationContext, message: str) -> List[ChatMessage]:
        """Handle preference confirmation."""
        
        if message in ['yes', 'y', 'ok', 'sure', 'start', 'go']:
            return self._start_workflow(ctx)
        
        elif message in ['no', 'n', 'cancel', 'restart']:
            ctx.state = ConversationState.COLLECTING_COMPANIES
            ctx.partial_preferences = {}
            return [
                ChatMessage(role="agent", content="No problem, let's start over."),
                ChatMessage(role="agent", content=self.templates.ASK_COMPANIES,
                          action_type="question", requires_response=True)
            ]
        
        else:
            return [ChatMessage(
                role="agent",
                content="Please say 'yes' to start searching or 'no' to change your preferences.",
                requires_response=True,
                options=["yes", "no"]
            )]
    
    def _handle_job_selection(self, ctx: ConversationContext, message: str) -> List[ChatMessage]:
        """Handle job selection from search results."""
        
        workflow = self.orchestrator.get_workflow(ctx.workflow_id)
        if not workflow:
            return [ChatMessage(role="agent", content="Workflow not found. Please start over.")]
        
        # Parse selection
        message_lower = message.lower()
        
        if message_lower in ['all', 'apply to all']:
            selected = list(range(len(workflow.discovered_jobs)))
        elif message_lower.startswith('top'):
            try:
                n = int(message_lower.replace('top', '').strip())
                selected = list(range(min(n, len(workflow.discovered_jobs))))
            except:
                selected = list(range(min(5, len(workflow.discovered_jobs))))
        else:
            # Parse comma-separated numbers
            try:
                selected = [int(x.strip()) - 1 for x in message.split(',')]
                selected = [i for i in selected if 0 <= i < len(workflow.discovered_jobs)]
            except:
                return [ChatMessage(
                    role="agent",
                    content="I didn't understand. Please enter job numbers like '1, 3, 5' or say 'all'",
                    requires_response=True
                )]
        
        if not selected:
            return [ChatMessage(
                role="agent",
                content="No valid jobs selected. Please try again.",
                requires_response=True
            )]
        
        # Apply selection
        self.orchestrator.step_select_jobs(workflow, selected)
        ctx.state = ConversationState.RUNNING
        
        return [
            ChatMessage(
                role="agent",
                content=f"Great! Selected {len(selected)} jobs. Starting CV generation..."
            ),
            *self._continue_workflow(ctx)
        ]
    
    def _handle_cv_approval(self, ctx: ConversationContext, message: str) -> List[ChatMessage]:
        """Handle CV approval."""
        
        workflow = self.orchestrator.get_workflow(ctx.workflow_id)
        if not workflow:
            return [ChatMessage(role="agent", content="Workflow not found.")]
        
        job_id = ctx.current_job_id
        
        if message in ['yes', 'y', 'approve', 'ok', 'good']:
            self.orchestrator.step_approve_cv(workflow, job_id, approved=True)
            ctx.state = ConversationState.RUNNING
            return [
                ChatMessage(role="agent", content="CV approved! Starting application..."),
                *self._continue_workflow(ctx)
            ]
        
        elif message in ['no', 'n', 'reject']:
            self.orchestrator.step_approve_cv(workflow, job_id, approved=False,
                                             feedback="User rejected")
            ctx.state = ConversationState.RUNNING
            return [
                ChatMessage(role="agent", content="CV rejected. Moving to next job..."),
                *self._continue_workflow(ctx)
            ]
        
        elif message.startswith('edit'):
            return [ChatMessage(
                role="agent",
                content="CV editing not yet implemented. Please approve or reject.",
                requires_response=True,
                options=["yes", "no"]
            )]
        
        return [ChatMessage(
            role="agent",
            content="Please say 'yes' to approve or 'no' to reject this CV.",
            requires_response=True,
            options=["yes", "no"]
        )]
    
    def _handle_form_answer(self, ctx: ConversationContext, message: str) -> List[ChatMessage]:
        """Handle form question answers."""
        
        if not ctx.pending_questions:
            ctx.state = ConversationState.RUNNING
            return self._continue_workflow(ctx)
        
        # Get current question
        question = ctx.pending_questions[0]
        
        # Store answer
        workflow = self.orchestrator.get_workflow(ctx.workflow_id)
        if workflow:
            self.orchestrator.step_handle_input(workflow, {question.question: message})
        
        # Remove answered question
        ctx.pending_questions.pop(0)
        
        # Check for more questions
        if ctx.pending_questions:
            next_q = ctx.pending_questions[0]
            return [self._create_question_message(next_q)]
        
        # All questions answered
        ctx.state = ConversationState.RUNNING
        return [
            ChatMessage(role="agent", content="Thanks! Continuing with the application..."),
            *self._continue_workflow(ctx)
        ]
    
    def _handle_running(self, ctx: ConversationContext, message: str) -> List[ChatMessage]:
        """Handle messages while workflow is running."""
        
        if message in ['status', 'progress']:
            return [self._create_status_message(ctx)]
        
        elif message in ['stop', 'cancel', 'pause']:
            return [ChatMessage(
                role="agent",
                content="Workflow paused. Say 'continue' to resume or 'cancel' to stop completely.",
                requires_response=True
            )]
        
        elif message in ['continue', 'resume']:
            return self._continue_workflow(ctx)
        
        return [ChatMessage(
            role="agent",
            content="Workflow is running. Say 'status' for progress or 'pause' to stop.",
            requires_response=True
        )]
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _try_parse_full_request(self, message: str) -> Optional[Dict]:
        """Try to parse a complete job search request from one message."""
        
        # Look for patterns like "apply for X at Y"
        message_lower = message.lower()
        
        # Common patterns
        if 'at' in message_lower and any(t in message_lower for t in 
                                         ['engineer', 'scientist', 'developer', 'analyst']):
            # Try to extract companies and titles
            parts = message_lower.split(' at ')
            if len(parts) == 2:
                titles = self._parse_title_list(parts[0])
                companies = self._parse_company_list(parts[1])
                
                if titles:
                    return {
                        'titles': titles,
                        'companies': companies,
                        'location': 'any',
                        'mode': 'supervised'
                    }
        
        return None
    
    def _parse_company_list(self, text: str) -> List[str]:
        """Parse company names from text."""
        
        text_lower = text.lower().strip()
        
        # Special keywords
        if text_lower in ['any', 'all', 'all companies']:
            return []
        
        if text_lower in ['tech giants', 'faang', 'big tech']:
            return ['Google', 'Meta', 'Amazon', 'Apple', 'Microsoft', 'Netflix']
        
        # Parse comma or "and" separated list
        text = text.replace(' and ', ',')
        companies = [c.strip().title() for c in text.split(',') if c.strip()]
        
        return companies
    
    def _parse_title_list(self, text: str) -> List[str]:
        """Parse job titles from text."""
        
        # Remove common prefixes
        text = text.lower()
        for prefix in ['looking for', 'want', 'apply for', 'interested in', 'i want']:
            text = text.replace(prefix, '')
        
        text = text.replace(' and ', ',').replace(' or ', ',')
        
        # Split and clean
        titles = []
        for part in text.split(','):
            part = part.strip()
            if part:
                # Title case job titles
                title = ' '.join(word.capitalize() for word in part.split())
                titles.append(title)
        
        return titles
    
    def _create_confirmation_message(self, ctx: ConversationContext) -> ChatMessage:
        """Create preference confirmation message."""
        
        prefs = ctx.partial_preferences
        
        companies = prefs.get('companies', [])
        companies_text = ', '.join(companies) if companies else "All 243 companies"
        
        content = self.templates.CONFIRM_PREFERENCES.format(
            companies=companies_text,
            titles=', '.join(prefs.get('titles', ['Not specified'])),
            location=prefs.get('location', 'any'),
            mode=prefs.get('mode', 'supervised')
        )
        
        return ChatMessage(
            role="agent",
            content=content,
            action_type="confirmation",
            requires_response=True,
            options=["yes", "no"]
        )
    
    def _create_question_message(self, question: HumanInputRequest) -> ChatMessage:
        """Create message for a form question."""
        
        options_text = ""
        if question.options:
            options_text = "\nOptions: " + ", ".join(question.options)
        
        content = self.templates.FORM_QUESTION.format(
            company=question.company,
            question=question.question,
            options_text=options_text
        )
        
        return ChatMessage(
            role="agent",
            content=content,
            action_type="question",
            requires_response=True,
            options=question.options
        )
    
    def _create_status_message(self, ctx: ConversationContext) -> ChatMessage:
        """Create workflow status message."""
        
        workflow = self.orchestrator.get_workflow(ctx.workflow_id)
        if not workflow:
            return ChatMessage(role="agent", content="No active workflow.")
        
        status = workflow.to_status()
        
        content = f"""**Workflow Status**

State: {status.state.value}
Jobs found: {status.jobs_found}
Jobs selected: {status.jobs_selected}
CVs generated: {status.cvs_generated}
CVs approved: {status.cvs_approved}
Applications submitted: {status.applications_submitted}
Pending input: {status.applications_pending_input}
"""
        
        return ChatMessage(role="agent", content=content)
    
    def _start_workflow(self, ctx: ConversationContext) -> List[ChatMessage]:
        """Start the job search workflow."""
        
        prefs = ctx.partial_preferences
        
        # Create preferences object
        preferences = JobSearchPreferences(
            companies=prefs.get('companies', []),
            job_titles=prefs.get('titles', []),
            location_type=LocationType(prefs.get('location', 'any')),
            application_mode=ApplicationMode(prefs.get('mode', 'supervised'))
        )
        
        # TODO: Get base CV and profile from user data
        # For now, use placeholders
        base_cv = "User CV placeholder"
        user_profile = {
            'full_name': 'User Name',
            'email': 'user@example.com',
            'work_authorized': True,
            'requires_sponsorship': False
        }
        
        # Create workflow
        workflow = self.orchestrator.create_workflow(
            user_id=ctx.user_id,
            preferences=preferences,
            base_cv=base_cv,
            user_profile=user_profile
        )
        
        ctx.workflow_id = workflow.workflow_id
        ctx.state = ConversationState.RUNNING
        
        return [
            ChatMessage(role="agent", content=self.templates.SEARCHING),
            *self._continue_workflow(ctx)
        ]
    
    def _continue_workflow(self, ctx: ConversationContext) -> List[ChatMessage]:
        """Continue running the workflow and return appropriate messages."""
        
        workflow = self.orchestrator.get_workflow(ctx.workflow_id)
        if not workflow:
            return [ChatMessage(role="agent", content="Workflow not found.")]
        
        # Run workflow until it needs user input
        workflow = self.orchestrator.run_workflow(workflow)
        
        # Generate response based on new state
        state = workflow.current_state
        
        if state == WorkflowState.JOBS_FOUND:
            ctx.state = ConversationState.AWAITING_JOB_SELECTION
            return [self._create_jobs_found_message(workflow)]
        
        elif state == WorkflowState.AWAITING_USER_APPROVAL:
            ctx.state = ConversationState.AWAITING_CV_APPROVAL
            return [self._create_cv_approval_message(workflow, ctx)]
        
        elif state == WorkflowState.NEEDS_INPUT:
            ctx.state = ConversationState.AWAITING_FORM_ANSWERS
            ctx.pending_questions = workflow.pending_inputs.copy()
            if ctx.pending_questions:
                return [self._create_question_message(ctx.pending_questions[0])]
        
        elif state == WorkflowState.COMPLETE:
            ctx.state = ConversationState.IDLE
            return [self._create_completion_message(workflow)]
        
        elif state == WorkflowState.FAILED:
            ctx.state = ConversationState.IDLE
            error = workflow.errors[-1] if workflow.errors else {}
            return [ChatMessage(
                role="agent",
                content=f"Workflow failed: {error.get('message', 'Unknown error')}"
            )]
        
        # Still running
        return [self._create_status_message(ctx)]
    
    def _create_jobs_found_message(self, workflow: WorkflowContext) -> ChatMessage:
        """Create message showing found jobs."""
        
        jobs = workflow.discovered_jobs
        
        if not jobs:
            mcp_urls = getattr(self.orchestrator.discovery_agent, '_pending_mcp_urls', {})
            return ChatMessage(
                role="agent",
                content=self.templates.NO_JOBS_FOUND.format(
                    workday_count=len(mcp_urls.get('workday', [])),
                    custom_count=len(mcp_urls.get('custom', [])),
                    aggregator_count=len(mcp_urls.get('aggregator', []))
                ),
                requires_response=True,
                options=["yes", "no"]
            )
        
        job_list = "\n".join([
            f"{i+1}. **{job.company}** - {job.title}\n   {job.location} | {job.location_type.value} | Score: {job.relevance_score or 'N/A'}"
            for i, job in enumerate(jobs[:10])
        ])
        
        return ChatMessage(
            role="agent",
            content=self.templates.JOBS_FOUND.format(
                count=len(jobs),
                job_list=job_list
            ),
            action_type="question",
            requires_response=True
        )
    
    def _create_cv_approval_message(self, workflow: WorkflowContext,
                                    ctx: ConversationContext) -> ChatMessage:
        """Create CV approval request message."""
        
        job = workflow.selected_jobs[workflow.current_job_index]
        job_id = job.job_id or str(workflow.current_job_index)
        ctx.current_job_id = job_id
        
        cv = workflow.generated_cvs.get(job_id)
        if not cv:
            return ChatMessage(role="agent", content="CV not found.")
        
        # Create preview
        preview = f"""
**Experience Highlights:**
"""
        for exp in cv.experiences[:2]:
            preview += f"\n*{exp.company}* - {exp.title}"
            for bullet in exp.bullets[:2]:
                preview += f"\n  - {bullet.text[:100]}..."
        
        return ChatMessage(
            role="agent",
            content=self.templates.CV_READY.format(
                company=job.company,
                title=job.title,
                score=cv.ats_score,
                covered=cv.requirements_covered,
                total=cv.total_requirements,
                cv_preview=preview
            ),
            action_type="confirmation",
            requires_response=True,
            options=["yes", "no", "edit"]
        )
    
    def _create_completion_message(self, workflow: WorkflowContext) -> ChatMessage:
        """Create workflow completion message."""
        
        return ChatMessage(
            role="agent",
            content=self.templates.WORKFLOW_COMPLETE.format(
                submitted=len(workflow.submitted_applications),
                total=len(workflow.selected_jobs)
            )
        )

