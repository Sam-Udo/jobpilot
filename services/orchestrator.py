"""
Workflow Orchestrator

Coordinates the entire job application workflow:
1. Discovery -> 2. Selection -> 3. CV Generation -> 4. Approval -> 5. Application

Implements:
- State machine for workflow progress
- Interrupt/resume for human-in-the-loop
- Parallel processing where possible
- Error handling and retry logic

In production, this would use Celery for task queuing.
"""

import logging
import json
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

from jobpilot.core.schemas import (
    WorkflowState, JobSearchPreferences, JobListing, JobListingBatch,
    GeneratedCV, ApplicationForm, HumanInputRequest, WorkflowStatus,
    ApplicationMode
)
from jobpilot.core.config import get_settings
from jobpilot.agents.discovery.discovery_agent import DiscoveryAgent
from jobpilot.agents.cv_architect.cv_architect import CVArchitectAgent
from jobpilot.agents.form_filler.form_filler import FormFillerAgent, FormState
from jobpilot.agents.vault.vault import KnowledgeBase, CredentialVault, EncryptionManager

logger = logging.getLogger(__name__)


# ============================================================================
# Workflow Events
# ============================================================================

class WorkflowEvent(str, Enum):
    """Events that trigger state transitions."""
    
    START = "start"
    SEARCH_COMPLETE = "search_complete"
    JOBS_SELECTED = "jobs_selected"
    CV_GENERATED = "cv_generated"
    CV_APPROVED = "cv_approved"
    CV_REJECTED = "cv_rejected"
    APPLICATION_STARTED = "application_started"
    INPUT_NEEDED = "input_needed"
    INPUT_RECEIVED = "input_received"
    APPLICATION_SUBMITTED = "application_submitted"
    APPLICATION_FAILED = "application_failed"
    CANCEL = "cancel"
    ERROR = "error"


# ============================================================================
# Workflow Context (State Container)
# ============================================================================

@dataclass
class WorkflowContext:
    """
    Contains all state for a running workflow.
    
    This is passed through all workflow steps and updated as we progress.
    """
    
    # Identity
    workflow_id: str
    user_id: str
    
    # Settings
    preferences: JobSearchPreferences
    mode: ApplicationMode = ApplicationMode.SUPERVISED
    
    # State
    current_state: WorkflowState = WorkflowState.INIT
    
    # Progress tracking
    discovered_jobs: List[JobListing] = field(default_factory=list)
    selected_jobs: List[JobListing] = field(default_factory=list)
    current_job_index: int = 0
    
    # CV data
    base_cv: Optional[str] = None
    generated_cvs: Dict[str, GeneratedCV] = field(default_factory=dict)  # job_id -> cv
    cv_validations: Dict[str, Dict] = field(default_factory=dict)  # job_id -> validation
    
    # Application data
    form_states: Dict[str, FormState] = field(default_factory=dict)  # job_id -> state
    pending_inputs: List[HumanInputRequest] = field(default_factory=list)
    submitted_applications: List[str] = field(default_factory=list)  # job_ids
    
    # User profile
    user_profile: Dict = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Error tracking
    errors: List[Dict] = field(default_factory=list)
    
    def update(self):
        """Update timestamp."""
        self.updated_at = datetime.utcnow()
    
    def to_status(self) -> WorkflowStatus:
        """Convert to status summary."""
        return WorkflowStatus(
            workflow_id=self.workflow_id,
            user_id=self.user_id,
            state=self.current_state,
            jobs_found=len(self.discovered_jobs),
            jobs_selected=len(self.selected_jobs),
            cvs_generated=len(self.generated_cvs),
            cvs_approved=sum(1 for v in self.cv_validations.values() 
                           if v.get('approved', False)),
            applications_submitted=len(self.submitted_applications),
            applications_pending_input=len(self.pending_inputs),
            current_job=self.selected_jobs[self.current_job_index].title 
                       if self.current_job_index < len(self.selected_jobs) else None,
            current_action=self.current_state.value,
            created_at=self.created_at,
            updated_at=self.updated_at,
            last_error=self.errors[-1].get('message') if self.errors else None
        )


# ============================================================================
# State Machine
# ============================================================================

class WorkflowStateMachine:
    """
    Manages state transitions for the workflow.
    
    Valid transitions:
    INIT -> SEARCHING
    SEARCHING -> JOBS_FOUND
    JOBS_FOUND -> JOBS_SELECTED
    JOBS_SELECTED -> CV_GENERATING
    CV_GENERATING -> CV_READY
    CV_READY -> AWAITING_USER_APPROVAL (supervised) | CV_APPROVED (autonomous)
    AWAITING_USER_APPROVAL -> CV_APPROVED | CV_GENERATING (if rejected)
    CV_APPROVED -> APPLYING
    APPLYING -> NEEDS_INPUT | SUBMITTED
    NEEDS_INPUT -> INPUT_RECEIVED
    INPUT_RECEIVED -> APPLYING
    SUBMITTED -> COMPLETE (if all done) | CV_GENERATING (next job)
    """
    
    TRANSITIONS = {
        WorkflowState.INIT: [WorkflowState.SEARCHING],
        WorkflowState.SEARCHING: [WorkflowState.JOBS_FOUND, WorkflowState.FAILED],
        WorkflowState.JOBS_FOUND: [WorkflowState.JOBS_SELECTED],
        WorkflowState.JOBS_SELECTED: [WorkflowState.CV_GENERATING],
        WorkflowState.CV_GENERATING: [WorkflowState.CV_READY, WorkflowState.FAILED],
        WorkflowState.CV_READY: [WorkflowState.AWAITING_USER_APPROVAL, WorkflowState.CV_APPROVED],
        WorkflowState.AWAITING_USER_APPROVAL: [WorkflowState.CV_APPROVED, WorkflowState.CV_GENERATING],
        WorkflowState.CV_APPROVED: [WorkflowState.APPLYING],
        WorkflowState.APPLYING: [WorkflowState.NEEDS_INPUT, WorkflowState.SUBMITTED, WorkflowState.FAILED],
        WorkflowState.NEEDS_INPUT: [WorkflowState.INPUT_RECEIVED],
        WorkflowState.INPUT_RECEIVED: [WorkflowState.APPLYING],
        WorkflowState.SUBMITTED: [WorkflowState.CV_GENERATING, WorkflowState.COMPLETE],
    }
    
    @classmethod
    def can_transition(cls, from_state: WorkflowState, to_state: WorkflowState) -> bool:
        """Check if transition is valid."""
        valid_targets = cls.TRANSITIONS.get(from_state, [])
        return to_state in valid_targets
    
    @classmethod
    def transition(cls, ctx: WorkflowContext, to_state: WorkflowState) -> bool:
        """
        Attempt state transition.
        
        Returns True if successful, False if invalid.
        """
        if cls.can_transition(ctx.current_state, to_state):
            logger.info(f"Workflow {ctx.workflow_id}: {ctx.current_state.value} -> {to_state.value}")
            ctx.current_state = to_state
            ctx.update()
            return True
        
        logger.warning(f"Invalid transition: {ctx.current_state.value} -> {to_state.value}")
        return False


# ============================================================================
# Main Orchestrator
# ============================================================================

class WorkflowOrchestrator:
    """
    Main orchestrator that coordinates all agents.
    
    Handles:
    - Workflow creation and management
    - Agent coordination
    - State persistence
    - Interrupt/resume
    - User notifications
    """
    
    def __init__(self, llm_client=None, notification_callback: Callable = None):
        """
        Initialize the orchestrator.
        
        Args:
            llm_client: Claude (Anthropic) client for LLM calls
            notification_callback: Function to call for user notifications
        """
        self.settings = get_settings()
        self.llm_client = llm_client
        self.notification_callback = notification_callback or self._default_notification
        
        # Initialize components
        self.encryption = EncryptionManager()
        self.credential_vault = CredentialVault(self.encryption)
        self.knowledge_base = KnowledgeBase()
        
        # Initialize agents
        self.discovery_agent = DiscoveryAgent()
        self.cv_architect = CVArchitectAgent(llm_client)
        
        # Active workflows
        self._workflows: Dict[str, WorkflowContext] = {}
    
    def _default_notification(self, user_id: str, message: str, data: Dict = None):
        """Default notification handler (just logs)."""
        logger.info(f"NOTIFICATION [{user_id}]: {message}")
    
    # =========================================================================
    # Workflow Lifecycle
    # =========================================================================
    
    def create_workflow(self, user_id: str, preferences: JobSearchPreferences,
                       base_cv: str, user_profile: Dict) -> WorkflowContext:
        """
        Create a new job application workflow.
        
        Args:
            user_id: User identifier
            preferences: Job search preferences
            base_cv: User's base CV text
            user_profile: User profile data
            
        Returns:
            New workflow context
        """
        import uuid
        workflow_id = str(uuid.uuid4())
        
        ctx = WorkflowContext(
            workflow_id=workflow_id,
            user_id=user_id,
            preferences=preferences,
            mode=preferences.application_mode,
            base_cv=base_cv,
            user_profile=user_profile
        )
        
        self._workflows[workflow_id] = ctx
        logger.info(f"Created workflow {workflow_id} for user {user_id}")
        
        # Populate knowledge base from profile
        self.knowledge_base.populate_from_profile(user_id, user_profile)
        
        return ctx
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowStatus]:
        """Get workflow status summary."""
        ctx = self.get_workflow(workflow_id)
        if ctx:
            return ctx.to_status()
        return None
    
    # =========================================================================
    # Workflow Steps
    # =========================================================================
    
    def step_search(self, ctx: WorkflowContext) -> WorkflowContext:
        """
        Step 1: Search for jobs.
        
        Uses Discovery Agent to find matching jobs.
        """
        WorkflowStateMachine.transition(ctx, WorkflowState.SEARCHING)
        
        try:
            # Run discovery
            batch = self.discovery_agent.search(ctx.preferences)
            ctx.discovered_jobs = batch.jobs
            
            WorkflowStateMachine.transition(ctx, WorkflowState.JOBS_FOUND)
            
            # Notify user
            self.notification_callback(
                ctx.user_id,
                f"Found {len(batch.jobs)} matching jobs",
                {'job_count': len(batch.jobs), 'jobs': [j.dict() for j in batch.jobs[:10]]}
            )
            
        except Exception as e:
            ctx.errors.append({'step': 'search', 'message': str(e)})
            WorkflowStateMachine.transition(ctx, WorkflowState.FAILED)
            logger.error(f"Search failed: {e}")
        
        return ctx
    
    def step_select_jobs(self, ctx: WorkflowContext, selected_indices: List[int] = None) -> WorkflowContext:
        """
        Step 2: User selects which jobs to apply to.
        
        In autonomous mode, all jobs above threshold are selected.
        In supervised mode, user picks from list.
        """
        if ctx.mode == ApplicationMode.AUTONOMOUS:
            # Auto-select top jobs
            ctx.selected_jobs = ctx.discovered_jobs[:10]  # Top 10
        elif selected_indices:
            ctx.selected_jobs = [
                ctx.discovered_jobs[i] for i in selected_indices
                if i < len(ctx.discovered_jobs)
            ]
        else:
            # No selection yet - wait for user
            return ctx
        
        WorkflowStateMachine.transition(ctx, WorkflowState.JOBS_SELECTED)
        logger.info(f"Selected {len(ctx.selected_jobs)} jobs for application")
        
        return ctx
    
    def step_generate_cv(self, ctx: WorkflowContext) -> WorkflowContext:
        """
        Step 3: Generate tailored CV for current job.
        """
        if ctx.current_job_index >= len(ctx.selected_jobs):
            logger.info("All CVs generated")
            return ctx
        
        job = ctx.selected_jobs[ctx.current_job_index]
        WorkflowStateMachine.transition(ctx, WorkflowState.CV_GENERATING)
        
        try:
            # Generate CV
            generated_cv, validation = self.cv_architect.create_tailored_cv(
                ctx.base_cv, job
            )
            
            # Store results
            job_id = job.job_id or str(ctx.current_job_index)
            ctx.generated_cvs[job_id] = generated_cv
            ctx.cv_validations[job_id] = validation
            
            WorkflowStateMachine.transition(ctx, WorkflowState.CV_READY)
            
            # Decide on approval
            if ctx.mode == ApplicationMode.AUTONOMOUS:
                if self.cv_architect.should_auto_approve(generated_cv, validation):
                    ctx.cv_validations[job_id]['approved'] = True
                    WorkflowStateMachine.transition(ctx, WorkflowState.CV_APPROVED)
                else:
                    # Needs manual review even in autonomous mode
                    WorkflowStateMachine.transition(ctx, WorkflowState.AWAITING_USER_APPROVAL)
                    self.notification_callback(
                        ctx.user_id,
                        f"CV for {job.company} needs review (ATS: {generated_cv.ats_score}%)",
                        {'job_id': job_id, 'ats_score': generated_cv.ats_score}
                    )
            else:
                # Supervised mode - always ask for approval
                WorkflowStateMachine.transition(ctx, WorkflowState.AWAITING_USER_APPROVAL)
                self.notification_callback(
                    ctx.user_id,
                    f"CV ready for {job.company} - please review",
                    {'job_id': job_id, 'cv': generated_cv.dict(), 'validation': validation}
                )
            
        except Exception as e:
            ctx.errors.append({'step': 'cv_generation', 'job': job.title, 'message': str(e)})
            logger.error(f"CV generation failed: {e}")
            # Skip to next job
            ctx.current_job_index += 1
        
        return ctx
    
    def step_approve_cv(self, ctx: WorkflowContext, job_id: str, approved: bool,
                       feedback: str = None) -> WorkflowContext:
        """
        Step 4: User approves or rejects CV.
        """
        if approved:
            ctx.cv_validations[job_id]['approved'] = True
            WorkflowStateMachine.transition(ctx, WorkflowState.CV_APPROVED)
            logger.info(f"CV approved for job {job_id}")
        else:
            ctx.cv_validations[job_id]['approved'] = False
            ctx.cv_validations[job_id]['feedback'] = feedback
            
            # Could regenerate with feedback, for now just skip
            ctx.current_job_index += 1
            WorkflowStateMachine.transition(ctx, WorkflowState.CV_GENERATING)
            logger.info(f"CV rejected for job {job_id}, moving to next")
        
        return ctx
    
    def step_apply(self, ctx: WorkflowContext) -> WorkflowContext:
        """
        Step 5: Apply to job using Form Filler Agent.
        """
        if ctx.current_state != WorkflowState.CV_APPROVED:
            return ctx
        
        job = ctx.selected_jobs[ctx.current_job_index]
        job_id = job.job_id or str(ctx.current_job_index)
        
        WorkflowStateMachine.transition(ctx, WorkflowState.APPLYING)
        
        # Initialize form filler for this user
        form_filler = FormFillerAgent(
            knowledge_base=self.knowledge_base,
            credential_vault=self.credential_vault,
            user_id=ctx.user_id,
            user_profile=ctx.user_profile
        )
        
        try:
            # In production: Navigate to job URL and get page source
            # For now, this is a placeholder
            logger.info(f"Would apply to {job.job_url}")
            
            # Simulate form analysis (would scrape actual page)
            # form, human_requests = form_filler.analyze_application(page_source, job.job_url, job)
            
            # For demonstration, simulate successful application
            ctx.submitted_applications.append(job_id)
            WorkflowStateMachine.transition(ctx, WorkflowState.SUBMITTED)
            
            self.notification_callback(
                ctx.user_id,
                f"Application submitted for {job.title} at {job.company}",
                {'job_id': job_id, 'company': job.company}
            )
            
            # Move to next job
            ctx.current_job_index += 1
            if ctx.current_job_index < len(ctx.selected_jobs):
                WorkflowStateMachine.transition(ctx, WorkflowState.CV_GENERATING)
            else:
                WorkflowStateMachine.transition(ctx, WorkflowState.COMPLETE)
            
        except Exception as e:
            ctx.errors.append({'step': 'apply', 'job': job.title, 'message': str(e)})
            WorkflowStateMachine.transition(ctx, WorkflowState.FAILED)
            logger.error(f"Application failed: {e}")
        
        return ctx
    
    def step_handle_input(self, ctx: WorkflowContext, 
                         answers: Dict[str, str]) -> WorkflowContext:
        """
        Handle human-provided answers for pending questions.
        """
        if ctx.current_state != WorkflowState.NEEDS_INPUT:
            return ctx
        
        # Update knowledge base with new answers
        for question, answer in answers.items():
            self.knowledge_base.add_entry(
                ctx.user_id,
                question,
                answer,
                source='user_input'
            )
        
        # Clear pending inputs
        ctx.pending_inputs = []
        
        WorkflowStateMachine.transition(ctx, WorkflowState.INPUT_RECEIVED)
        WorkflowStateMachine.transition(ctx, WorkflowState.APPLYING)
        
        # Resume application
        return self.step_apply(ctx)
    
    # =========================================================================
    # Main Run Loop
    # =========================================================================
    
    def run_workflow(self, ctx: WorkflowContext, 
                    until_state: WorkflowState = None) -> WorkflowContext:
        """
        Run workflow until completion or specified state.
        
        In production, each step would be a separate Celery task.
        
        Args:
            ctx: Workflow context
            until_state: Stop when reaching this state (for testing)
            
        Returns:
            Updated context
        """
        max_iterations = 100  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            current = ctx.current_state
            
            # Check stop conditions
            if current in [WorkflowState.COMPLETE, WorkflowState.FAILED, WorkflowState.CANCELLED]:
                break
            
            if until_state and current == until_state:
                break
            
            # Check for states that need human input
            if current in [WorkflowState.AWAITING_USER_APPROVAL, WorkflowState.NEEDS_INPUT]:
                logger.info(f"Workflow paused at {current.value} - waiting for user")
                break
            
            # Execute next step based on state
            if current == WorkflowState.INIT:
                ctx = self.step_search(ctx)
            
            elif current == WorkflowState.JOBS_FOUND:
                # In supervised mode, wait for user selection
                if ctx.mode == ApplicationMode.SUPERVISED:
                    break
                ctx = self.step_select_jobs(ctx)
            
            elif current == WorkflowState.JOBS_SELECTED:
                ctx = self.step_generate_cv(ctx)
            
            elif current == WorkflowState.CV_GENERATING:
                ctx = self.step_generate_cv(ctx)
            
            elif current == WorkflowState.CV_APPROVED:
                ctx = self.step_apply(ctx)
            
            elif current == WorkflowState.INPUT_RECEIVED:
                ctx = self.step_apply(ctx)
            
            elif current == WorkflowState.SUBMITTED:
                # Check if more jobs
                if ctx.current_job_index < len(ctx.selected_jobs):
                    ctx = self.step_generate_cv(ctx)
                else:
                    WorkflowStateMachine.transition(ctx, WorkflowState.COMPLETE)
            
            else:
                logger.warning(f"Unhandled state: {current.value}")
                break
        
        logger.info(f"Workflow run complete. Final state: {ctx.current_state.value}")
        return ctx


# ============================================================================
# Convenience Functions
# ============================================================================

def create_orchestrator(llm_client=None) -> WorkflowOrchestrator:
    """Create an orchestrator instance."""
    return WorkflowOrchestrator(llm_client=llm_client)


def start_job_search(user_id: str, preferences: dict, base_cv: str,
                    user_profile: dict, llm_client=None) -> WorkflowStatus:
    """
    Convenience function to start a job search workflow.
    
    Args:
        user_id: User identifier
        preferences: Job search preferences dict
        base_cv: User's CV text
        user_profile: User profile dict
        llm_client: Optional LLM client
        
    Returns:
        Initial workflow status
    """
    orchestrator = create_orchestrator(llm_client)
    
    # Parse preferences
    prefs = JobSearchPreferences(**preferences)
    
    # Create workflow
    ctx = orchestrator.create_workflow(user_id, prefs, base_cv, user_profile)
    
    # Run until first user interaction needed
    ctx = orchestrator.run_workflow(ctx)
    
    return ctx.to_status()

