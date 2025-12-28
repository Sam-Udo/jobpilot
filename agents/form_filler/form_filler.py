"""
Form Filler Agent - Layer 3

Automates job application form filling using:
1. DOM Analysis - Identifies form fields
2. Knowledge Base - Retrieves answers via RAG
3. Interrupt Pattern - Pauses for unknown questions
4. Vision (optional) - Uses Claude for complex forms

Supports the interrupt workflow:
- Encounters unknown field -> Serialize state -> Notify user -> Wait for input
"""

import logging
import json
import time
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from jobpilot.core.schemas import (
    FormField, ApplicationForm, HumanInputRequest,
    JobListing
)
from jobpilot.core.config import get_settings
from jobpilot.agents.vault.vault import KnowledgeBase, CredentialVault

logger = logging.getLogger(__name__)


# ============================================================================
# Form Field Types and Detection
# ============================================================================

class FieldType(str, Enum):
    """Types of form fields we can handle."""
    TEXT = "text"
    EMAIL = "email"
    PHONE = "phone"
    SELECT = "select"
    RADIO = "radio"
    CHECKBOX = "checkbox"
    TEXTAREA = "textarea"
    FILE = "file"
    DATE = "date"
    NUMBER = "number"
    UNKNOWN = "unknown"


@dataclass
class DetectedField:
    """A form field detected on the page."""
    
    selector: str
    field_type: FieldType
    label: str
    required: bool
    options: Optional[List[str]] = None
    placeholder: Optional[str] = None
    current_value: Optional[str] = None
    
    # Agent analysis
    question_category: Optional[str] = None
    suggested_answer: Optional[str] = None
    confidence: float = 0.0
    needs_human_input: bool = False


@dataclass
class FormState:
    """
    Serializable form state for interrupt/resume.
    
    When the agent needs to pause (waiting for human input),
    it serializes this state so it can resume later.
    """
    
    job_id: str
    page_url: str
    fields: List[Dict]
    filled_fields: List[str]
    pending_fields: List[str]
    screenshot_path: Optional[str] = None
    
    # State
    status: str = "in_progress"  # in_progress, needs_input, ready_to_submit
    created_at: str = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
    
    def to_json(self) -> str:
        """Serialize to JSON for storage."""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, data: str) -> 'FormState':
        """Deserialize from JSON."""
        return cls(**json.loads(data))


# ============================================================================
# DOM Analyzer - Detects and classifies form fields
# ============================================================================

class DOMAnalyzer:
    """
    Analyzes page DOM to detect form fields.
    
    Uses CSS selectors and heuristics to identify:
    - Input fields and their types
    - Labels and placeholders
    - Required fields
    - Select/radio options
    """
    
    # Common label patterns for field classification
    FIELD_PATTERNS = {
        'first_name': ['first name', 'given name', 'firstname'],
        'last_name': ['last name', 'surname', 'family name', 'lastname'],
        'email': ['email', 'e-mail'],
        'phone': ['phone', 'telephone', 'mobile', 'cell'],
        'linkedin': ['linkedin', 'linkedin profile', 'linkedin url'],
        'resume': ['resume', 'cv', 'curriculum vitae'],
        'cover_letter': ['cover letter', 'motivation letter'],
        'salary': ['salary', 'compensation', 'expected salary'],
        'start_date': ['start date', 'available from', 'availability'],
        'work_authorization': ['authorized to work', 'work authorization', 'eligible to work'],
        'sponsorship': ['sponsorship', 'visa', 'require sponsorship'],
        'experience_years': ['years of experience', 'how many years'],
    }
    
    def analyze_form(self, page_source: str, page_url: str) -> List[DetectedField]:
        """
        Analyze HTML to detect form fields.
        
        Args:
            page_source: HTML page source
            page_url: Current page URL
            
        Returns:
            List of detected form fields
        """
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(page_source, 'html.parser')
        fields = []
        
        # Find all input elements
        for input_elem in soup.find_all(['input', 'select', 'textarea']):
            field = self._analyze_element(input_elem)
            if field:
                fields.append(field)
        
        logger.info(f"Detected {len(fields)} form fields")
        return fields
    
    def _analyze_element(self, elem) -> Optional[DetectedField]:
        """Analyze a single form element."""
        
        # Get field type
        tag_name = elem.name
        input_type = elem.get('type', 'text').lower()
        
        if tag_name == 'select':
            field_type = FieldType.SELECT
        elif tag_name == 'textarea':
            field_type = FieldType.TEXTAREA
        elif input_type in ['radio']:
            field_type = FieldType.RADIO
        elif input_type in ['checkbox']:
            field_type = FieldType.CHECKBOX
        elif input_type in ['file']:
            field_type = FieldType.FILE
        elif input_type in ['email']:
            field_type = FieldType.EMAIL
        elif input_type in ['tel']:
            field_type = FieldType.PHONE
        elif input_type in ['date']:
            field_type = FieldType.DATE
        elif input_type in ['number']:
            field_type = FieldType.NUMBER
        else:
            field_type = FieldType.TEXT
        
        # Get selector (prefer ID, then name, then generate)
        elem_id = elem.get('id', '')
        elem_name = elem.get('name', '')
        selector = f"#{elem_id}" if elem_id else f"[name='{elem_name}']" if elem_name else None
        
        if not selector:
            return None
        
        # Find label
        label = self._find_label(elem, elem_id)
        
        # Check if required
        required = elem.get('required') is not None or elem.get('aria-required') == 'true'
        
        # Get options for select/radio
        options = None
        if field_type == FieldType.SELECT:
            options = [opt.text.strip() for opt in elem.find_all('option') if opt.text.strip()]
        
        # Get placeholder
        placeholder = elem.get('placeholder', '')
        
        # Get current value
        current_value = elem.get('value', '')
        
        # Classify the question
        question_category = self._classify_question(label, placeholder)
        
        return DetectedField(
            selector=selector,
            field_type=field_type,
            label=label,
            required=required,
            options=options,
            placeholder=placeholder,
            current_value=current_value,
            question_category=question_category
        )
    
    def _find_label(self, elem, elem_id: str) -> str:
        """Find the label for a form element."""
        # Check for associated label
        if elem_id:
            label = elem.find_previous('label', {'for': elem_id})
            if label:
                return label.text.strip()
        
        # Check parent label
        parent = elem.find_parent('label')
        if parent:
            return parent.text.strip()
        
        # Check aria-label
        aria_label = elem.get('aria-label', '')
        if aria_label:
            return aria_label
        
        # Check nearby text
        prev_sibling = elem.find_previous_sibling(string=True)
        if prev_sibling:
            return prev_sibling.strip()[:100]
        
        return elem.get('placeholder', '') or elem.get('name', 'Unknown field')
    
    def _classify_question(self, label: str, placeholder: str) -> Optional[str]:
        """Classify what kind of question this field is asking."""
        combined = f"{label} {placeholder}".lower()
        
        for category, patterns in self.FIELD_PATTERNS.items():
            if any(pattern in combined for pattern in patterns):
                return category
        
        return None


# ============================================================================
# Form Filler Agent
# ============================================================================

class FormFillerAgent:
    """
    Main agent for filling job application forms.
    
    Workflow:
    1. Navigate to application page
    2. Analyze form fields
    3. For each field:
       a. Look up answer in knowledge base
       b. If found with high confidence -> fill
       c. If not found or low confidence -> mark for human input
    4. If any fields need human input -> INTERRUPT
    5. Otherwise -> submit
    """
    
    def __init__(self, knowledge_base: KnowledgeBase,
                 credential_vault: CredentialVault,
                 user_id: str,
                 user_profile: Dict):
        """
        Initialize the form filler.
        
        Args:
            knowledge_base: For looking up answers
            credential_vault: For login credentials
            user_id: Current user ID
            user_profile: User's profile data
        """
        self.knowledge_base = knowledge_base
        self.credential_vault = credential_vault
        self.user_id = user_id
        self.user_profile = user_profile
        self.settings = get_settings()
        
        self.dom_analyzer = DOMAnalyzer()
        
        # Current state
        self._current_state: Optional[FormState] = None
        self._browser = None
    
    def analyze_application(self, page_source: str, page_url: str,
                           job: JobListing) -> Tuple[ApplicationForm, List[HumanInputRequest]]:
        """
        Analyze an application page and determine what's needed.
        
        Args:
            page_source: HTML of the application page
            page_url: Current URL
            job: Job being applied to
            
        Returns:
            Tuple of (ApplicationForm, list of questions needing human input)
        """
        # Detect fields
        detected_fields = self.dom_analyzer.analyze_form(page_source, page_url)
        
        form_fields = []
        human_requests = []
        
        for field in detected_fields:
            # Try to find answer
            answer_result = self._find_answer_for_field(field)
            
            form_field = FormField(
                field_id=field.selector,
                field_type=field.field_type.value,
                label=field.label,
                required=field.required,
                options=field.options,
                current_value=field.current_value,
                confidence=answer_result.get('confidence', 0),
                suggested_value=answer_result.get('answer'),
                needs_human_input=answer_result.get('needs_human', False)
            )
            
            form_fields.append(form_field)
            
            # Create human input request if needed
            if form_field.needs_human_input and field.required:
                request = HumanInputRequest(
                    workflow_id="",  # Set by orchestrator
                    job_id=job.job_id or "",
                    company=job.company,
                    question=field.label,
                    field_type=field.field_type.value,
                    options=field.options,
                    context=f"Applying to {job.title} at {job.company}"
                )
                human_requests.append(request)
        
        # Create application form summary
        application_form = ApplicationForm(
            job_id=job.job_id or "",
            page_url=page_url,
            form_fields=form_fields,
            total_fields=len(form_fields),
            fields_filled=sum(1 for f in form_fields if f.suggested_value),
            fields_needing_input=len(human_requests)
        )
        
        logger.info(f"Form analysis: {application_form.fields_filled}/{application_form.total_fields} "
                   f"fields auto-fillable, {application_form.fields_needing_input} need human input")
        
        return application_form, human_requests
    
    def _find_answer_for_field(self, field: DetectedField) -> Dict:
        """
        Find answer for a form field.
        
        Checks:
        1. User profile (for basic info)
        2. Knowledge base (for learned answers)
        3. Default values (for common optional fields)
        
        Returns:
            Dict with 'answer', 'confidence', 'needs_human'
        """
        confidence_threshold = self.settings.workflow.confidence_threshold
        
        # Profile-based answers (highest confidence)
        profile_answer = self._get_profile_answer(field)
        if profile_answer:
            return {
                'answer': profile_answer,
                'confidence': 1.0,
                'needs_human': False
            }
        
        # Knowledge base lookup
        if field.label:
            kb_result = self.knowledge_base.find_answer(self.user_id, field.label)
            if kb_result and kb_result['confidence'] >= confidence_threshold:
                return {
                    'answer': kb_result['answer'],
                    'confidence': kb_result['confidence'],
                    'needs_human': False
                }
        
        # Default values for optional fields
        if not field.required:
            default = self._get_default_value(field)
            if default:
                return {
                    'answer': default,
                    'confidence': 0.7,
                    'needs_human': False
                }
        
        # Can't determine - needs human input
        return {
            'answer': None,
            'confidence': 0.0,
            'needs_human': True
        }
    
    def _get_profile_answer(self, field: DetectedField) -> Optional[str]:
        """Get answer from user profile."""
        category = field.question_category
        
        if not category:
            return None
        
        profile_mapping = {
            'first_name': self.user_profile.get('full_name', '').split()[0] if self.user_profile.get('full_name') else None,
            'last_name': self.user_profile.get('full_name', '').split()[-1] if self.user_profile.get('full_name') else None,
            'email': self.user_profile.get('email'),
            'phone': self.user_profile.get('phone'),
            'linkedin': self.user_profile.get('linkedin_url'),
        }
        
        return profile_mapping.get(category)
    
    def _get_default_value(self, field: DetectedField) -> Optional[str]:
        """Get default value for optional fields."""
        # Skip optional EEO questions
        eeo_categories = ['veteran_status', 'disability', 'gender', 'ethnicity']
        if field.question_category in eeo_categories:
            # Look for "Decline to answer" or similar option
            if field.options:
                for opt in field.options:
                    if any(phrase in opt.lower() for phrase in 
                          ['decline', 'prefer not', 'choose not', 'not to say']):
                        return opt
        
        return None
    
    # =========================================================================
    # Interrupt Pattern Implementation
    # =========================================================================
    
    def create_interrupt_state(self, job: JobListing, page_url: str,
                               form: ApplicationForm) -> FormState:
        """
        Create serializable state for interrupt/resume.
        
        Called when the agent needs to pause for human input.
        
        Args:
            job: Current job
            page_url: Current page URL
            form: Analyzed form
            
        Returns:
            FormState that can be serialized and stored
        """
        fields_data = [
            {
                'field_id': f.field_id,
                'label': f.label,
                'type': f.field_type,
                'required': f.required,
                'suggested_value': f.suggested_value,
                'needs_human_input': f.needs_human_input
            }
            for f in form.form_fields
        ]
        
        filled = [f.field_id for f in form.form_fields if f.suggested_value]
        pending = [f.field_id for f in form.form_fields if f.needs_human_input]
        
        state = FormState(
            job_id=job.job_id or "",
            page_url=page_url,
            fields=fields_data,
            filled_fields=filled,
            pending_fields=pending,
            status="needs_input"
        )
        
        self._current_state = state
        logger.info(f"Created interrupt state: {len(pending)} fields pending")
        
        return state
    
    def resume_from_state(self, state: FormState, human_answers: Dict[str, str]) -> FormState:
        """
        Resume application with human-provided answers.
        
        Args:
            state: Stored form state
            human_answers: Dict mapping field_id to answer
            
        Returns:
            Updated form state
        """
        logger.info(f"Resuming from state with {len(human_answers)} answers")
        
        # Update fields with human answers
        for field in state.fields:
            field_id = field['field_id']
            if field_id in human_answers:
                field['suggested_value'] = human_answers[field_id]
                field['needs_human_input'] = False
                
                # Also learn this answer for future
                self.knowledge_base.add_entry(
                    self.user_id,
                    field['label'],
                    human_answers[field_id],
                    source='form_learned'
                )
        
        # Update state
        state.pending_fields = [
            f['field_id'] for f in state.fields if f.get('needs_human_input', False)
        ]
        state.filled_fields = [
            f['field_id'] for f in state.fields if f.get('suggested_value')
        ]
        
        if not state.pending_fields:
            state.status = "ready_to_submit"
        
        self._current_state = state
        return state
    
    # =========================================================================
    # Form Filling Execution (would use Playwright in production)
    # =========================================================================
    
    def fill_form(self, form: ApplicationForm, cv_path: str) -> Dict:
        """
        Fill the form with determined values.
        
        Note: In production, this would use Playwright to interact with browser.
        This is a placeholder showing the structure.
        
        Args:
            form: Analyzed form with suggested values
            cv_path: Path to CV file to upload
            
        Returns:
            Dict with fill results
        """
        results = {
            'fields_attempted': 0,
            'fields_filled': 0,
            'fields_failed': 0,
            'errors': []
        }
        
        for field in form.form_fields:
            if not field.suggested_value:
                continue
            
            results['fields_attempted'] += 1
            
            try:
                # In production: self._fill_field_with_playwright(field)
                logger.info(f"Would fill {field.field_id} with: {field.suggested_value[:50]}...")
                results['fields_filled'] += 1
                
            except Exception as e:
                results['fields_failed'] += 1
                results['errors'].append(f"{field.field_id}: {str(e)}")
        
        return results
    
    def submit_application(self) -> Dict:
        """
        Submit the completed application.
        
        Returns:
            Dict with submission results
        """
        # In production: Find and click submit button
        logger.info("Would submit application here")
        
        return {
            'submitted': True,
            'timestamp': datetime.utcnow().isoformat()
        }


# ============================================================================
# ATS-Specific Adapters (Optional - for common systems)
# ============================================================================

class WorkdayAdapter:
    """
    Adapter for Workday application forms.
    
    Workday has a specific flow:
    1. Create account or login
    2. Upload resume
    3. Auto-fill from resume
    4. Review and edit
    5. Answer additional questions
    6. Submit
    """
    
    def detect_workday(self, page_url: str) -> bool:
        """Check if URL is a Workday page."""
        return 'myworkdayjobs.com' in page_url
    
    def get_login_selectors(self) -> Dict:
        """Get common Workday login selectors."""
        return {
            'email': 'input[data-automation-id="email"]',
            'password': 'input[data-automation-id="password"]',
            'submit': 'button[data-automation-id="signInSubmitButton"]'
        }


class GreenhouseAdapter:
    """Adapter for Greenhouse application forms."""
    
    def detect_greenhouse(self, page_url: str) -> bool:
        """Check if URL is a Greenhouse page."""
        return 'greenhouse.io' in page_url or 'boards.greenhouse.io' in page_url


class LeverAdapter:
    """Adapter for Lever application forms."""
    
    def detect_lever(self, page_url: str) -> bool:
        """Check if URL is a Lever page."""
        return 'lever.co' in page_url or 'jobs.lever.co' in page_url

