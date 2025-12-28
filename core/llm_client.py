"""
LLM Client Wrapper for JobPilot

Provides a unified interface for Claude (Anthropic) API.
Can be extended to support other providers if needed.

Usage:
    from jobpilot.core.llm_client import get_llm_client
    
    client = get_llm_client()
    response = client.generate("Your prompt here")
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env file if it exists
def _load_dotenv():
    """Load environment variables from .env file."""
    env_paths = [
        Path.cwd() / '.env',
        Path(__file__).parent.parent.parent.parent / '.env',  # job_scrap_us/.env
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip().strip('"\''))
            logger.debug(f"Loaded .env from {env_path}")
            break

_load_dotenv()


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Any = None
    
    def to_json(self) -> Optional[Dict]:
        """Parse content as JSON if possible."""
        try:
            return json.loads(self.content)
        except json.JSONDecodeError:
            return None


class ClaudeLLMClient:
    """
    Claude (Anthropic) LLM client.
    
    Uses the Anthropic Python SDK for API calls.
    Handles structured output via system prompts.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.3
    ):
        """
        Initialize Claude client.
        
        Args:
            api_key: Anthropic API key. If not provided, reads from
                    ANTHROPIC_API_KEY environment variable.
            model: Claude model to use. Options:
                   - claude-sonnet-4-20250514 (recommended, best balance)
                   - claude-opus-4-20250514 (most capable)
                   - claude-3-5-haiku-20241022 (fastest, cheapest)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var "
                "or pass api_key parameter."
            )
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize Anthropic client
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info(f"Claude client initialized with model: {model}")
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_output: bool = False
    ) -> LLMResponse:
        """
        Generate a response from Claude.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_output: If True, instructs Claude to return valid JSON
            
        Returns:
            LLMResponse with content and metadata
        """
        # Build system prompt
        system = system_prompt or "You are a helpful assistant."
        
        if json_output:
            system += "\n\nIMPORTANT: You must respond with valid JSON only. No markdown, no explanations, just the JSON object."
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                system=system,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            
            # Clean up JSON response if needed
            if json_output:
                content = self._extract_json(content)
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise
    
    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate with conversation history.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            
        Returns:
            LLMResponse
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                system=system_prompt or "You are a helpful assistant.",
                messages=messages
            )
            
            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from response text."""
        text = text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
        
        return text.strip()


class MockLLMClient:
    """
    Mock LLM client for testing without API calls.
    
    Returns predefined responses based on prompt patterns.
    """
    
    def __init__(self):
        """Initialize mock client."""
        logger.info("Using Mock LLM Client (no API calls)")
        self.model = "mock-claude"
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_output: bool = False
    ) -> LLMResponse:
        """Return mock response."""
        
        # Detect what kind of response is needed
        prompt_lower = prompt.lower()
        
        if "job description" in prompt_lower and "requirements" in prompt_lower:
            # JD Analysis
            content = json.dumps({
                "role_title": "Data Engineer",
                "company": "Example Corp",
                "department": "Engineering",
                "seniority_level": "Senior",
                "requirements": [
                    {
                        "text": "5+ years Python experience",
                        "category": "Technical",
                        "priority": "MUST-HAVE",
                        "keywords": ["Python", "experience"]
                    }
                ],
                "total_requirements": 1,
                "tools_mentioned": ["Python", "SQL", "Spark"],
                "methodologies": ["Agile"],
                "certifications_mentioned": [],
                "key_verbs": ["build", "design", "lead"],
                "key_terms": ["data pipeline", "ETL"]
            })
        
        elif "cv" in prompt_lower and "generate" in prompt_lower:
            # CV Generation
            content = json.dumps({
                "candidate_name": "John Doe",
                "contact_info": {"email": "john@example.com"},
                "experiences": [
                    {
                        "company": "Previous Corp",
                        "title": "Data Engineer",
                        "start_date": "2020-01",
                        "end_date": "2023-12",
                        "bullets": [
                            {
                                "text": "Led migration of 50TB data warehouse",
                                "addresses_requirement": "Data engineering",
                                "metrics_included": True
                            }
                        ]
                    }
                ],
                "skills_section": "Python, SQL, Spark, AWS",
                "education": ["BS Computer Science"],
                "certifications": [],
                "ats_score": 88.5,
                "requirements_covered": 12,
                "total_requirements": 15,
                "fabricated_content": ["Project descriptions"],
                "preserved_facts": ["Employment history"]
            })
        
        elif "validate" in prompt_lower or "audit" in prompt_lower:
            # CV Validation
            content = json.dumps({
                "passes_validation": True,
                "hallucinations_found": [],
                "constraint_violations": [],
                "ats_score": 88.5,
                "must_have_coverage": 0.90,
                "should_have_coverage": 0.75,
                "quality_issues": [],
                "improvement_suggestions": [],
                "overall_verdict": "PASS"
            })
        
        else:
            # Generic response
            content = "This is a mock response for testing."
            if json_output:
                content = json.dumps({"response": content})
        
        return LLMResponse(
            content=content,
            model=self.model,
            usage={"input_tokens": 100, "output_tokens": 200}
        )
    
    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """Return mock response for history-based generation."""
        last_message = messages[-1]["content"] if messages else ""
        return self.generate(last_message)


# ============================================================================
# Factory Function
# ============================================================================

def get_llm_client(
    provider: str = "claude",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    use_mock: bool = False
) -> ClaudeLLMClient:
    """
    Get an LLM client instance.
    
    Args:
        provider: LLM provider ("claude" supported)
        api_key: API key (or set ANTHROPIC_API_KEY env var)
        model: Model name override
        use_mock: If True, return mock client (for testing)
        
    Returns:
        LLM client instance
    """
    if use_mock:
        return MockLLMClient()
    
    if provider == "claude":
        return ClaudeLLMClient(
            api_key=api_key,
            model=model or "claude-sonnet-4-20250514"
        )
    
    raise ValueError(f"Unsupported provider: {provider}")


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_json(
    prompt: str,
    client: Optional[ClaudeLLMClient] = None,
    system_prompt: Optional[str] = None
) -> Dict:
    """
    Generate JSON response from Claude.
    
    Args:
        prompt: The prompt
        client: Optional client (creates new one if not provided)
        system_prompt: Optional system context
        
    Returns:
        Parsed JSON dict
    """
    if client is None:
        client = get_llm_client()
    
    response = client.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        json_output=True
    )
    
    result = response.to_json()
    if result is None:
        raise ValueError(f"Failed to parse JSON from response: {response.content[:200]}")
    
    return result

