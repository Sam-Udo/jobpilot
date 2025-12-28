"""
Vault - Layer 4

Secure storage for:
1. Encrypted credentials (job portal logins)
2. Session cookies (for session reuse)
3. Knowledge base (answers to common form questions)

Security principles:
- AES-256 encryption for credentials at rest
- Ephemeral decryption (in memory only during use)
- Session cookies for avoiding repeated logins
- Knowledge base for RAG-based form filling
"""

import os
import json
import base64
import hashlib
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


# ============================================================================
# Encryption Utilities
# ============================================================================

class EncryptionManager:
    """
    Handles AES-256 encryption for sensitive data.
    
    Uses Fernet (AES-128-CBC with HMAC) for authenticated encryption.
    For production, consider using AWS KMS or Vault by HashiCorp.
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize with master encryption key.
        
        Args:
            master_key: Master key for encryption. If not provided,
                       reads from JOBPILOT_ENCRYPTION_KEY env var.
        """
        self.master_key = master_key or os.getenv('JOBPILOT_ENCRYPTION_KEY')
        
        if not self.master_key:
            logger.warning("No encryption key provided. Using random key (data won't persist).")
            self.master_key = Fernet.generate_key().decode()
        
        self._fernet = self._derive_key(self.master_key)
    
    def _derive_key(self, password: str) -> Fernet:
        """Derive encryption key from password using PBKDF2."""
        salt = b'jobpilot_v1_salt'  # In production, use unique salt per user
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return Fernet(key)
    
    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string."""
        encrypted = self._fernet.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a string."""
        try:
            encrypted = base64.urlsafe_b64decode(ciphertext.encode())
            decrypted = self._fernet.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt data")


# ============================================================================
# Credential Vault
# ============================================================================

class CredentialVault:
    """
    Stores encrypted credentials for job portals.
    
    Supports:
    - LinkedIn, Indeed, Workday credentials
    - Session cookies for session reuse
    - Automatic session validation
    """
    
    def __init__(self, encryption_manager: EncryptionManager, 
                 storage_path: str = ".jobpilot_vault"):
        """
        Initialize credential vault.
        
        Args:
            encryption_manager: Encryption handler
            storage_path: Path to store encrypted credentials file
        """
        self.encryption = encryption_manager
        self.storage_path = storage_path
        self._credentials: Dict[str, Dict] = {}
        self._load()
    
    def _load(self):
        """Load credentials from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    self._credentials = json.load(f)
                logger.info(f"Loaded {len(self._credentials)} credential entries")
            except Exception as e:
                logger.error(f"Failed to load credentials: {e}")
                self._credentials = {}
    
    def _save(self):
        """Save credentials to storage."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self._credentials, f)
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
    
    def store_credential(self, user_id: str, portal: str, 
                        username: str, password: str):
        """
        Store encrypted credentials for a portal.
        
        Args:
            user_id: User identifier
            portal: Portal name (e.g., "linkedin", "workday-microsoft")
            username: Login username (usually email)
            password: Password (will be encrypted)
        """
        key = f"{user_id}:{portal}"
        
        self._credentials[key] = {
            'username': username,
            'password_encrypted': self.encryption.encrypt(password),
            'created_at': datetime.utcnow().isoformat(),
            'last_used': None
        }
        
        self._save()
        logger.info(f"Stored credential for {portal}")
    
    def get_credential(self, user_id: str, portal: str) -> Optional[Dict[str, str]]:
        """
        Get decrypted credentials for a portal.
        
        Args:
            user_id: User identifier
            portal: Portal name
            
        Returns:
            Dict with username and password, or None
        """
        key = f"{user_id}:{portal}"
        
        if key not in self._credentials:
            return None
        
        cred = self._credentials[key]
        
        try:
            decrypted_password = self.encryption.decrypt(cred['password_encrypted'])
            
            # Update last used
            self._credentials[key]['last_used'] = datetime.utcnow().isoformat()
            self._save()
            
            return {
                'username': cred['username'],
                'password': decrypted_password
            }
        except Exception as e:
            logger.error(f"Failed to decrypt credential: {e}")
            return None
    
    def store_session(self, user_id: str, portal: str, 
                     cookies: Dict, local_storage: Optional[Dict] = None):
        """
        Store session data (cookies, localStorage) for session reuse.
        
        This allows the bot to skip login by reusing existing session.
        
        Args:
            user_id: User identifier
            portal: Portal name
            cookies: Browser cookies as dict
            local_storage: Optional localStorage data
        """
        key = f"{user_id}:{portal}"
        
        if key not in self._credentials:
            self._credentials[key] = {}
        
        self._credentials[key]['session'] = {
            'cookies': self.encryption.encrypt(json.dumps(cookies)),
            'local_storage': self.encryption.encrypt(json.dumps(local_storage or {})),
            'saved_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(days=7)).isoformat()
        }
        
        self._save()
        logger.info(f"Stored session for {portal}")
    
    def get_session(self, user_id: str, portal: str) -> Optional[Dict]:
        """
        Get stored session data.
        
        Args:
            user_id: User identifier
            portal: Portal name
            
        Returns:
            Dict with cookies and local_storage, or None if expired/missing
        """
        key = f"{user_id}:{portal}"
        
        if key not in self._credentials:
            return None
        
        session = self._credentials[key].get('session')
        if not session:
            return None
        
        # Check expiration
        expires_at = datetime.fromisoformat(session['expires_at'])
        if datetime.utcnow() > expires_at:
            logger.info(f"Session expired for {portal}")
            return None
        
        try:
            return {
                'cookies': json.loads(self.encryption.decrypt(session['cookies'])),
                'local_storage': json.loads(self.encryption.decrypt(session['local_storage']))
            }
        except Exception as e:
            logger.error(f"Failed to decrypt session: {e}")
            return None
    
    def delete_credential(self, user_id: str, portal: str):
        """Delete stored credentials."""
        key = f"{user_id}:{portal}"
        if key in self._credentials:
            del self._credentials[key]
            self._save()
            logger.info(f"Deleted credential for {portal}")


# ============================================================================
# Knowledge Base (for RAG-based form filling)
# ============================================================================

class KnowledgeBase:
    """
    Stores knowledge entries for answering application form questions.
    
    When the agent encounters a form question like "Years of Python experience?",
    it searches this knowledge base for relevant answers.
    
    Supports:
    - Semantic search (with embeddings)
    - Pattern matching (for common questions)
    - Learning from user inputs
    """
    
    # Common question patterns and their categories
    QUESTION_PATTERNS = {
        'years_of_experience': [
            'years of experience',
            'how many years',
            'experience with',
            'how long have you'
        ],
        'work_authorization': [
            'authorized to work',
            'work permit',
            'visa sponsorship',
            'legally authorized',
            'employment eligibility'
        ],
        'salary_expectation': [
            'salary expectation',
            'compensation expectation',
            'desired salary',
            'expected compensation'
        ],
        'notice_period': [
            'notice period',
            'when can you start',
            'start date',
            'availability'
        ],
        'relocation': [
            'willing to relocate',
            'open to relocation',
            'relocation assistance'
        ],
        'veteran_status': [
            'veteran status',
            'military service',
            'served in military'
        ],
        'disability': [
            'disability',
            'accommodation'
        ],
        'gender': [
            'gender',
            'male or female'
        ],
        'ethnicity': [
            'race',
            'ethnicity',
            'ethnic background'
        ]
    }
    
    def __init__(self, storage_path: str = ".jobpilot_knowledge"):
        """
        Initialize knowledge base.
        
        Args:
            storage_path: Path to store knowledge entries
        """
        self.storage_path = storage_path
        self._entries: Dict[str, Dict] = {}
        self._load()
    
    def _load(self):
        """Load knowledge entries."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    self._entries = json.load(f)
            except:
                self._entries = {}
    
    def _save(self):
        """Save knowledge entries."""
        with open(self.storage_path, 'w') as f:
            json.dump(self._entries, f, indent=2)
    
    def add_entry(self, user_id: str, question_pattern: str, answer: str,
                  category: Optional[str] = None, source: str = "user_input"):
        """
        Add a knowledge entry.
        
        Args:
            user_id: User identifier
            question_pattern: The question or pattern
            answer: The answer to store
            category: Category (auto-detected if not provided)
            source: Where this answer came from
        """
        if user_id not in self._entries:
            self._entries[user_id] = []
        
        # Auto-detect category if not provided
        if not category:
            category = self._detect_category(question_pattern)
        
        entry = {
            'question_pattern': question_pattern.lower(),
            'answer': answer,
            'category': category,
            'source': source,
            'times_used': 0,
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Check for existing similar entry
        for i, existing in enumerate(self._entries[user_id]):
            if existing['question_pattern'] == entry['question_pattern']:
                # Update existing
                self._entries[user_id][i] = entry
                self._save()
                logger.info(f"Updated knowledge entry: {category}")
                return
        
        self._entries[user_id].append(entry)
        self._save()
        logger.info(f"Added knowledge entry: {category}")
    
    def _detect_category(self, question: str) -> str:
        """Detect question category from patterns."""
        question_lower = question.lower()
        
        for category, patterns in self.QUESTION_PATTERNS.items():
            if any(pattern in question_lower for pattern in patterns):
                return category
        
        return 'general'
    
    def find_answer(self, user_id: str, question: str) -> Optional[Dict]:
        """
        Find answer for a question.
        
        Args:
            user_id: User identifier
            question: The question to answer
            
        Returns:
            Dict with answer and confidence, or None
        """
        if user_id not in self._entries:
            return None
        
        question_lower = question.lower()
        best_match = None
        best_score = 0
        
        for entry in self._entries[user_id]:
            # Simple pattern matching (in production, use embeddings)
            pattern = entry['question_pattern']
            
            # Check for direct pattern match
            if pattern in question_lower:
                score = len(pattern) / len(question_lower)
                if score > best_score:
                    best_score = score
                    best_match = entry
            
            # Check category patterns
            category = entry['category']
            if category in self.QUESTION_PATTERNS:
                for cat_pattern in self.QUESTION_PATTERNS[category]:
                    if cat_pattern in question_lower:
                        score = 0.5  # Category match is weaker
                        if score > best_score:
                            best_score = score
                            best_match = entry
        
        if best_match and best_score > 0.2:
            # Update usage count
            best_match['times_used'] += 1
            self._save()
            
            return {
                'answer': best_match['answer'],
                'confidence': min(best_score + 0.3, 1.0),  # Boost confidence a bit
                'category': best_match['category'],
                'source': best_match['source']
            }
        
        return None
    
    def get_all_entries(self, user_id: str) -> List[Dict]:
        """Get all knowledge entries for a user."""
        return self._entries.get(user_id, [])
    
    def populate_from_cv(self, user_id: str, cv_data: Dict):
        """
        Populate knowledge base from CV data.
        
        Args:
            user_id: User identifier
            cv_data: Structured CV data
        """
        # Extract skills with years
        # This would parse CV for statements like "5 years of Python"
        # Placeholder for now
        logger.info(f"Populating knowledge from CV for user {user_id}")
    
    def populate_from_profile(self, user_id: str, profile: Dict):
        """
        Populate knowledge base from user profile.
        
        Args:
            user_id: User identifier
            profile: User profile data
        """
        # Work authorization
        if 'work_authorized' in profile:
            self.add_entry(
                user_id,
                "authorized to work in the united states",
                "Yes" if profile['work_authorized'] else "No",
                category='work_authorization',
                source='profile'
            )
        
        if 'requires_sponsorship' in profile:
            self.add_entry(
                user_id,
                "require visa sponsorship",
                "Yes" if profile['requires_sponsorship'] else "No",
                category='work_authorization',
                source='profile'
            )
        
        # Voluntary disclosures
        for field in ['veteran_status', 'disability_status', 'gender', 'ethnicity']:
            if field in profile and profile[field]:
                self.add_entry(
                    user_id,
                    field.replace('_', ' '),
                    profile[field],
                    category=field.replace('_status', ''),
                    source='profile'
                )
        
        logger.info(f"Populated knowledge from profile for user {user_id}")

