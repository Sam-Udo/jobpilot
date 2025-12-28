"""
Notification Service

Handles user notifications across multiple channels:
1. In-app notifications (stored in DB)
2. Email notifications
3. Push notifications (for mobile)
4. Webhook callbacks (for integrations)

Used when the agent needs user attention:
- Job search complete
- CV ready for review
- Application needs input
- Application submitted
"""

import logging
import json
from typing import Optional, List, Dict, Callable, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from jobpilot.core.config import get_settings

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Types of notifications."""
    
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    ACTION_REQUIRED = "action_required"


class NotificationChannel(str, Enum):
    """Notification delivery channels."""
    
    IN_APP = "in_app"
    EMAIL = "email"
    PUSH = "push"
    WEBHOOK = "webhook"


@dataclass
class Notification:
    """A notification to be sent to a user."""
    
    user_id: str
    title: str
    message: str
    notification_type: NotificationType = NotificationType.INFO
    
    # Additional data
    data: Dict = field(default_factory=dict)
    action_url: Optional[str] = None
    
    # Delivery
    channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.IN_APP]
    )
    
    # Metadata
    id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    read: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'message': self.message,
            'type': self.notification_type.value,
            'data': self.data,
            'action_url': self.action_url,
            'created_at': self.created_at.isoformat(),
            'read': self.read
        }


class NotificationService:
    """
    Service for sending notifications to users.
    
    Supports multiple channels and can be extended with
    custom handlers for specific notification types.
    """
    
    def __init__(self):
        """Initialize notification service."""
        self.settings = get_settings()
        
        # In-memory storage for demo (use Redis/DB in production)
        self._notifications: Dict[str, List[Notification]] = {}
        
        # Channel handlers
        self._handlers: Dict[NotificationChannel, Callable] = {
            NotificationChannel.IN_APP: self._handle_in_app,
            NotificationChannel.EMAIL: self._handle_email,
            NotificationChannel.PUSH: self._handle_push,
            NotificationChannel.WEBHOOK: self._handle_webhook,
        }
        
        # Webhook endpoints per user
        self._webhooks: Dict[str, str] = {}
        
        # Email settings
        self._email_config: Optional[Dict] = None
    
    def configure_email(self, smtp_host: str, smtp_port: int,
                       username: str, password: str, from_address: str):
        """Configure email settings."""
        self._email_config = {
            'host': smtp_host,
            'port': smtp_port,
            'username': username,
            'password': password,
            'from': from_address
        }
    
    def register_webhook(self, user_id: str, webhook_url: str):
        """Register a webhook URL for a user."""
        self._webhooks[user_id] = webhook_url
    
    def send(self, notification: Notification) -> bool:
        """
        Send a notification through all configured channels.
        
        Args:
            notification: Notification to send
            
        Returns:
            True if sent successfully to at least one channel
        """
        import uuid
        notification.id = notification.id or str(uuid.uuid4())
        
        success = False
        
        for channel in notification.channels:
            handler = self._handlers.get(channel)
            if handler:
                try:
                    if handler(notification):
                        success = True
                        logger.info(f"Notification sent via {channel.value}: {notification.title}")
                except Exception as e:
                    logger.error(f"Failed to send via {channel.value}: {e}")
        
        return success
    
    def _handle_in_app(self, notification: Notification) -> bool:
        """Store notification for in-app display."""
        
        user_id = notification.user_id
        if user_id not in self._notifications:
            self._notifications[user_id] = []
        
        self._notifications[user_id].append(notification)
        
        # Keep only last 100 notifications per user
        if len(self._notifications[user_id]) > 100:
            self._notifications[user_id] = self._notifications[user_id][-100:]
        
        return True
    
    def _handle_email(self, notification: Notification) -> bool:
        """Send notification via email."""
        
        if not self._email_config:
            logger.warning("Email not configured")
            return False
        
        # Get user email (would come from user profile in production)
        # For now, skip if email not in data
        user_email = notification.data.get('email')
        if not user_email:
            logger.warning("No email address for user")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self._email_config['from']
            msg['To'] = user_email
            msg['Subject'] = f"JobPilot: {notification.title}"
            
            body = f"""
{notification.message}

---
Data: {json.dumps(notification.data, indent=2)}

{f'Action required: {notification.action_url}' if notification.action_url else ''}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self._email_config['host'], 
                             self._email_config['port']) as server:
                server.starttls()
                server.login(self._email_config['username'],
                           self._email_config['password'])
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False
    
    def _handle_push(self, notification: Notification) -> bool:
        """Send push notification."""
        # Placeholder - would integrate with Firebase/APNs
        logger.info(f"Push notification (not implemented): {notification.title}")
        return False
    
    def _handle_webhook(self, notification: Notification) -> bool:
        """Send notification via webhook."""
        
        webhook_url = self._webhooks.get(notification.user_id)
        if not webhook_url:
            return False
        
        import requests
        
        try:
            response = requests.post(
                webhook_url,
                json=notification.to_dict(),
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Webhook send failed: {e}")
            return False
    
    def get_unread(self, user_id: str) -> List[Notification]:
        """Get unread notifications for a user."""
        
        notifications = self._notifications.get(user_id, [])
        return [n for n in notifications if not n.read]
    
    def get_all(self, user_id: str, limit: int = 50) -> List[Notification]:
        """Get all notifications for a user."""
        
        notifications = self._notifications.get(user_id, [])
        return notifications[-limit:]
    
    def mark_read(self, user_id: str, notification_id: str):
        """Mark a notification as read."""
        
        for n in self._notifications.get(user_id, []):
            if n.id == notification_id:
                n.read = True
                break
    
    def mark_all_read(self, user_id: str):
        """Mark all notifications as read."""
        
        for n in self._notifications.get(user_id, []):
            n.read = True


# ============================================================================
# Notification Factory - Common notification types
# ============================================================================

class NotificationFactory:
    """Factory for creating common notification types."""
    
    @staticmethod
    def jobs_found(user_id: str, job_count: int, 
                  jobs_preview: List[Dict]) -> Notification:
        """Create 'jobs found' notification."""
        return Notification(
            user_id=user_id,
            title="Jobs Found",
            message=f"Found {job_count} matching jobs. Review and select which to apply to.",
            notification_type=NotificationType.ACTION_REQUIRED,
            data={'job_count': job_count, 'preview': jobs_preview},
            action_url="/jobs/select"
        )
    
    @staticmethod
    def cv_ready(user_id: str, company: str, title: str, 
                ats_score: float) -> Notification:
        """Create 'CV ready for review' notification."""
        return Notification(
            user_id=user_id,
            title="CV Ready for Review",
            message=f"Your tailored CV for {company} - {title} is ready (ATS: {ats_score}%)",
            notification_type=NotificationType.ACTION_REQUIRED,
            data={'company': company, 'title': title, 'ats_score': ats_score},
            action_url="/cv/review"
        )
    
    @staticmethod
    def input_needed(user_id: str, company: str, 
                    question: str) -> Notification:
        """Create 'input needed' notification."""
        return Notification(
            user_id=user_id,
            title="Application Needs Input",
            message=f"Application for {company} has a question: {question[:50]}...",
            notification_type=NotificationType.ACTION_REQUIRED,
            data={'company': company, 'question': question},
            action_url="/application/input"
        )
    
    @staticmethod
    def application_submitted(user_id: str, company: str, title: str,
                             confirmation: Optional[str] = None) -> Notification:
        """Create 'application submitted' notification."""
        return Notification(
            user_id=user_id,
            title="Application Submitted",
            message=f"Successfully applied to {company} - {title}",
            notification_type=NotificationType.SUCCESS,
            data={
                'company': company, 
                'title': title,
                'confirmation': confirmation
            }
        )
    
    @staticmethod
    def workflow_complete(user_id: str, submitted: int, 
                         total: int) -> Notification:
        """Create 'workflow complete' notification."""
        return Notification(
            user_id=user_id,
            title="Job Applications Complete",
            message=f"Submitted {submitted} out of {total} applications. Good luck!",
            notification_type=NotificationType.SUCCESS,
            data={'submitted': submitted, 'total': total}
        )
    
    @staticmethod
    def error(user_id: str, error_message: str, 
             context: Optional[Dict] = None) -> Notification:
        """Create error notification."""
        return Notification(
            user_id=user_id,
            title="Error Occurred",
            message=error_message,
            notification_type=NotificationType.ERROR,
            data=context or {}
        )


# Global instance
notification_service = NotificationService()


def get_notification_service() -> NotificationService:
    """Get the global notification service instance."""
    return notification_service

