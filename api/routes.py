"""
FastAPI Routes for JobPilot

REST API and WebSocket endpoints for the job application platform.

Run with: uvicorn jobpilot.api.routes:app --reload
"""

from typing import Optional, List, Dict
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from jobpilot.api.chat_handler import ChatHandler
from jobpilot.services.orchestrator import WorkflowOrchestrator, create_orchestrator
from jobpilot.services.notification import get_notification_service, NotificationFactory
from jobpilot.core.schemas import (
    JobSearchPreferences, WorkflowStatus, ChatMessage,
    JobListing, GeneratedCV, ApplicationMode, LocationType
)

logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="JobPilot API",
    description="AI-powered job application platform",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
orchestrator = create_orchestrator()
chat_handler = ChatHandler(orchestrator)
notification_service = get_notification_service()


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    """Chat message request."""
    user_id: str
    message: str
    attachments: Optional[List[Dict]] = None


class ChatResponse(BaseModel):
    """Chat response."""
    messages: List[Dict]
    conversation_state: str


class WorkflowCreateRequest(BaseModel):
    """Request to create a new workflow."""
    user_id: str
    preferences: Dict
    base_cv: str
    user_profile: Dict


class WorkflowActionRequest(BaseModel):
    """Request to perform a workflow action."""
    action: str  # select_jobs, approve_cv, reject_cv, provide_input
    data: Dict


class JobSelectionRequest(BaseModel):
    """Request to select jobs."""
    job_indices: List[int]


class CVApprovalRequest(BaseModel):
    """Request to approve/reject CV."""
    job_id: str
    approved: bool
    feedback: Optional[str] = None


class FormInputRequest(BaseModel):
    """Request to provide form input."""
    answers: Dict[str, str]


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0"
    }


# ============================================================================
# Chat Endpoints
# ============================================================================

@app.post("/chat", response_model=ChatResponse)
async def send_chat_message(request: ChatRequest):
    """
    Send a chat message and get response.
    
    This is the main interaction endpoint for the conversational interface.
    """
    try:
        responses = chat_handler.process_message(
            user_id=request.user_id,
            message=request.message,
            attachments=request.attachments
        )
        
        ctx = chat_handler.get_conversation(request.user_id)
        
        return ChatResponse(
            messages=[msg.dict() for msg in responses],
            conversation_state=ctx.state.value
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history/{user_id}")
async def get_chat_history(user_id: str, limit: int = 50):
    """Get chat history for a user."""
    ctx = chat_handler.get_conversation(user_id)
    
    messages = ctx.messages[-limit:]
    return {
        "messages": [msg.dict() for msg in messages],
        "conversation_state": ctx.state.value
    }


# ============================================================================
# Workflow Endpoints
# ============================================================================

@app.post("/workflows", response_model=Dict)
async def create_workflow(request: WorkflowCreateRequest):
    """Create a new job application workflow."""
    try:
        prefs = JobSearchPreferences(**request.preferences)
        
        workflow = orchestrator.create_workflow(
            user_id=request.user_id,
            preferences=prefs,
            base_cv=request.base_cv,
            user_profile=request.user_profile
        )
        
        return {
            "workflow_id": workflow.workflow_id,
            "status": workflow.to_status().dict()
        }
    
    except Exception as e:
        logger.error(f"Workflow creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflows/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get workflow status."""
    status = orchestrator.get_workflow_status(workflow_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return status.dict()


@app.post("/workflows/{workflow_id}/search")
async def start_search(workflow_id: str):
    """Start job search for a workflow."""
    workflow = orchestrator.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow = orchestrator.step_search(workflow)
    
    return {
        "status": workflow.to_status().dict(),
        "jobs_found": len(workflow.discovered_jobs)
    }


@app.get("/workflows/{workflow_id}/jobs")
async def get_discovered_jobs(workflow_id: str):
    """Get discovered jobs for a workflow."""
    workflow = orchestrator.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return {
        "jobs": [job.dict() for job in workflow.discovered_jobs],
        "total": len(workflow.discovered_jobs)
    }


@app.post("/workflows/{workflow_id}/select")
async def select_jobs(workflow_id: str, request: JobSelectionRequest):
    """Select jobs to apply to."""
    workflow = orchestrator.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow = orchestrator.step_select_jobs(workflow, request.job_indices)
    
    return {
        "status": workflow.to_status().dict(),
        "selected": len(workflow.selected_jobs)
    }


@app.post("/workflows/{workflow_id}/cv/approve")
async def approve_cv(workflow_id: str, request: CVApprovalRequest):
    """Approve or reject a generated CV."""
    workflow = orchestrator.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow = orchestrator.step_approve_cv(
        workflow,
        request.job_id,
        approved=request.approved,
        feedback=request.feedback
    )
    
    return workflow.to_status().dict()


@app.post("/workflows/{workflow_id}/input")
async def provide_input(workflow_id: str, request: FormInputRequest):
    """Provide answers to form questions."""
    workflow = orchestrator.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow = orchestrator.step_handle_input(workflow, request.answers)
    
    return workflow.to_status().dict()


@app.post("/workflows/{workflow_id}/run")
async def run_workflow(workflow_id: str):
    """Run workflow until next user interaction needed."""
    workflow = orchestrator.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow = orchestrator.run_workflow(workflow)
    
    return workflow.to_status().dict()


# ============================================================================
# Notifications Endpoints
# ============================================================================

@app.get("/notifications/{user_id}")
async def get_notifications(user_id: str, unread_only: bool = False):
    """Get notifications for a user."""
    if unread_only:
        notifications = notification_service.get_unread(user_id)
    else:
        notifications = notification_service.get_all(user_id)
    
    return {
        "notifications": [n.to_dict() for n in notifications],
        "unread_count": len(notification_service.get_unread(user_id))
    }


@app.post("/notifications/{user_id}/read/{notification_id}")
async def mark_notification_read(user_id: str, notification_id: str):
    """Mark a notification as read."""
    notification_service.mark_read(user_id, notification_id)
    return {"status": "ok"}


@app.post("/notifications/{user_id}/read-all")
async def mark_all_notifications_read(user_id: str):
    """Mark all notifications as read."""
    notification_service.mark_all_read(user_id)
    return {"status": "ok"}


# ============================================================================
# WebSocket for Real-time Updates
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[user_id] = websocket
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
    
    async def send_message(self, user_id: str, message: Dict):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(message)
    
    async def broadcast(self, message: Dict):
        for connection in self.active_connections.values():
            await connection.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time chat and notifications.
    
    Messages:
    - {"type": "chat", "message": "..."}
    - {"type": "ping"}
    """
    await manager.connect(user_id, websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "chat":
                # Process chat message
                responses = chat_handler.process_message(
                    user_id=user_id,
                    message=data.get("message", "")
                )
                
                for msg in responses:
                    await manager.send_message(user_id, {
                        "type": "chat_response",
                        "message": msg.dict()
                    })
            
            elif data.get("type") == "ping":
                await manager.send_message(user_id, {"type": "pong"})
            
            elif data.get("type") == "get_status":
                ctx = chat_handler.get_conversation(user_id)
                if ctx.workflow_id:
                    status = orchestrator.get_workflow_status(ctx.workflow_id)
                    if status:
                        await manager.send_message(user_id, {
                            "type": "status",
                            "data": status.dict()
                        })
    
    except WebSocketDisconnect:
        manager.disconnect(user_id)


# ============================================================================
# MCP Integration Endpoints
# ============================================================================

@app.get("/mcp/urls/{workflow_id}")
async def get_mcp_urls(workflow_id: str):
    """
    Get URLs that need Bright Data MCP scraping.
    
    These are URLs that couldn't be scraped via direct APIs.
    Use with: mcp_Bright_Data_scrape_batch()
    """
    workflow = orchestrator.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Get URLs from discovery agent
    mcp_urls = orchestrator.discovery_agent.get_pending_mcp_urls()
    
    return {
        "workday_urls": mcp_urls.get('workday', []),
        "custom_urls": mcp_urls.get('custom', []),
        "aggregator_urls": mcp_urls.get('aggregator', []),
        "total_urls": sum(len(v) for v in mcp_urls.values())
    }


@app.post("/mcp/results/{workflow_id}")
async def submit_mcp_results(workflow_id: str, results: Dict[str, str]):
    """
    Submit results from Bright Data MCP scraping.
    
    Args:
        results: Dict mapping URL to markdown content
    """
    workflow = orchestrator.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Parse results using brightdata_mcp_scraper parsers
    # This would integrate with your existing parsing code
    
    return {
        "status": "received",
        "urls_processed": len(results)
    }


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("JobPilot API starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("JobPilot API shutting down...")

