"""
Pydantic schemas for request/response validation and database models.

This module defines the data models used throughout the Peer Agent API,
including request validation, response formatting, and database schemas.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
import uuid


class TaskRequest(BaseModel):
    """Request schema for task execution endpoint."""
    
    task: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The task description to be processed by the agents",
        example="Write a blog post about artificial intelligence trends"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for tracking user sessions",
        example="550e8400-e29b-41d4-a716-446655440000"
    )
    
    preferences: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional user preferences for task processing",
        example={"language": "python", "style": "professional"}
    )
    
    @validator('task')
    def validate_task(cls, v):
        """Validate task is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError('Task cannot be empty or contain only whitespace')
        return v.strip()
    
    @validator('session_id')
    def validate_session_id(cls, v):
        """Validate session_id format if provided."""
        if v is not None:
            try:
                uuid.UUID(v)
            except ValueError:
                raise ValueError('session_id must be a valid UUID format')
        return v


class TaskResponse(BaseModel):
    """Response schema for task execution results."""
    
    success: bool = Field(
        ...,
        description="Whether the task was processed successfully"
    )
    
    agent_type: str = Field(
        ...,
        description="The type of agent that processed the task",
        example="content"
    )
    
    task: str = Field(
        ...,
        description="The original task description"
    )
    
    result: Dict[str, Any] = Field(
        ...,
        description="The detailed result from the agent"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID if provided in request"
    )
    
    processing_time_seconds: float = Field(
        ...,
        description="Time taken to process the task in seconds"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the task was processed"
    )
    
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this request"
    )


class ErrorResponse(BaseModel):
    """Response schema for error cases."""
    
    success: bool = Field(
        default=False,
        description="Always false for error responses"
    )
    
    error_type: str = Field(
        ...,
        description="Type of error that occurred",
        example="ValidationError"
    )
    
    error_message: str = Field(
        ...,
        description="Human-readable error message",
        example="Task cannot be empty"
    )
    
    task: Optional[str] = Field(
        default=None,
        description="The task that caused the error, if available"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID if provided in request"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the error occurred"
    )
    
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this request"
    )


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    
    status: str = Field(
        default="healthy",
        description="Health status of the API"
    )
    
    version: str = Field(
        default="v1",
        description="API version"
    )
    
    agents: Dict[str, bool] = Field(
        ...,
        description="Status of available agents"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Current timestamp"
    )


class AgentCapabilities(BaseModel):
    """Schema for agent capabilities information."""
    
    agent_type: str = Field(
        ...,
        description="Type of the agent"
    )
    
    capabilities: List[str] = Field(
        ...,
        description="List of agent capabilities"
    )
    
    tools: List[str] = Field(
        ...,
        description="Tools available to the agent"
    )
    
    enabled: bool = Field(
        ...,
        description="Whether the agent is currently enabled"
    )
    
    model: Optional[str] = Field(
        default=None,
        description="AI model used by the agent, if applicable"
    )


# Database Models (for MongoDB integration in Step 4)

class InteractionRecord(BaseModel):
    """Database schema for storing interaction records."""
    
    request_id: str = Field(
        ...,
        description="Unique identifier for the request"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID if provided"
    )
    
    task: str = Field(
        ...,
        description="The task description"
    )
    
    agent_type: str = Field(
        ...,
        description="Agent that processed the task"
    )
    
    success: bool = Field(
        ...,
        description="Whether processing was successful"
    )
    
    result: Dict[str, Any] = Field(
        ...,
        description="The complete result from processing"
    )
    
    processing_time_seconds: float = Field(
        ...,
        description="Processing time in seconds"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the interaction occurred"
    )
    
    user_preferences: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="User preferences if provided"
    )
    
    error_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error details if processing failed"
    )
    
    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SessionSummary(BaseModel):
    """Schema for session summary information."""
    
    session_id: str = Field(
        ...,
        description="Session identifier"
    )
    
    total_interactions: int = Field(
        ...,
        description="Total number of interactions in session"
    )
    
    successful_interactions: int = Field(
        ...,
        description="Number of successful interactions"
    )
    
    agents_used: List[str] = Field(
        ...,
        description="List of agents used in this session"
    )
    
    session_start: datetime = Field(
        ...,
        description="When the session started"
    )
    
    last_interaction: datetime = Field(
        ...,
        description="Last interaction timestamp"
    )
    
    total_processing_time: float = Field(
        ...,
        description="Total processing time for all interactions"
    ) 