"""
Agent-related endpoints for Peer Agent API.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError

from models.schemas import (
    TaskRequest, TaskResponse, AgentCapabilities
)
from agents.peer_agent import PeerAgent
from database.mongo import mongo_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["agents"])


@router.get("/agents", response_model=Dict[str, AgentCapabilities])
async def get_agent_capabilities():
    """Get information about available agents and their capabilities."""
    try:
        # Import at runtime to avoid circular imports
        from api.main import peer_agent
        
        if not peer_agent:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Peer Agent not initialized"
            )
        
        capabilities = {}
        
        # Get Content Agent capabilities
        if hasattr(peer_agent, 'content_agent'):
            content_caps = peer_agent.content_agent.get_capabilities()
            capabilities["content"] = AgentCapabilities(**content_caps)
        
        # Get Code Agent capabilities
        if hasattr(peer_agent, 'code_agent'):
            code_caps = peer_agent.code_agent.get_capabilities()
            capabilities["code"] = AgentCapabilities(**code_caps)
        
        return capabilities
        
    except Exception as e:
        logger.error(f"Failed to get agent capabilities: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agent capabilities"
        )


@router.post("/agent/execute", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """
    Main endpoint for task execution.
    
    This endpoint receives a task description, routes it to the appropriate agent,
    and returns the result with detailed metadata.
    """
    start_time = datetime.now()
    request_id = str(uuid.uuid4())
    
    logger.info(f"[{request_id}] Received task: {request.task[:100]}...")
    
    try:
        # Import at runtime to avoid circular imports
        from api.main import peer_agent
        
        # Validate peer agent is available
        if not peer_agent:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Peer Agent system not available"
            )
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"[{request_id}] Session: {session_id}, Processing task...")
        
        # Route task to appropriate agent
        routing_result = peer_agent.route_task(request.task)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Build response
        response = TaskResponse(
            success=routing_result["success"],
            agent_type=routing_result["agent_type"],
            task=request.task,
            result=routing_result["result"],
            session_id=session_id,
            processing_time_seconds=processing_time,
            request_id=request_id
        )
        
        # Save interaction to database
        try:
            interaction_data = {
                "request_id": request_id,
                "session_id": session_id,
                "task": request.task,
                "agent_type": routing_result["agent_type"],
                "success": routing_result["success"],
                "result": routing_result["result"],
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now(),
                "user_preferences": request.preferences or {},
                "error_details": routing_result["result"] if not routing_result["success"] else None
            }
            
            saved = mongo_db.save_interaction(interaction_data)
            if saved:
                logger.debug(f"[{request_id}] Interaction saved to database")
            else:
                logger.warning(f"[{request_id}] Failed to save interaction to database")
                
        except Exception as e:
            logger.error(f"[{request_id}] Error saving interaction to database: {str(e)}")
        
        if routing_result["success"]:
            logger.info(
                f"[{request_id}] Task completed successfully by {routing_result['agent_type']} agent "
                f"in {processing_time:.2f}s"
            )
        else:
            logger.warning(
                f"[{request_id}] Task failed with {routing_result['agent_type']} agent: "
                f"{routing_result['result'].get('error', 'Unknown error')}"
            )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except ValidationError as e:
        logger.error(f"[{request_id}] Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Request validation failed: {str(e)}"
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"[{request_id}] Unexpected error after {processing_time:.2f}s: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your task"
        )


@router.get("/agent/routing", response_model=Dict[str, Any])
async def get_routing_info():
    """Get information about task routing rules and keywords."""
    try:
        # Import at runtime to avoid circular imports
        from api.main import peer_agent
        
        if not peer_agent:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Peer Agent not initialized"
            )
        
        routing_keywords = peer_agent.get_available_agents()
        
        return {
            "routing_keywords": routing_keywords,
            "supported_agents": list(routing_keywords.keys()),
            "routing_algorithm": "keyword-based with scoring",
            "description": "Tasks are routed based on keyword matches with each agent type"
        }
        
    except Exception as e:
        logger.error(f"Failed to get routing info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve routing information"
        ) 