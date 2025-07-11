"""
Peer Agent API - FastAPI application for the agentic task delegation system.

This module implements the main API endpoints for the Peer Agent system,
providing a robust HTTP interface for task delegation to specialized AI agents.
"""

import logging
import sys
from datetime import datetime
from typing import Dict, Any, Union, List
import traceback
import uuid

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

# Local imports
from models.schemas import (
    TaskRequest, TaskResponse, ErrorResponse, HealthResponse, AgentCapabilities
)
from agents.peer_agent import PeerAgent
from database.mongo import mongo_db
from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Peer Agent API",
    description="AI-powered task delegation system with specialized agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this more restrictively in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
peer_agent: PeerAgent = None
startup_time: datetime = datetime.now()


@app.on_event("startup")
async def startup_event():
    """Initialize the Peer Agent system on application startup."""
    global peer_agent
    
    try:
        logger.info("Starting Peer Agent API...")
        logger.info(f"API Configuration: {config.get_api_config()}")
        
        # Initialize MongoDB connection
        mongo_connected = mongo_db.connect()
        if mongo_connected:
            logger.info("MongoDB connected successfully")
        else:
            logger.warning("MongoDB connection failed - continuing without database logging")
        
        # Initialize the Peer Agent
        peer_agent = PeerAgent()
        
        logger.info("Peer Agent API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start Peer Agent API: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Peer Agent API...")
    
    # Close MongoDB connection
    mongo_db.close()


# Exception handlers

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(f"Validation error for {request.url}: {exc.errors()}")
    
    error_response = ErrorResponse(
        error_type="ValidationError",
        error_message=f"Request validation failed: {exc.errors()[0]['msg']}",
        request_id=str(uuid.uuid4())
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP error for {request.url}: {exc.detail}")
    
    error_response = ErrorResponse(
        error_type="HTTPException",
        error_message=exc.detail,
        request_id=str(uuid.uuid4())
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error for {request.url}: {str(exc)}")
    logger.error(traceback.format_exc())
    
    error_response = ErrorResponse(
        error_type="InternalServerError",
        error_message="An unexpected error occurred. Please try again later.",
        request_id=str(uuid.uuid4())
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.dict()
    )


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Peer Agent API",
        "version": "v1",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API and agent status."""
    try:
        # Check agent status
        agents_status = {
            "content": config.CONTENT_AGENT_ENABLED,
            "code": config.CODE_AGENT_ENABLED,
            "peer": peer_agent is not None,
            "database": mongo_db.is_connected()
        }
        
        # Verify agents are actually working
        if peer_agent:
            try:
                # Quick test of routing functionality
                test_result = peer_agent.analyze_task("test")
                agents_status["routing"] = True
            except Exception as e:
                logger.warning(f"Agent routing test failed: {str(e)}")
                agents_status["routing"] = False
        
        # Determine overall health status
        critical_services = ["peer", "routing"]
        critical_status = all(agents_status.get(service, False) for service in critical_services)
        
        if critical_status:
            status = "healthy"
        elif agents_status.get("database", False):
            status = "degraded"  # API works but some features may be limited
        else:
            status = "degraded"
        
        return HealthResponse(
            status=status,
            agents=agents_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service health check failed"
        )


@app.get("/v1/agents", response_model=Dict[str, AgentCapabilities])
async def get_agent_capabilities():
    """Get information about available agents and their capabilities."""
    try:
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


@app.post("/v1/agent/execute", response_model=TaskResponse)
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
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your task"
        )


@app.get("/v1/agent/routing", response_model=Dict[str, Any])
async def get_routing_info():
    """Get information about task routing rules and keywords."""
    try:
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


# Database and analytics endpoints

@app.get("/v1/sessions/{session_id}/history", response_model=List[Dict[str, Any]])
async def get_session_history(session_id: str, limit: int = 50):
    """Get interaction history for a specific session."""
    try:
        if not mongo_db.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available"
            )
        
        history = mongo_db.get_session_history(session_id, limit)
        return history
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session history"
        )


@app.get("/v1/sessions/{session_id}/summary", response_model=Dict[str, Any])
async def get_session_summary(session_id: str):
    """Get session summary information."""
    try:
        if not mongo_db.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available"
            )
        
        summary = mongo_db.get_session_summary(session_id)
        if summary is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session summary"
        )


@app.get("/v1/interactions/recent", response_model=List[Dict[str, Any]])
async def get_recent_interactions(limit: int = 100):
    """Get recent interactions across all sessions."""
    try:
        if not mongo_db.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available"
            )
        
        interactions = mongo_db.get_recent_interactions(limit)
        return interactions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving recent interactions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recent interactions"
        )


@app.get("/v1/analytics", response_model=Dict[str, Any])
async def get_analytics(days: int = 7):
    """Get analytics data for the specified time period."""
    try:
        if not mongo_db.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available"
            )
        
        analytics = mongo_db.get_analytics(days)
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analytics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics"
        )


# Additional utility endpoints

@app.get("/v1/stats", response_model=Dict[str, Any])
async def get_api_stats():
    """Get basic API statistics."""
    uptime = (datetime.now() - startup_time).total_seconds()
    
    return {
        "uptime_seconds": uptime,
        "startup_time": startup_time.isoformat(),
        "version": "v1",
        "agents_available": peer_agent is not None,
        "database_connected": mongo_db.is_connected(),
        "config": {
            "content_agent_enabled": config.CONTENT_AGENT_ENABLED,
            "code_agent_enabled": config.CODE_AGENT_ENABLED,
            "max_search_results": config.MAX_SEARCH_RESULTS,
            "openai_model": config.OPENAI_MODEL,
            "mongodb_database": config.MONGODB_DATABASE
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_DEBUG,
        log_level=config.LOG_LEVEL.lower()
    ) 