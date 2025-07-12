"""
Health and basic info endpoints for Peer Agent API.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status, Depends

from models.schemas import HealthResponse
from agents.peer_agent import PeerAgent
from database.mongo import mongo_db
from config import config

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Peer Agent API",
        "version": "v1",
        "docs": "/docs",
        "health": "/health"
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API and agent status."""
    try:
        # Import at runtime to avoid circular imports
        from api.main import peer_agent
        
        # Check agent status
        agents_status = {
            "content": config.CONTENT_AGENT_ENABLED,
            "code": config.CODE_AGENT_ENABLED,
            "peer": peer_agent is not None,
            "database": await mongo_db.is_connected()
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
            status_value = "healthy"
        elif agents_status.get("database", False):
            status_value = "degraded"  # API works but some features may be limited
        else:
            status_value = "degraded"
        
        return HealthResponse(
            status=status_value,
            agents=agents_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service health check failed"
        )


@router.get("/v1/stats", response_model=Dict[str, Any])
async def get_api_stats():
    """Get basic API statistics."""
    # Import at runtime to avoid circular imports
    from api.main import startup_time, peer_agent
    
    uptime = (datetime.now() - startup_time).total_seconds()
    
    return {
        "uptime_seconds": uptime,
        "startup_time": startup_time.isoformat(),
        "version": "v1",
        "agents_available": peer_agent is not None,
        "database_connected": await mongo_db.is_connected(),
        "config": {
            "content_agent_enabled": config.CONTENT_AGENT_ENABLED,
            "code_agent_enabled": config.CODE_AGENT_ENABLED,
            "max_search_results": config.MAX_SEARCH_RESULTS,
            "openai_model": config.OPENAI_MODEL,
            "mongodb_database": config.MONGODB_DATABASE
        }
    } 