"""
Analytics and reporting endpoints for Peer Agent API.
"""

import logging
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, status

from database.mongo import mongo_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["analytics"])


@router.get("/interactions/recent", response_model=List[Dict[str, Any]])
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


@router.get("/analytics", response_model=Dict[str, Any])
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