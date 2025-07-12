"""
Session-related endpoints for Peer Agent API.
"""

import logging
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, status

from database.mongo import mongo_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.get("/{session_id}/history", response_model=List[Dict[str, Any]])
async def get_session_history(session_id: str, limit: int = 50):
    """Get interaction history for a specific session."""
    try:
        if not await mongo_db.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available"
            )
        
        history = await mongo_db.get_session_history(session_id, limit)
        return history
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session history"
        )


@router.get("/{session_id}/summary", response_model=Dict[str, Any])
async def get_session_summary(session_id: str):
    """Get session summary information."""
    try:
        if not await mongo_db.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available"
            )
        
        summary = await mongo_db.get_session_summary(session_id)
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