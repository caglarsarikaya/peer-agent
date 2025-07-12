"""
Peer Agent API - FastAPI application for the agentic task delegation system.

This module implements the main API application setup for the Peer Agent system,
providing a robust HTTP interface for task delegation to specialized AI agents.
"""

import logging
import sys
from datetime import datetime
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local imports
from agents.peer_agent import PeerAgent
from database.mongo import mongo_db
from config import config

# Import routers
from api.routers.health import router as health_router
from api.routers.agents import router as agents_router
from api.routers.sessions import router as sessions_router
from api.routers.analytics import router as analytics_router

# Import middleware
from api.middleware.exceptions import register_exception_handlers

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global variables
peer_agent: PeerAgent = None
startup_time: datetime = datetime.now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events for the FastAPI application."""
    # Startup
    global peer_agent
    
    try:
        logger.info("Starting Peer Agent API...")
        logger.info(f"API Configuration: {config.get_api_config()}")
        
        # Initialize MongoDB connection
        mongo_connected = await mongo_db.connect()
        if mongo_connected:
            logger.info("MongoDB connected successfully")
        else:
            logger.warning("MongoDB connection failed - continuing without database logging")
        
        # Initialize the Peer Agent
        peer_agent = PeerAgent()
        
        logger.info("Peer Agent API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start Peer Agent API: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Peer Agent API...")
    
    # Close MongoDB connection
    await mongo_db.close()


# Initialize FastAPI app
app = FastAPI(
    title="Peer Agent API",
    description="AI-powered task delegation system with specialized agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this more restrictively in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Note: Dependencies are handled via runtime imports in router modules
# to avoid circular import issues


# Register exception handlers
register_exception_handlers(app)

# Include routers
app.include_router(health_router)
app.include_router(agents_router)
app.include_router(sessions_router)
app.include_router(analytics_router)



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