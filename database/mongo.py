"""
MongoDB integration for the Peer Agent system.

This module handles all database operations including interaction logging,
session management, and conversation history storage.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure, ServerSelectionTimeoutError
import uuid

from models.schemas import InteractionRecord, SessionSummary
from config import config

logger = logging.getLogger(__name__)


class MongoDB:
    """
    MongoDB interface for the Peer Agent system.
    
    Handles connections, CRUD operations, and session management.
    """
    
    def __init__(self):
        """Initialize MongoDB connection."""
        self.client: Optional[MongoClient] = None
        self.db = None
        self.interactions_collection = None
        self.sessions_collection = None
        self._connected = False
        
    def connect(self) -> bool:
        """
        Establish connection to MongoDB.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            mongo_config = config.get_mongodb_config()
            
            # Create MongoDB client with timeout settings
            self.client = MongoClient(
                mongo_config["uri"],
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            
            # Test the connection
            self.client.admin.command('ping')
            
            # Set up database and collections
            self.db = self.client[mongo_config["database"]]
            self.interactions_collection = self.db[mongo_config["collection"]]
            self.sessions_collection = self.db["sessions"]
            
            # Create indexes for better performance
            self._create_indexes()
            
            self._connected = True
            logger.info(f"Successfully connected to MongoDB: {mongo_config['database']}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.warning(f"Failed to connect to MongoDB: {str(e)}")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {str(e)}")
            self._connected = False
            return False
    
    def _create_indexes(self):
        """Create database indexes for better query performance."""
        try:
            # Index on session_id for fast session queries
            self.interactions_collection.create_index("session_id")
            
            # Index on timestamp for chronological queries
            self.interactions_collection.create_index([("timestamp", DESCENDING)])
            
            # Compound index for session + timestamp
            self.interactions_collection.create_index([
                ("session_id", 1),
                ("timestamp", DESCENDING)
            ])
            
            # Index on agent_type for analytics
            self.interactions_collection.create_index("agent_type")
            
            # Session collection indexes
            self.sessions_collection.create_index("session_id", unique=True)
            self.sessions_collection.create_index([("last_interaction", DESCENDING)])
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create MongoDB indexes: {str(e)}")
    
    def is_connected(self) -> bool:
        """Check if MongoDB connection is active."""
        if not self._connected or not self.client:
            return False
        
        try:
            self.client.admin.command('ping')
            return True
        except Exception:
            self._connected = False
            return False
    
    def save_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """
        Save an interaction record to the database.
        
        Args:
            interaction_data: Dictionary containing interaction details
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if not self.is_connected():
            logger.warning("MongoDB not connected, cannot save interaction")
            return False
        
        try:
            # Create InteractionRecord from the data
            interaction = InteractionRecord(**interaction_data)
            
            # Insert into database
            result = self.interactions_collection.insert_one(interaction.dict())
            
            if result.inserted_id:
                logger.info(f"Interaction saved with ID: {result.inserted_id}")
                
                # Update session information
                self._update_session(interaction_data.get("session_id"))
                
                return True
            else:
                logger.warning("Failed to save interaction - no ID returned")
                return False
                
        except Exception as e:
            logger.error(f"Error saving interaction: {str(e)}")
            return False
    
    def _update_session(self, session_id: str):
        """Update session summary information."""
        if not session_id or not self.is_connected():
            return
        
        try:
            # Get session statistics
            pipeline = [
                {"$match": {"session_id": session_id}},
                {"$group": {
                    "_id": "$session_id",
                    "total_interactions": {"$sum": 1},
                    "successful_interactions": {
                        "$sum": {"$cond": [{"$eq": ["$success", True]}, 1, 0]}
                    },
                    "agents_used": {"$addToSet": "$agent_type"},
                    "session_start": {"$min": "$timestamp"},
                    "last_interaction": {"$max": "$timestamp"},
                    "total_processing_time": {"$sum": "$processing_time_seconds"}
                }}
            ]
            
            result = list(self.interactions_collection.aggregate(pipeline))
            
            if result:
                session_data = result[0]
                session_data["session_id"] = session_id
                del session_data["_id"]
                
                # Upsert session summary
                self.sessions_collection.replace_one(
                    {"session_id": session_id},
                    session_data,
                    upsert=True
                )
                
                logger.debug(f"Session {session_id} updated")
                
        except Exception as e:
            logger.error(f"Error updating session {session_id}: {str(e)}")
    
    def get_session_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get interaction history for a specific session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of interactions to return
            
        Returns:
            List of interaction records
        """
        if not self.is_connected():
            logger.warning("MongoDB not connected, cannot retrieve session history")
            return []
        
        try:
            cursor = self.interactions_collection.find(
                {"session_id": session_id}
            ).sort("timestamp", DESCENDING).limit(limit)
            
            interactions = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
                interactions.append(doc)
            
            logger.info(f"Retrieved {len(interactions)} interactions for session {session_id}")
            return interactions
            
        except Exception as e:
            logger.error(f"Error retrieving session history: {str(e)}")
            return []
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session summary information.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary or None if not found
        """
        if not self.is_connected():
            return None
        
        try:
            session = self.sessions_collection.find_one({"session_id": session_id})
            
            if session:
                session["_id"] = str(session["_id"])
                logger.debug(f"Retrieved session summary for {session_id}")
                return session
            else:
                logger.debug(f"No session summary found for {session_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving session summary: {str(e)}")
            return None
    
    def get_recent_interactions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent interactions across all sessions.
        
        Args:
            limit: Maximum number of interactions to return
            
        Returns:
            List of recent interaction records
        """
        if not self.is_connected():
            return []
        
        try:
            cursor = self.interactions_collection.find().sort(
                "timestamp", DESCENDING
            ).limit(limit)
            
            interactions = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                interactions.append(doc)
            
            logger.info(f"Retrieved {len(interactions)} recent interactions")
            return interactions
            
        except Exception as e:
            logger.error(f"Error retrieving recent interactions: {str(e)}")
            return []
    
    def get_analytics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get analytics data for the specified time period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Analytics summary
        """
        if not self.is_connected():
            return {}
        
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            pipeline = [
                {"$match": {"timestamp": {"$gte": since_date}}},
                {"$group": {
                    "_id": None,
                    "total_interactions": {"$sum": 1},
                    "successful_interactions": {
                        "$sum": {"$cond": [{"$eq": ["$success", True]}, 1, 0]}
                    },
                    "unique_sessions": {"$addToSet": "$session_id"},
                    "agent_usage": {"$push": "$agent_type"},
                    "avg_processing_time": {"$avg": "$processing_time_seconds"},
                    "total_processing_time": {"$sum": "$processing_time_seconds"}
                }}
            ]
            
            result = list(self.interactions_collection.aggregate(pipeline))
            
            if result:
                analytics = result[0]
                analytics["unique_sessions_count"] = len(analytics.get("unique_sessions", []))
                analytics["success_rate"] = (
                    analytics["successful_interactions"] / analytics["total_interactions"]
                    if analytics["total_interactions"] > 0 else 0
                )
                
                # Count agent usage
                agent_counts = {}
                for agent in analytics.get("agent_usage", []):
                    agent_counts[agent] = agent_counts.get(agent, 0) + 1
                analytics["agent_distribution"] = agent_counts
                
                del analytics["_id"]
                del analytics["unique_sessions"]
                del analytics["agent_usage"]
                
                logger.info(f"Generated analytics for last {days} days")
                return analytics
            else:
                return {
                    "total_interactions": 0,
                    "successful_interactions": 0,
                    "unique_sessions_count": 0,
                    "success_rate": 0,
                    "avg_processing_time": 0,
                    "total_processing_time": 0,
                    "agent_distribution": {}
                }
                
        except Exception as e:
            logger.error(f"Error generating analytics: {str(e)}")
            return {}
    
    def cleanup_old_sessions(self, days: int = 30) -> int:
        """
        Clean up old session data.
        
        Args:
            days: Remove sessions older than this many days
            
        Returns:
            Number of sessions removed
        """
        if not self.is_connected():
            return 0
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Remove old interactions
            interactions_result = self.interactions_collection.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            
            # Remove old sessions
            sessions_result = self.sessions_collection.delete_many({
                "last_interaction": {"$lt": cutoff_date}
            })
            
            total_removed = interactions_result.deleted_count + sessions_result.deleted_count
            
            logger.info(
                f"Cleaned up {interactions_result.deleted_count} interactions "
                f"and {sessions_result.deleted_count} sessions older than {days} days"
            )
            
            return total_removed
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return 0
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("MongoDB connection closed")


# Global MongoDB instance
mongo_db = MongoDB() 