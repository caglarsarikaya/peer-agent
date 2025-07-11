"""
Configuration management for the Peer Agent system.

This module handles loading and managing environment variables for API keys,
database connections, and other configuration settings.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for managing environment variables and settings."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1500"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    
    # Search API Configuration (for ContentAgent web search)
    SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")
    SEARCH_ENGINE: str = os.getenv("SEARCH_ENGINE", "serpapi")  # serpapi, duckduckgo, etc.
    
    # MongoDB Configuration
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "peer_agent_db")
    MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "interactions")
    
    # FastAPI Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_DEBUG: bool = os.getenv("API_DEBUG", "False").lower() == "true"
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Agent Configuration
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    CONTENT_AGENT_ENABLED: bool = os.getenv("CONTENT_AGENT_ENABLED", "True").lower() == "true"
    CODE_AGENT_ENABLED: bool = os.getenv("CODE_AGENT_ENABLED", "True").lower() == "true"
    
    @classmethod
    def validate_required_keys(cls) -> dict:
        """
        Validate that required environment variables are set.
        
        Returns:
            dict: Dictionary with validation results and missing keys
        """
        required_keys = {
            "OPENAI_API_KEY": cls.OPENAI_API_KEY,
        }
        
        optional_keys = {
            "SERPAPI_KEY": cls.SERPAPI_KEY,
            "MONGODB_URI": cls.MONGODB_URI,
        }
        
        missing_required = [key for key, value in required_keys.items() if not value]
        missing_optional = [key for key, value in optional_keys.items() if not value]
        
        return {
            "valid": len(missing_required) == 0,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "warnings": [
                f"Missing optional key: {key}" for key in missing_optional
            ]
        }
    
    @classmethod
    def get_openai_config(cls) -> dict:
        """Get OpenAI configuration as a dictionary."""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.OPENAI_MODEL,
            "max_tokens": cls.OPENAI_MAX_TOKENS,
            "temperature": cls.OPENAI_TEMPERATURE
        }
    
    @classmethod
    def get_mongodb_config(cls) -> dict:
        """Get MongoDB configuration as a dictionary."""
        return {
            "uri": cls.MONGODB_URI,
            "database": cls.MONGODB_DATABASE,
            "collection": cls.MONGODB_COLLECTION
        }
    
    @classmethod
    def get_api_config(cls) -> dict:
        """Get FastAPI configuration as a dictionary."""
        return {
            "host": cls.API_HOST,
            "port": cls.API_PORT,
            "debug": cls.API_DEBUG
        }


# Global config instance
config = Config()

# Validate configuration on import
validation_result = config.validate_required_keys()
if not validation_result["valid"]:
    missing = ", ".join(validation_result["missing_required"])
    raise EnvironmentError(
        f"Missing required environment variables: {missing}. "
        f"Please check your .env file or environment variables."
    )

# Log warnings for missing optional keys
if validation_result["warnings"]:
    import logging
    logger = logging.getLogger(__name__)
    for warning in validation_result["warnings"]:
        logger.warning(warning) 