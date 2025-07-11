"""
Pytest configuration and shared fixtures for the Peer Agent test suite.

This module provides common fixtures, mock utilities, and test configuration
that can be used across all test files.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import os
import tempfile
import json
from datetime import datetime
from typing import Dict, Any

# Test data constants
MOCK_OPENAI_RESPONSE_CODE = {
    "agent": "code",
    "task": "Write a Python function to calculate fibonacci numbers",
    "code": """def fibonacci(n):
    '''Calculate nth fibonacci number'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Usage example
print(fibonacci(10))""",
    "language": "python",
    "task_type": "function",
    "processing_time_seconds": 2.5,
    "timestamp": "2024-01-01T12:00:00",
    "success": True
}

MOCK_OPENAI_RESPONSE_CONTENT = {
    "agent": "content",
    "task": "Write a brief article about dogs",
    "content": """# Dogs: Man's Best Friend

Dogs have been loyal companions to humans for thousands of years. These remarkable animals are known for their intelligence, loyalty, and diverse breeds.

## Key Characteristics
- Loyal and faithful companions
- Intelligent and trainable
- Available in many breeds and sizes

Dogs make excellent pets and working animals, serving in roles from family companions to service animals.""",
    "search_performed": False,
    "search_results_count": 0,
    "processing_time_seconds": 1.8,
    "timestamp": "2024-01-01T12:00:00",
    "success": True
}

MOCK_OPENAI_ERROR_RESPONSE = {
    "error": "API Error",
    "message": "Mock API error for testing",
    "success": False
}

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-api-key',
        'MONGODB_URI': 'mongodb://localhost:27017/test_db',
        'MONGODB_DATABASE': 'test_db',
        'API_DEBUG': 'True'
    }):
        yield

@pytest.fixture
def mock_openai_success():
    """Mock successful OpenAI API responses."""
    with patch('langchain_openai.ChatOpenAI') as mock_llm:
        mock_instance = Mock()
        mock_instance.predict.return_value = "Mocked LLM response"
        mock_llm.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_openai_error():
    """Mock OpenAI API error responses."""
    with patch('langchain_openai.ChatOpenAI') as mock_llm:
        mock_instance = Mock()
        mock_instance.predict.side_effect = Exception("Mocked API error")
        mock_llm.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_mongodb_connected():
    """Mock MongoDB connection that is connected."""
    with patch('database.mongo.mongo_db') as mock_db:
        mock_db.is_connected.return_value = True
        mock_db.save_interaction.return_value = True
        mock_db.get_session_history.return_value = []
        mock_db.get_analytics.return_value = {
            "total_interactions": 10,
            "successful_interactions": 8,
            "success_rate": 0.8
        }
        yield mock_db

@pytest.fixture
def mock_mongodb_disconnected():
    """Mock MongoDB connection that is disconnected."""
    with patch('database.mongo.mongo_db') as mock_db:
        mock_db.is_connected.return_value = False
        mock_db.save_interaction.return_value = False
        yield mock_db

@pytest.fixture
def mock_duckduckgo_search():
    """Mock DuckDuckGo search responses."""
    with patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_search:
        mock_instance = Mock()
        mock_instance.run.return_value = "Mocked search results about the query"
        mock_search.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def sample_task_requests():
    """Sample task requests for testing."""
    return {
        "code_task": {
            "task": "Write a Python function to calculate fibonacci numbers",
            "expected_agent": "code"
        },
        "content_task": {
            "task": "Write a blog post about artificial intelligence",
            "expected_agent": "content"
        },
        "unknown_task": {
            "task": "fly to the moon",
            "expected_agent": "unknown"
        },
        "empty_task": {
            "task": "",
            "expected_agent": "unknown"
        },
        "whitespace_task": {
            "task": "   ",
            "expected_agent": "unknown"
        }
    }

@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("""
OPENAI_API_KEY=test-key
MONGODB_URI=mongodb://localhost:27017/test_db
API_DEBUG=True
""")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)

@pytest.fixture
def mock_datetime():
    """Mock datetime for consistent testing."""
    mock_dt = datetime(2024, 1, 1, 12, 0, 0)
    with patch('datetime.datetime') as mock:
        mock.now.return_value = mock_dt
        mock.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield mock_dt

@pytest.fixture
def api_client():
    """Create a test client for FastAPI testing."""
    from fastapi.testclient import TestClient
    
    # Import after mocking environment variables
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-api-key',
        'MONGODB_URI': 'mongodb://localhost:27017/test_db'
    }):
        from api.main import app
        with TestClient(app) as client:
            yield client

class MockResponse:
    """Mock HTTP response class for testing."""
    
    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        self.json_data = json_data
        self.status_code = status_code
    
    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

def create_mock_agent_response(agent_type: str, success: bool = True) -> Dict[str, Any]:
    """Create a mock agent response for testing."""
    if agent_type == "code" and success:
        return MOCK_OPENAI_RESPONSE_CODE
    elif agent_type == "content" and success:
        return MOCK_OPENAI_RESPONSE_CONTENT
    else:
        return MOCK_OPENAI_ERROR_RESPONSE

# Test helpers
def assert_valid_uuid(uuid_string: str) -> bool:
    """Assert that a string is a valid UUID."""
    import uuid
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False

def assert_valid_iso_datetime(datetime_string: str) -> bool:
    """Assert that a string is a valid ISO datetime."""
    try:
        datetime.fromisoformat(datetime_string.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False 