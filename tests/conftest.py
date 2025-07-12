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
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

# Test data constants with more realistic variability
MOCK_OPENAI_RESPONSE_CODE_VARIANTS = [
    {
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
    },
    {
        "agent": "code",
        "task": "Write a Python function to calculate fibonacci numbers",
        "code": """def fib(n):
    \"\"\"Returns the nth fibonacci number\"\"\"
    if n <= 1:
        return n
    else:
        return fib(n-1) + fib(n-2)

# Test it
for i in range(10):
    print(f"fib({i}) = {fib(i)}")""",
        "language": "python",
        "task_type": "function",
        "processing_time_seconds": 1.8,
        "timestamp": "2024-01-01T12:00:00",
        "success": True
    }
]

MOCK_OPENAI_RESPONSE_CONTENT_VARIANTS = [
    {
        "agent": "content",
        "task": "Write a brief article about dogs",
        "content": """# Dogs: Loyal Companions

Dogs have been faithful companions to humans for thousands of years. These amazing animals are known for their loyalty, intelligence, and diverse breeds.

## Key Traits
- Extremely loyal and devoted
- Highly intelligent and trainable
- Come in many different breeds and sizes

Dogs serve as excellent pets and working animals, from family companions to service dogs.""",
        "search_performed": False,
        "search_results_count": 0,
        "processing_time_seconds": 1.8,
        "timestamp": "2024-01-01T12:00:00",
        "success": True
    },
    {
        "agent": "content",
        "task": "Write a brief article about dogs",
        "content": """# Man's Best Friend: Understanding Dogs

For millennia, dogs have stood by humans as trusted companions. These remarkable creatures exhibit intelligence, loyalty, and remarkable diversity in their breeds.

## Notable Characteristics
- Unwavering loyalty to their owners
- Remarkable intelligence and learning ability
- Incredible variety across different breeds

Whether as beloved family pets or dedicated working animals, dogs continue to play vital roles in human society.""",
        "search_performed": False,
        "search_results_count": 0,
        "processing_time_seconds": 2.1,
        "timestamp": "2024-01-01T12:00:00",
        "success": True
    }
]

# Utility functions for semantic content validation
def validate_code_response(response: Dict[str, Any], expected_language: str = None) -> bool:
    """Validate that a code response has the expected structure and content."""
    if not isinstance(response, dict):
        return False
    
    # Check required fields
    required_fields = ["agent", "task", "code", "language", "success"]
    if not all(field in response for field in required_fields):
        return False
    
    # Check that code is actually present
    if not response.get("code") or not isinstance(response["code"], str):
        return False
    
    # Check language if specified
    if expected_language and response.get("language") != expected_language:
        return False
    
    # Check for basic code patterns (functions, classes, etc.)
    code = response["code"]
    return bool(re.search(r'(def |class |import |from |=|if |for |while )', code))

def validate_content_response(response: Dict[str, Any], expected_topics: List[str] = None) -> bool:
    """Validate that a content response has the expected structure and topics."""
    if not isinstance(response, dict):
        return False
    
    # Check required fields
    required_fields = ["agent", "task", "content", "success"]
    if not all(field in response for field in required_fields):
        return False
    
    # Check that content is actually present
    content = response.get("content")
    if not content or not isinstance(content, str):
        return False
    
    # Check for basic content structure (headers, paragraphs)
    has_structure = bool(re.search(r'(#|##|###|\n\n)', content))
    
    # Check for expected topics if specified
    if expected_topics:
        content_lower = content.lower()
        has_topics = any(topic.lower() in content_lower for topic in expected_topics)
        return has_structure and has_topics
    
    return has_structure and len(content.strip()) > 50

def validate_search_integration(response: Dict[str, Any]) -> bool:
    """Validate that search integration fields are present and consistent."""
    if not isinstance(response, dict):
        return False
    
    # Check search-related fields exist
    search_fields = ["search_performed", "search_results_count"]
    if not all(field in response for field in search_fields):
        return False
    
    # If search was performed, should have results count >= 0
    if response.get("search_performed") and response.get("search_results_count") < 0:
        return False
    
    # If search wasn't performed, results count should be 0
    if not response.get("search_performed") and response.get("search_results_count") != 0:
        return False
    
    return True

def extract_key_concepts(text: str) -> List[str]:
    """Extract key concepts from text for semantic comparison."""
    # Simple keyword extraction - remove common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
    
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word not in common_words and len(word) > 2]

def semantic_similarity(text1: str, text2: str, threshold: float = 0.3) -> bool:
    """Check if two texts have semantic similarity based on key concepts."""
    concepts1 = set(extract_key_concepts(text1))
    concepts2 = set(extract_key_concepts(text2))
    
    if not concepts1 or not concepts2:
        return False
    
    intersection = concepts1.intersection(concepts2)
    union = concepts1.union(concepts2)
    
    similarity = len(intersection) / len(union) if union else 0
    return similarity >= threshold

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
def realistic_llm_mock():
    """Mock LLM with realistic variability in responses."""
    with patch('langchain_openai.ChatOpenAI') as mock_llm:
        mock_instance = Mock()
        
        # Create a side effect that returns different responses
        def dynamic_response(prompt):
            if any(keyword in prompt.lower() for keyword in ['fibonacci', 'fib', 'function']):
                return MOCK_OPENAI_RESPONSE_CODE_VARIANTS[0]["code"]
            elif any(keyword in prompt.lower() for keyword in ['dog', 'pet', 'animal']):
                return MOCK_OPENAI_RESPONSE_CONTENT_VARIANTS[0]["content"]
            elif any(keyword in prompt.lower() for keyword in ['python', 'code', 'script']):
                return """def example_function():
    '''Example function'''
    return "Hello, World!"

# Usage
result = example_function()
print(result)"""
            elif any(keyword in prompt.lower() for keyword in ['article', 'blog', 'write']):
                return """# Example Article

This is an example article with proper structure and content.

## Key Points
- Important information
- Relevant details
- Useful insights

The article provides comprehensive coverage of the topic."""
            else:
                return "I understand your request. Let me provide a helpful response."
        
        mock_instance.predict.side_effect = dynamic_response
        mock_llm.return_value = mock_instance
        yield mock_instance

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
        # Use AsyncMock for async methods
        mock_db.is_connected = AsyncMock(return_value=True)
        mock_db.save_interaction = AsyncMock(return_value=True)
        mock_db.get_session_history = AsyncMock(return_value=[])
        mock_db.get_analytics = AsyncMock(return_value={
            "total_interactions": 10,
            "successful_interactions": 8,
            "success_rate": 0.8
        })
        yield mock_db

@pytest.fixture
def mock_mongodb_disconnected():
    """Mock MongoDB connection that is disconnected."""
    with patch('database.mongo.mongo_db') as mock_db:
        # Use AsyncMock for async methods
        mock_db.is_connected = AsyncMock(return_value=False)
        mock_db.save_interaction = AsyncMock(return_value=False)
        yield mock_db

@pytest.fixture
def flexible_search_mock():
    """Mock search with realistic variability."""
    with patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_search:
        mock_instance = Mock()
        
        def dynamic_search(query):
            query_lower = query.lower()
            if 'ai' in query_lower or 'artificial' in query_lower:
                return "AI Research Updates\nLatest artificial intelligence developments\nMachine Learning News\nBreakthroughs in AI technology"
            elif 'python' in query_lower or 'programming' in query_lower:
                return "Python Programming Guide\nLatest Python updates\nCoding Best Practices\nPython development tips"
            elif 'dog' in query_lower or 'pet' in query_lower:
                return "Dog Care Information\nPet training guides\nCanine health tips\nBest dog breeds for families"
            else:
                return f"Search results for {query}\nRelevant information about the topic\nUseful resources and links"
        
        mock_instance.run.side_effect = dynamic_search
        mock_search.return_value = mock_instance
        yield mock_instance

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
    base_response = {
        "agent": agent_type,
        "task": f"Mock {agent_type} task",
        "processing_time_seconds": 2.0,
        "timestamp": "2024-01-01T12:00:00",
        "success": success
    }
    
    if agent_type == "code":
        base_response.update({
            "code": "def mock_function():\n    return 'mock result'",
            "language": "python",
            "task_type": "function"
        })
    elif agent_type == "content":
        base_response.update({
            "content": "# Mock Content\n\nThis is mock content for testing.",
            "search_performed": False,
            "search_results_count": 0
        })
    
    return base_response

def assert_valid_uuid(uuid_string: str) -> bool:
    """Assert that a string is a valid UUID."""
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

def assert_response_structure(response: Dict[str, Any], expected_agent: str = None) -> bool:
    """Assert that a response has the expected structure."""
    # Check basic structure
    required_fields = ["success", "agent", "task", "timestamp"]
    if not all(field in response for field in required_fields):
        return False
    
    # Check agent-specific fields
    if expected_agent == "code":
        return "code" in response and "language" in response
    elif expected_agent == "content":
        return "content" in response
    
    return True

# Test categories for better organization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "api: API tests")
    config.addinivalue_line("markers", "agent: Agent tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "llm: LLM-dependent tests")
    config.addinivalue_line("markers", "search: Search-dependent tests") 