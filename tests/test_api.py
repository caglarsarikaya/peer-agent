"""
Tests for the FastAPI application endpoints.

This module tests all API endpoints, request/response validation, error handling,
and integration with the PeerAgent system.
"""

import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from tests.conftest import (
    assert_valid_uuid, 
    assert_valid_iso_datetime,
    assert_response_structure,
    validate_code_response,
    validate_content_response,
    validate_search_integration
)


class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    @pytest.mark.api
    def test_health_check_success(self, api_client, mock_mongodb_connected):
        """Test successful health check with MongoDB connected."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["agents"]["database"] is True
        assert "timestamp" in data
        assert assert_valid_iso_datetime(data["timestamp"])
    
    @pytest.mark.api
    def test_health_check_db_disconnected(self, api_client, mock_mongodb_disconnected):
        """Test health check with MongoDB disconnected."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"  # API still works without DB
        assert data["agents"]["database"] is False
        assert "timestamp" in data


class TestMainExecuteEndpoint:
    """Test the main agent execution endpoint."""
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_execute_code_task_success(self, mock_peer_agent, api_client, mock_mongodb_connected):
        """Test successful execution of a code task."""
        # Setup mock with realistic code response
        mock_peer_instance = Mock()
        mock_peer_agent.return_value = mock_peer_instance
        mock_peer_instance.route_task.return_value = {
            "success": True,
            "agent_type": "code",
            "result": {
                "agent": "code",
                "task": "Write a Python function to calculate fibonacci",
                "code": """def fibonacci(n):
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
        }
        
        # Make request
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": "Write a Python function to calculate fibonacci"}
        )
        
        # Behavioral assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["agent_type"] == "code"
        assert assert_valid_uuid(data["session_id"])
        assert assert_valid_iso_datetime(data["timestamp"])
        assert "request_id" in data
        
        # Validate the result structure
        result = data["result"]
        assert validate_code_response(result, expected_language="python")
        assert assert_response_structure(result, expected_agent="code")
        
        # Verify code contains expected patterns
        code = result["code"]
        assert isinstance(code, str)
        assert len(code.strip()) > 0
        assert "def " in code.lower()  # Should contain function definition
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_execute_content_task_success(self, mock_peer_agent, api_client, mock_mongodb_connected):
        """Test successful execution of a content task."""
        # Setup mock with realistic content response
        mock_peer_instance = Mock()
        mock_peer_agent.return_value = mock_peer_instance
        mock_peer_instance.route_task.return_value = {
            "success": True,
            "agent_type": "content",
            "result": {
                "agent": "content",
                "task": "Write a blog post about AI",
                "content": """# The Future of Artificial Intelligence

Artificial intelligence continues to evolve at an unprecedented pace, transforming various industries and aspects of our daily lives.

## Key Developments
- Advanced machine learning algorithms
- Natural language processing improvements
- Computer vision breakthroughs

These developments are creating new opportunities while also presenting challenges that society must address.""",
                "search_performed": True,
                "search_results_count": 5,
                "processing_time_seconds": 3.2,
                "timestamp": "2024-01-01T12:00:00",
                "success": True
            }
        }
        
        # Make request
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": "Write a blog post about AI"}
        )
        
        # Behavioral assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["agent_type"] == "content"
        assert assert_valid_uuid(data["session_id"])
        
        # Validate the result structure
        result = data["result"]
        assert validate_content_response(result, expected_topics=["ai", "artificial", "intelligence"])
        assert validate_search_integration(result)
        assert assert_response_structure(result, expected_agent="content")
        
        # Verify content quality
        content = result["content"]
        assert isinstance(content, str)
        assert len(content.strip()) > 50  # Substantial content
        ai_concepts = ["ai", "artificial", "intelligence", "machine", "learning"]
        content_lower = content.lower()
        assert any(concept in content_lower for concept in ai_concepts)
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_execute_with_session_id(self, mock_peer_agent, api_client, mock_mongodb_connected):
        """Test execution with provided session ID."""
        # Setup mock
        mock_peer_instance = Mock()
        mock_peer_agent.return_value = mock_peer_instance
        mock_peer_instance.route_task.return_value = {
            "success": True,
            "agent_type": "code",
            "result": {
                "agent": "code", 
                "task": "Write Python code",
                "code": "print('Hello, World!')",
                "language": "python",
                "task_type": "script",
                "success": True
            }
        }
        
        session_id = "12345678-1234-5678-9012-123456789abc"
        
        # Make request
        response = api_client.post(
            "/v1/agent/execute",
            json={
                "task": "Write Python code",
                "session_id": session_id
            }
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["success"] is True
        assert validate_code_response(data["result"], expected_language="python")
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_execute_unknown_task(self, mock_peer_agent, api_client, mock_mongodb_connected):
        """Test execution of unknown task type."""
        # Setup mock
        mock_peer_instance = Mock()
        mock_peer_agent.return_value = mock_peer_instance
        mock_peer_instance.route_task.return_value = {
            "success": False,
            "agent_type": "unknown",
            "result": {
                "error": "Unknown task type: Cannot determine appropriate agent for task 'teleport to mars'",
                "suggestion": "Try tasks related to content creation (blog, article) or code generation (function, script)"
            }
        }
        
        # Make request
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": "teleport to mars"}
        )
        
        # Assertions
        assert response.status_code == 422  # Unprocessable Entity
        data = response.json()
        assert data["success"] is False
        assert data["agent_type"] == "unknown"
        assert "error" in data["result"]
        assert "unknown task type" in data["result"]["error"].lower()
        assert "suggestion" in data["result"]
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_execute_agent_error(self, mock_peer_agent, api_client, mock_mongodb_connected):
        """Test execution when agent encounters an error."""
        # Setup mock
        mock_peer_instance = Mock()
        mock_peer_agent.return_value = mock_peer_instance
        mock_peer_instance.route_task.return_value = {
            "success": False,
            "agent_type": "error",
            "result": {
                "error": "Internal error while processing task",
                "details": "Mock agent error for testing"
            }
        }
        
        # Make request
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": "Write code that fails"}
        )
        
        # Assertions
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert data["agent_type"] == "error"
        assert "error" in data["result"]
        assert isinstance(data["result"]["error"], str)
        assert len(data["result"]["error"]) > 0
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_execute_with_different_task_types(self, mock_peer_agent, api_client, mock_mongodb_connected):
        """Test execution with various task types."""
        mock_peer_instance = Mock()
        mock_peer_agent.return_value = mock_peer_instance
        
        # Test cases for different task types
        test_cases = [
            {
                "task": "Write a Python function",
                "expected_agent": "code",
                "mock_result": {
                    "agent": "code",
                    "task": "Write a Python function",
                    "code": "def example(): return True",
                    "language": "python",
                    "task_type": "function",
                    "success": True
                }
            },
            {
                "task": "Write an article about technology",
                "expected_agent": "content",
                "mock_result": {
                    "agent": "content",
                    "task": "Write an article about technology",
                    "content": "# Technology Article\n\nTechnology continues to advance...",
                    "search_performed": False,
                    "search_results_count": 0,
                    "success": True
                }
            }
        ]
        
        for test_case in test_cases:
            # Setup mock for this test case
            mock_peer_instance.route_task.return_value = {
                "success": True,
                "agent_type": test_case["expected_agent"],
                "result": test_case["mock_result"]
            }
            
            # Make request
            response = api_client.post(
                "/v1/agent/execute",
                json={"task": test_case["task"]}
            )
            
            # Assertions
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["agent_type"] == test_case["expected_agent"]
            
            # Validate result based on agent type
            result = data["result"]
            if test_case["expected_agent"] == "code":
                assert validate_code_response(result)
            elif test_case["expected_agent"] == "content":
                assert validate_content_response(result)
                assert validate_search_integration(result)


class TestRequestValidation:
    """Test request validation and error handling."""
    
    @pytest.mark.api
    def test_empty_task_validation(self, api_client):
        """Test validation of empty task."""
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": ""}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert "task" in data["error"].lower() or "empty" in data["error"].lower()
    
    @pytest.mark.api
    def test_whitespace_task_validation(self, api_client):
        """Test validation of whitespace-only task."""
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": "   "}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    @pytest.mark.api
    def test_missing_task_field(self, api_client):
        """Test validation when task field is missing."""
        response = api_client.post(
            "/v1/agent/execute",
            json={"message": "This is not a task field"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data  # FastAPI validation error format
    
    @pytest.mark.api
    def test_invalid_json(self, api_client):
        """Test handling of invalid JSON."""
        response = api_client.post(
            "/v1/agent/execute",
            data="Invalid JSON content",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    @pytest.mark.api
    def test_invalid_session_id_format(self, api_client):
        """Test validation of invalid session ID format."""
        response = api_client.post(
            "/v1/agent/execute",
            json={
                "task": "Write Python code",
                "session_id": "invalid-uuid-format"
            }
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        # Should indicate UUID format error
        error_msg = str(data["detail"]).lower()
        assert "uuid" in error_msg or "session_id" in error_msg
    
    @pytest.mark.api
    def test_task_too_long(self, api_client):
        """Test validation of excessively long task."""
        long_task = "A" * 10000  # Very long task
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": long_task}
        )
        
        # Should either accept it or reject with appropriate error
        assert response.status_code in [200, 422]
        
        if response.status_code == 422:
            data = response.json()
            assert "error" in data or "detail" in data
    
    @pytest.mark.api
    def test_request_with_extra_fields(self, api_client):
        """Test request with extra fields beyond the schema."""
        with patch('agents.peer_agent.PeerAgent') as mock_peer_agent:
            mock_peer_instance = Mock()
            mock_peer_agent.return_value = mock_peer_instance
            mock_peer_instance.route_task.return_value = {
                "success": True,
                "agent_type": "code",
                "result": {
                    "agent": "code",
                    "task": "Write Python code",
                    "code": "print('Hello')",
                    "language": "python",
                    "task_type": "script",
                    "success": True
                }
            }
            
            response = api_client.post(
                "/v1/agent/execute",
                json={
                    "task": "Write Python code",
                    "extra_field": "should be ignored",
                    "another_field": 123
                }
            )
            
            # Should succeed and ignore extra fields
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


class TestAgentCapabilitiesEndpoint:
    """Test the agent capabilities endpoint."""
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_get_capabilities(self, mock_peer_agent, api_client):
        """Test getting agent capabilities."""
        # Setup mock
        mock_peer_instance = Mock()
        mock_peer_agent.return_value = mock_peer_instance
        mock_peer_instance.get_capabilities.return_value = {
            "supported_tasks": ["code", "content"],
            "supported_languages": ["python", "javascript", "java"],
            "search_tools": ["DuckDuckGo"],
            "max_task_length": 5000
        }
        
        response = api_client.get("/v1/agent/capabilities")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "supported_tasks" in data
        assert "supported_languages" in data
        assert isinstance(data["supported_tasks"], list)
        assert isinstance(data["supported_languages"], list)
        
        # Verify content
        assert len(data["supported_tasks"]) > 0
        assert len(data["supported_languages"]) > 0


class TestRoutingInfoEndpoint:
    """Test the routing information endpoint."""
    
    @pytest.mark.api
    def test_get_routing_info(self, api_client):
        """Test getting routing information."""
        response = api_client.get("/v1/agent/routing-info")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "routing_rules" in data
        assert isinstance(data["routing_rules"], list)
        
        # Verify routing rules structure
        for rule in data["routing_rules"]:
            assert isinstance(rule, dict)
            assert "keywords" in rule
            assert "agent_type" in rule
            assert isinstance(rule["keywords"], list)
            assert isinstance(rule["agent_type"], str)


class TestStatsEndpoint:
    """Test the statistics endpoint."""
    
    @pytest.mark.api
    def test_get_stats_with_db(self, api_client, mock_mongodb_connected):
        """Test getting stats with database connected."""
        response = api_client.get("/v1/agent/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "database_connected" in data
        assert data["database_connected"] is True
        
        # Should have analytics data
        assert "analytics" in data
        assert isinstance(data["analytics"], dict)
        assert "total_interactions" in data["analytics"]
        assert isinstance(data["analytics"]["total_interactions"], int)
    
    @pytest.mark.api
    def test_get_stats_without_db(self, api_client, mock_mongodb_disconnected):
        """Test getting stats without database connection."""
        response = api_client.get("/v1/agent/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "database_connected" in data
        assert data["database_connected"] is False
        
        # Should still return some stats
        assert "analytics" in data


class TestSessionEndpoints:
    """Test session-related endpoints."""
    
    @pytest.mark.api
    def test_get_session_history_with_db(self, api_client, mock_mongodb_connected):
        """Test getting session history with database connected."""
        session_id = "12345678-1234-5678-9012-123456789abc"
        response = api_client.get(f"/v1/agent/sessions/{session_id}/history")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "session_id" in data
        assert "history" in data
        assert isinstance(data["history"], list)
    
    @pytest.mark.api
    def test_get_session_history_without_db(self, api_client, mock_mongodb_disconnected):
        """Test getting session history without database connection."""
        session_id = "12345678-1234-5678-9012-123456789abc"
        response = api_client.get(f"/v1/agent/sessions/{session_id}/history")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "session_id" in data
        assert "history" in data
        assert data["history"] == []  # Empty without DB
    
    @pytest.mark.api
    def test_get_session_history_invalid_uuid(self, api_client):
        """Test getting session history with invalid UUID."""
        response = api_client.get("/v1/agent/sessions/invalid-uuid/history")
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.api
    def test_get_recent_sessions(self, api_client, mock_mongodb_connected):
        """Test getting recent sessions."""
        response = api_client.get("/v1/agent/sessions/recent")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "sessions" in data
        assert isinstance(data["sessions"], list)


class TestErrorHandling:
    """Test error handling across the API."""
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_peer_agent_initialization_error(self, mock_peer_agent, api_client):
        """Test handling of PeerAgent initialization errors."""
        # Mock PeerAgent to raise an error on initialization
        mock_peer_agent.side_effect = Exception("PeerAgent initialization failed")
        
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": "Write Python code"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert isinstance(data["error"], str)
    
    @pytest.mark.api
    def test_method_not_allowed(self, api_client):
        """Test method not allowed error."""
        response = api_client.get("/v1/agent/execute")  # Should be POST
        
        assert response.status_code == 405
    
    @pytest.mark.api
    def test_not_found_endpoint(self, api_client):
        """Test 404 error for non-existent endpoints."""
        response = api_client.get("/v1/agent/nonexistent")
        
        assert response.status_code == 404
    
    @pytest.mark.api
    def test_unsupported_media_type(self, api_client):
        """Test unsupported media type error."""
        response = api_client.post(
            "/v1/agent/execute",
            data="task=Write Python code",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code == 422  # FastAPI converts to validation error


class TestResponseFormat:
    """Test response format consistency."""
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_response_headers(self, mock_peer_agent, api_client, mock_mongodb_connected):
        """Test that responses have correct headers."""
        # Setup mock
        mock_peer_instance = Mock()
        mock_peer_agent.return_value = mock_peer_instance
        mock_peer_instance.route_task.return_value = {
            "success": True,
            "agent_type": "code",
            "result": {
                "agent": "code",
                "task": "Write Python code",
                "code": "print('Hello')",
                "language": "python",
                "task_type": "script",
                "success": True
            }
        }
        
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": "Write Python code"}
        )
        
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
        
        # Check for common security headers
        headers = response.headers
        assert "content-type" in headers
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_response_structure_consistency(self, mock_peer_agent, api_client, mock_mongodb_connected):
        """Test that all successful responses have consistent structure."""
        mock_peer_instance = Mock()
        mock_peer_agent.return_value = mock_peer_instance
        
        # Test with different agent types
        test_cases = [
            {
                "agent_type": "code",
                "result": {
                    "agent": "code",
                    "task": "Write Python code",
                    "code": "print('test')",
                    "language": "python",
                    "task_type": "script",
                    "success": True
                }
            },
            {
                "agent_type": "content",
                "result": {
                    "agent": "content",
                    "task": "Write an article",
                    "content": "# Article\n\nContent here...",
                    "search_performed": False,
                    "search_results_count": 0,
                    "success": True
                }
            }
        ]
        
        for test_case in test_cases:
            # Setup mock
            mock_peer_instance.route_task.return_value = {
                "success": True,
                "agent_type": test_case["agent_type"],
                "result": test_case["result"]
            }
            
            response = api_client.post(
                "/v1/agent/execute",
                json={"task": "Test task"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Check consistent top-level structure
            required_fields = ["success", "agent_type", "result", "session_id", "timestamp", "request_id"]
            for field in required_fields:
                assert field in data, f"Missing field {field} in response"
            
            # Check data types
            assert isinstance(data["success"], bool)
            assert isinstance(data["agent_type"], str)
            assert isinstance(data["result"], dict)
            assert isinstance(data["session_id"], str)
            assert isinstance(data["timestamp"], str)
            assert isinstance(data["request_id"], str)
            
            # Validate UUIDs and timestamps
            assert assert_valid_uuid(data["session_id"])
            assert assert_valid_iso_datetime(data["timestamp"])
    
    @pytest.mark.api
    def test_error_response_structure(self, api_client):
        """Test that error responses have consistent structure."""
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": ""}  # Empty task to trigger error
        )
        
        assert response.status_code == 422
        data = response.json()
        
        # Error responses should have consistent structure
        assert "success" in data
        assert data["success"] is False
        assert "error" in data or "detail" in data
        
        # Should have timestamp
        if "timestamp" in data:
            assert assert_valid_iso_datetime(data["timestamp"])
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_response_timing_consistency(self, mock_peer_agent, api_client, mock_mongodb_connected):
        """Test that response timing is consistent."""
        mock_peer_instance = Mock()
        mock_peer_agent.return_value = mock_peer_instance
        mock_peer_instance.route_task.return_value = {
            "success": True,
            "agent_type": "code",
            "result": {
                "agent": "code",
                "task": "Write Python code",
                "code": "print('Hello')",
                "language": "python",
                "task_type": "script",
                "processing_time_seconds": 1.5,
                "success": True
            }
        }
        
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": "Write Python code"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check processing time is included
        result = data["result"]
        assert "processing_time_seconds" in result
        assert isinstance(result["processing_time_seconds"], (int, float))
        assert result["processing_time_seconds"] >= 0 