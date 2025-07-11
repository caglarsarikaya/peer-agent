"""
Tests for the FastAPI application endpoints.

This module tests all API endpoints, request/response validation, error handling,
and integration with the PeerAgent system.
"""

import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from tests.conftest import assert_valid_uuid, assert_valid_iso_datetime


class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    @pytest.mark.api
    def test_health_check_success(self, api_client, mock_mongodb_connected):
        """Test successful health check with MongoDB connected."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"]["connected"] is True
        assert "timestamp" in data
        assert assert_valid_iso_datetime(data["timestamp"])
    
    @pytest.mark.api
    def test_health_check_db_disconnected(self, api_client, mock_mongodb_disconnected):
        """Test health check with MongoDB disconnected."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"  # API still works without DB
        assert data["database"]["connected"] is False
        assert "timestamp" in data


class TestMainExecuteEndpoint:
    """Test the main agent execution endpoint."""
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_execute_code_task_success(self, mock_peer_agent, api_client, mock_mongodb_connected):
        """Test successful execution of a code task."""
        # Setup mock
        mock_peer_instance = Mock()
        mock_peer_agent.return_value = mock_peer_instance
        mock_peer_instance.route_task.return_value = {
            "success": True,
            "agent_type": "code",
            "result": {
                "agent": "code",
                "task": "Write a Python function to calculate fibonacci",
                "code": "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
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
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["agent_type"] == "code"
        assert data["result"]["agent"] == "code"
        assert data["result"]["code"] is not None
        assert assert_valid_uuid(data["session_id"])
        assert assert_valid_iso_datetime(data["timestamp"])
        assert "request_id" in data
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_execute_content_task_success(self, mock_peer_agent, api_client, mock_mongodb_connected):
        """Test successful execution of a content task."""
        # Setup mock
        mock_peer_instance = Mock()
        mock_peer_agent.return_value = mock_peer_instance
        mock_peer_instance.route_task.return_value = {
            "success": True,
            "agent_type": "content",
            "result": {
                "agent": "content",
                "task": "Write a blog post about AI",
                "content": "# Artificial Intelligence\n\nAI is transforming our world...",
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
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["agent_type"] == "content"
        assert data["result"]["agent"] == "content"
        assert data["result"]["content"] is not None
        assert assert_valid_uuid(data["session_id"])
    
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
            "result": {"agent": "code", "success": True}
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
                "error": "Unknown task type: Cannot determine appropriate agent for task 'fly to the moon'",
                "suggestion": "Try tasks related to content creation (blog, article) or code generation (function, script)"
            }
        }
        
        # Make request
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": "fly to the moon"}
        )
        
        # Assertions
        assert response.status_code == 422  # Unprocessable Entity
        data = response.json()
        assert data["success"] is False
        assert data["agent_type"] == "unknown"
        assert "error" in data["result"]
        assert "Unknown task type" in data["result"]["error"]
    
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


class TestRequestValidation:
    """Test request validation and error handling."""
    
    @pytest.mark.api
    def test_empty_task_validation(self, api_client):
        """Test validation error for empty task."""
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": ""}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert any("Task cannot be empty" in str(error) for error in data["detail"])
    
    @pytest.mark.api
    def test_whitespace_task_validation(self, api_client):
        """Test validation error for whitespace-only task."""
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": "   "}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.api
    def test_missing_task_field(self, api_client):
        """Test validation error for missing task field."""
        response = api_client.post(
            "/v1/agent/execute",
            json={}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.api
    def test_invalid_json(self, api_client):
        """Test handling of invalid JSON."""
        response = api_client.post(
            "/v1/agent/execute",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    @pytest.mark.api
    def test_invalid_session_id_format(self, api_client):
        """Test validation of session ID format."""
        response = api_client.post(
            "/v1/agent/execute",
            json={
                "task": "Write code",
                "session_id": "invalid-uuid-format"
            }
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.api
    def test_task_too_long(self, api_client):
        """Test validation for excessively long tasks."""
        long_task = "Write code " * 1000  # Very long task
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": long_task}
        )
        
        # Should still work, but might be truncated or handled specially
        assert response.status_code in [200, 422]


class TestAgentCapabilitiesEndpoint:
    """Test the agent capabilities endpoint."""
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_get_capabilities(self, mock_peer_agent, api_client):
        """Test getting agent capabilities."""
        # Setup mock
        mock_peer_instance = Mock()
        mock_peer_agent.return_value = mock_peer_instance
        mock_peer_instance.get_available_agents.return_value = {
            "content": ["blog", "article", "write", "research"],
            "code": ["python", "javascript", "function", "debug"]
        }
        
        response = api_client.get("/v1/agent/capabilities")
        
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert "content" in data["agents"]
        assert "code" in data["agents"]
        assert isinstance(data["agents"]["content"], list)
        assert isinstance(data["agents"]["code"], list)


class TestRoutingInfoEndpoint:
    """Test the routing information endpoint."""
    
    @pytest.mark.api
    def test_get_routing_info(self, api_client):
        """Test getting routing information."""
        response = api_client.get("/v1/agent/routing-info")
        
        assert response.status_code == 200
        data = response.json()
        assert "routing_logic" in data
        assert "supported_agents" in data
        assert isinstance(data["supported_agents"], list)
        assert "content" in data["supported_agents"]
        assert "code" in data["supported_agents"]


class TestStatsEndpoint:
    """Test the statistics endpoint."""
    
    @pytest.mark.api
    def test_get_stats_with_db(self, api_client, mock_mongodb_connected):
        """Test getting stats with database connected."""
        response = api_client.get("/v1/agent/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_interactions" in data
        assert "successful_interactions" in data
        assert "success_rate" in data
        assert isinstance(data["total_interactions"], int)
        assert isinstance(data["success_rate"], float)
    
    @pytest.mark.api
    def test_get_stats_without_db(self, api_client, mock_mongodb_disconnected):
        """Test getting stats with database disconnected."""
        response = api_client.get("/v1/agent/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "Database not available" in data["error"]


class TestSessionEndpoints:
    """Test session-related endpoints."""
    
    @pytest.mark.api
    def test_get_session_history_with_db(self, api_client, mock_mongodb_connected):
        """Test getting session history with database."""
        session_id = "12345678-1234-5678-9012-123456789abc"
        response = api_client.get(f"/v1/agent/sessions/{session_id}/history")
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "interactions" in data
        assert isinstance(data["interactions"], list)
    
    @pytest.mark.api
    def test_get_session_history_without_db(self, api_client, mock_mongodb_disconnected):
        """Test getting session history without database."""
        session_id = "12345678-1234-5678-9012-123456789abc"
        response = api_client.get(f"/v1/agent/sessions/{session_id}/history")
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
    
    @pytest.mark.api
    def test_get_session_history_invalid_uuid(self, api_client):
        """Test getting session history with invalid UUID."""
        response = api_client.get("/v1/agent/sessions/invalid-uuid/history")
        
        assert response.status_code == 422
    
    @pytest.mark.api
    def test_get_recent_sessions(self, api_client, mock_mongodb_connected):
        """Test getting recent sessions."""
        response = api_client.get("/v1/agent/sessions/recent")
        
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_peer_agent_initialization_error(self, mock_peer_agent, api_client):
        """Test handling of PeerAgent initialization error."""
        mock_peer_agent.side_effect = Exception("Failed to initialize PeerAgent")
        
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": "Write code"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Internal server error" in data["detail"]
    
    @pytest.mark.api
    def test_method_not_allowed(self, api_client):
        """Test method not allowed error."""
        response = api_client.get("/v1/agent/execute")  # GET instead of POST
        
        assert response.status_code == 405
    
    @pytest.mark.api
    def test_not_found_endpoint(self, api_client):
        """Test 404 for non-existent endpoint."""
        response = api_client.get("/v1/agent/nonexistent")
        
        assert response.status_code == 404
    
    @pytest.mark.api
    def test_unsupported_media_type(self, api_client):
        """Test unsupported media type error."""
        response = api_client.post(
            "/v1/agent/execute",
            data="task=write code",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code == 422


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
            "result": {"agent": "code", "success": True}
        }
        
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": "Write code"}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
    
    @pytest.mark.api
    @patch('agents.peer_agent.PeerAgent')
    def test_response_structure_consistency(self, mock_peer_agent, api_client, mock_mongodb_connected):
        """Test that all successful responses have consistent structure."""
        # Setup mock
        mock_peer_instance = Mock()
        mock_peer_agent.return_value = mock_peer_instance
        mock_peer_instance.route_task.return_value = {
            "success": True,
            "agent_type": "code",
            "result": {"agent": "code", "success": True}
        }
        
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": "Write code"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required_fields = ["success", "agent_type", "result", "session_id", "timestamp", "request_id"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check field types
        assert isinstance(data["success"], bool)
        assert isinstance(data["agent_type"], str)
        assert isinstance(data["result"], dict)
        assert isinstance(data["session_id"], str)
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["request_id"], str) 