"""
Integration tests for the complete Peer Agent system.

This module tests the full pipeline from API requests through PeerAgent routing
to sub-agent execution and response formatting.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from tests.conftest import (
    assert_valid_uuid, 
    assert_valid_iso_datetime,
    MOCK_OPENAI_RESPONSE_CODE,
    MOCK_OPENAI_RESPONSE_CONTENT
)


class TestFullPipelineIntegration:
    """Test the complete request-response pipeline."""
    
    @pytest.mark.integration
    def test_code_task_full_pipeline(self, api_client, mock_mongodb_connected):
        """Test complete pipeline for code generation task."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            # Setup LLM mock
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.predict.return_value = """def fibonacci(n):
    '''Calculate nth fibonacci number'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
print(fibonacci(10))"""
            
            # Make API request
            response = api_client.post(
                "/v1/agent/execute",
                json={"task": "Write a Python function to calculate fibonacci numbers"}
            )
            
            # Verify full pipeline
            assert response.status_code == 200
            data = response.json()
            
            # API response structure
            assert data["success"] is True
            assert data["agent_type"] == "code"
            assert assert_valid_uuid(data["session_id"])
            assert assert_valid_iso_datetime(data["timestamp"])
            
            # Agent response content
            result = data["result"]
            assert result["agent"] == "code"
            assert result["language"] == "python"
            assert result["task_type"] == "function"
            assert "fibonacci" in result["code"]
            assert result["success"] is True
            
            # MongoDB interaction logged
            mock_mongodb_connected.save_interaction.assert_called_once()
    
    @pytest.mark.integration
    def test_content_task_full_pipeline(self, api_client, mock_mongodb_connected):
        """Test complete pipeline for content generation task."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_search:
            
            # Setup mocks
            mock_llm_instance = Mock()
            mock_search_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_search.return_value = mock_search_instance
            
            mock_search_instance.run.return_value = "Recent research about artificial intelligence shows rapid progress..."
            mock_llm_instance.predict.return_value = """# The Future of Artificial Intelligence

Artificial intelligence continues to evolve at an unprecedented pace. Recent developments show promising applications across various industries.

## Key Trends
- Machine learning advancements
- Natural language processing improvements
- Computer vision breakthroughs

AI is transforming how we work, communicate, and solve complex problems."""
            
            # Make API request
            response = api_client.post(
                "/v1/agent/execute",
                json={"task": "Research and write a blog post about artificial intelligence"}
            )
            
            # Verify full pipeline
            assert response.status_code == 200
            data = response.json()
            
            # API response structure
            assert data["success"] is True
            assert data["agent_type"] == "content"
            
            # Agent response content
            result = data["result"]
            assert result["agent"] == "content"
            assert result["search_performed"] is True
            assert result["search_results_count"] > 0
            assert "artificial intelligence" in result["content"].lower()
            assert result["success"] is True
            
            # Verify search was called
            mock_search_instance.run.assert_called_once()
    
    @pytest.mark.integration
    def test_unknown_task_full_pipeline(self, api_client, mock_mongodb_connected):
        """Test complete pipeline for unknown task type."""
        # Make API request with unrecognizable task
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": "teleport me to Mars"}
        )
        
        # Verify error handling
        assert response.status_code == 422
        data = response.json()
        
        assert data["success"] is False
        assert data["agent_type"] == "unknown"
        assert "error" in data["result"]
        assert "Unknown task type" in data["result"]["error"]
        
        # Should still log interaction
        mock_mongodb_connected.save_interaction.assert_called_once()
    
    @pytest.mark.integration
    def test_session_continuity(self, api_client, mock_mongodb_connected):
        """Test session continuity across multiple requests."""
        session_id = "12345678-1234-5678-9012-123456789abc"
        
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.predict.return_value = "print('Hello, World!')"
            
            # First request
            response1 = api_client.post(
                "/v1/agent/execute",
                json={
                    "task": "Write Python hello world",
                    "session_id": session_id
                }
            )
            
            # Second request with same session
            response2 = api_client.post(
                "/v1/agent/execute",
                json={
                    "task": "Write a simple loop",
                    "session_id": session_id
                }
            )
            
            # Verify session continuity
            assert response1.status_code == 200
            assert response2.status_code == 200
            
            data1 = response1.json()
            data2 = response2.json()
            
            assert data1["session_id"] == session_id
            assert data2["session_id"] == session_id
            
            # Both should be logged with same session
            assert mock_mongodb_connected.save_interaction.call_count == 2


class TestErrorIntegration:
    """Test error handling integration across the system."""
    
    @pytest.mark.integration
    def test_llm_api_error_integration(self, api_client, mock_mongodb_connected):
        """Test integration when LLM API fails."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            # Setup LLM to fail
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.predict.side_effect = Exception("OpenAI API Error")
            
            # Make request
            response = api_client.post(
                "/v1/agent/execute",
                json={"task": "Write Python code"}
            )
            
            # Should handle error gracefully
            assert response.status_code == 500
            data = response.json()
            
            assert data["success"] is False
            assert data["agent_type"] == "error"
            assert "error" in data["result"]
            
            # Should still log the interaction
            mock_mongodb_connected.save_interaction.assert_called_once()
    
    @pytest.mark.integration
    def test_database_error_integration(self, api_client, mock_mongodb_disconnected):
        """Test integration when database is unavailable."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.predict.return_value = "print('test')"
            
            # Make request
            response = api_client.post(
                "/v1/agent/execute",
                json={"task": "Write Python code"}
            )
            
            # Should succeed despite database issues
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            
            # Should attempt to save (but fail silently)
            mock_mongodb_disconnected.save_interaction.assert_called_once()


class TestPerformanceIntegration:
    """Test performance aspects of the integrated system."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_concurrent_requests(self, api_client, mock_mongodb_connected):
        """Test handling multiple concurrent requests."""
        import concurrent.futures
        import threading
        
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.predict.return_value = "def test(): return True"
            
            def make_request(task_num):
                """Make a single API request."""
                response = api_client.post(
                    "/v1/agent/execute",
                    json={"task": f"Write Python function {task_num}"}
                )
                return response.status_code, response.json()
            
            # Make 5 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request, i) for i in range(5)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # All should succeed
            for status_code, data in results:
                assert status_code == 200
                assert data["success"] is True
                assert data["agent_type"] == "code"
            
            # All should be logged
            assert mock_mongodb_connected.save_interaction.call_count == 5
    
    @pytest.mark.integration
    def test_large_task_handling(self, api_client, mock_mongodb_connected):
        """Test handling of large task descriptions."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.predict.return_value = "def process_large_task(): return 'success'"
            
            # Create a large task description
            large_task = "Write a Python function " + "with many requirements " * 100
            
            response = api_client.post(
                "/v1/agent/execute",
                json={"task": large_task}
            )
            
            # Should handle large tasks
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


class TestDataFlowIntegration:
    """Test data flow through the complete system."""
    
    @pytest.mark.integration
    def test_request_tracing(self, api_client, mock_mongodb_connected):
        """Test that requests can be traced through the system."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.predict.return_value = "def traced_function(): pass"
            
            # Make request
            response = api_client.post(
                "/v1/agent/execute",
                json={"task": "Write a traced function"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify tracing data
            assert "request_id" in data
            assert "timestamp" in data
            assert "session_id" in data
            assert len(data["request_id"]) > 0
            
            # Check that interaction was logged with tracing info
            save_call = mock_mongodb_connected.save_interaction.call_args
            logged_data = save_call[0][0]
            
            assert logged_data["request_id"] == data["request_id"]
            assert logged_data["session_id"] == data["session_id"]
    
    @pytest.mark.integration
    def test_response_consistency(self, api_client, mock_mongodb_connected):
        """Test response format consistency across different scenarios."""
        test_cases = [
            ("Write Python code", "code"),
            ("Create a blog post", "content"),
        ]
        
        responses = []
        
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            
            for task, expected_agent in test_cases:
                # Setup appropriate mock response
                if expected_agent == "code":
                    mock_llm_instance.predict.return_value = "def test(): return True"
                else:
                    mock_llm_instance.predict.return_value = "# Blog Post\n\nContent here..."
                
                response = api_client.post(
                    "/v1/agent/execute",
                    json={"task": task}
                )
                
                assert response.status_code == 200
                responses.append(response.json())
        
        # Verify all responses have consistent structure
        required_fields = ["success", "agent_type", "result", "session_id", "timestamp", "request_id"]
        
        for response_data in responses:
            for field in required_fields:
                assert field in response_data, f"Missing field {field} in response"
            
            assert isinstance(response_data["success"], bool)
            assert isinstance(response_data["timestamp"], str)
            assert isinstance(response_data["result"], dict)


class TestEndToEndWorkflows:
    """Test realistic end-to-end workflows."""
    
    @pytest.mark.integration
    def test_developer_workflow(self, api_client, mock_mongodb_connected):
        """Test a typical developer workflow with multiple code requests."""
        session_id = "dev-session-123"
        
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            
            # Step 1: Generate a function
            mock_llm_instance.predict.return_value = "def calculate_sum(a, b): return a + b"
            response1 = api_client.post(
                "/v1/agent/execute",
                json={
                    "task": "Write a function to calculate sum of two numbers",
                    "session_id": session_id
                }
            )
            
            # Step 2: Debug the function
            mock_llm_instance.predict.return_value = "def calculate_sum(a, b):\n    # Fixed: Added type checking\n    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):\n        raise TypeError('Arguments must be numbers')\n    return a + b"
            response2 = api_client.post(
                "/v1/agent/execute",
                json={
                    "task": "Debug and improve the sum function with error handling",
                    "session_id": session_id
                }
            )
            
            # Step 3: Write tests
            mock_llm_instance.predict.return_value = "def test_calculate_sum():\n    assert calculate_sum(2, 3) == 5\n    assert calculate_sum(-1, 1) == 0\n    pytest.raises(TypeError, calculate_sum, 'a', 2)"
            response3 = api_client.post(
                "/v1/agent/execute",
                json={
                    "task": "Write unit tests for the sum function",
                    "session_id": session_id
                }
            )
            
            # Verify workflow
            for response in [response1, response2, response3]:
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["agent_type"] == "code"
                assert data["session_id"] == session_id
            
            # All should be logged under same session
            assert mock_mongodb_connected.save_interaction.call_count == 3
    
    @pytest.mark.integration
    def test_content_creator_workflow(self, api_client, mock_mongodb_connected):
        """Test a typical content creator workflow."""
        session_id = "content-session-456"
        
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_search:
            
            mock_llm_instance = Mock()
            mock_search_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_search.return_value = mock_search_instance
            
            # Step 1: Research topic
            mock_search_instance.run.return_value = "Climate change research shows significant global warming trends..."
            mock_llm_instance.predict.return_value = "# Climate Change Research Summary\n\nRecent studies indicate accelerating trends..."
            
            response1 = api_client.post(
                "/v1/agent/execute",
                json={
                    "task": "Research latest information about climate change",
                    "session_id": session_id
                }
            )
            
            # Step 2: Create blog post
            mock_llm_instance.predict.return_value = "# Understanding Climate Change\n\nClimate change represents one of the most pressing challenges..."
            
            response2 = api_client.post(
                "/v1/agent/execute",
                json={
                    "task": "Write a comprehensive blog post about climate change",
                    "session_id": session_id
                }
            )
            
            # Verify workflow
            for response in [response1, response2]:
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["agent_type"] == "content"
                assert data["session_id"] == session_id
            
            # Research should have triggered search
            data1 = response1.json()
            assert data1["result"]["search_performed"] is True
            
            # Both logged under same session
            assert mock_mongodb_connected.save_interaction.call_count == 2 