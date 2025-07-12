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
    assert_response_structure,
    validate_code_response,
    validate_content_response,
    validate_search_integration,
    semantic_similarity
)


class TestFullPipelineIntegration:
    """Test the complete request-response pipeline."""
    
    @pytest.mark.integration
    def test_code_task_full_pipeline(self, api_client, mock_mongodb_connected):
        """Test complete pipeline for code generation task."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            # Setup LLM mock with realistic response
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.predict.return_value = """def fibonacci(n):
    '''Calculate nth fibonacci number using recursion'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the function
for i in range(10):
    print(f"fibonacci({i}) = {fibonacci(i)}")"""
            
            # Make API request
            response = api_client.post(
                "/v1/agent/execute",
                json={"task": "Write a Python function to calculate fibonacci numbers"}
            )
            
            # Verify full pipeline with behavioral assertions
            assert response.status_code == 200
            data = response.json()
            
            # API response structure
            assert data["success"] is True
            assert data["agent_type"] == "code"
            assert assert_valid_uuid(data["session_id"])
            assert assert_valid_iso_datetime(data["timestamp"])
            assert "request_id" in data
            
            # Agent response content validation
            result = data["result"]
            assert validate_code_response(result, expected_language="python")
            assert assert_response_structure(result, expected_agent="code")
            
            # Verify code quality without exact matching
            code = result["code"]
            assert isinstance(code, str)
            assert len(code.strip()) > 50  # Substantial code
            
            # Check for fibonacci-related patterns
            fibonacci_patterns = ["def", "fibonacci", "return", "n", "recursion", "fib"]
            code_lower = code.lower()
            assert any(pattern in code_lower for pattern in fibonacci_patterns)
            
            # MongoDB interaction should be logged
            mock_mongodb_connected.save_interaction.assert_called_once()
    
    @pytest.mark.integration
    def test_content_task_full_pipeline(self, api_client, mock_mongodb_connected):
        """Test complete pipeline for content generation task."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_search:
            
            # Setup mocks with realistic responses
            mock_llm_instance = Mock()
            mock_search_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_search.return_value = mock_search_instance
            
            mock_search_instance.run.return_value = "AI Research Updates\nLatest artificial intelligence developments\nMachine learning breakthroughs\nDeep learning advances"
            mock_llm_instance.predict.return_value = """# The Evolution of Artificial Intelligence

Artificial intelligence has undergone remarkable transformations in recent years, reshaping industries and redefining possibilities.

## Current Developments
- Advanced machine learning algorithms
- Improved natural language processing
- Enhanced computer vision capabilities
- Autonomous system innovations

## Future Implications
These technological advances continue to create new opportunities while presenting unique challenges for society to address.

## Sources
Based on recent research and industry developments."""
            
            # Make API request
            response = api_client.post(
                "/v1/agent/execute",
                json={"task": "Research and write a comprehensive article about artificial intelligence"}
            )
            
            # Verify full pipeline with behavioral assertions
            assert response.status_code == 200
            data = response.json()
            
            # API response structure
            assert data["success"] is True
            assert data["agent_type"] == "content"
            assert assert_valid_uuid(data["session_id"])
            assert assert_valid_iso_datetime(data["timestamp"])
            
            # Agent response content validation
            result = data["result"]
            assert validate_content_response(result, expected_topics=["ai", "artificial", "intelligence"])
            assert validate_search_integration(result)
            assert assert_response_structure(result, expected_agent="content")
            
            # Verify content quality
            content = result["content"]
            assert isinstance(content, str)
            assert len(content.strip()) > 200  # Comprehensive content
            
            # Check for AI-related concepts
            ai_concepts = ["artificial", "intelligence", "machine", "learning", "algorithm", "technology"]
            content_lower = content.lower()
            assert any(concept in content_lower for concept in ai_concepts)
            
            # Should have proper article structure
            assert any(marker in content for marker in ["#", "##", "###"])
            
            # Search integration should be working
            assert result["search_performed"] is True
            assert result["search_results_count"] >= 0
            
            # Verify search was called
            mock_search_instance.run.assert_called_once()
    
    @pytest.mark.integration
    def test_unknown_task_full_pipeline(self, api_client, mock_mongodb_connected):
        """Test complete pipeline for unknown task type."""
        # Make API request with unrecognizable task
        response = api_client.post(
            "/v1/agent/execute",
            json={"task": "please help me time travel to the year 3000"}
        )
        
        # Verify error handling
        assert response.status_code == 422
        data = response.json()
        
        assert data["success"] is False
        assert data["agent_type"] == "unknown"
        assert "error" in data["result"]
        assert "unknown task type" in data["result"]["error"].lower()
        
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
            
            # Setup different responses for different requests
            responses = [
                "print('Hello, World!')",
                """for i in range(5):
    print(f"Number: {i}")"""
            ]
            mock_llm_instance.predict.side_effect = responses
            
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
            
            # Both should be code responses
            assert data1["agent_type"] == "code"
            assert data2["agent_type"] == "code"
            
            # Verify code quality
            assert validate_code_response(data1["result"], expected_language="python")
            assert validate_code_response(data2["result"], expected_language="python")
            
            # Both should be logged with same session
            assert mock_mongodb_connected.save_interaction.call_count == 2
    
    @pytest.mark.integration
    def test_task_routing_accuracy(self, api_client, mock_mongodb_connected):
        """Test that tasks are routed to appropriate agents."""
        test_cases = [
            {
                "task": "Write a Python class for data validation",
                "expected_agent": "code",
                "expected_language": "python"
            },
            {
                "task": "Create a JavaScript function for form validation",
                "expected_agent": "code",
                "expected_language": "javascript"
            },
            {
                "task": "Write a blog post about sustainable energy",
                "expected_agent": "content",
                "expected_topics": ["sustainable", "energy", "blog"]
            },
            {
                "task": "Research and write about climate change impacts",
                "expected_agent": "content",
                "expected_topics": ["climate", "change", "research"]
            }
        ]
        
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_search:
            
            mock_llm_instance = Mock()
            mock_search_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_search.return_value = mock_search_instance
            
            # Setup dynamic responses
            def dynamic_llm_response(prompt):
                prompt_lower = prompt.lower()
                if "python" in prompt_lower and "class" in prompt_lower:
                    return """class DataValidator:
    def __init__(self):
        self.rules = []
    
    def validate(self, data):
        return all(rule(data) for rule in self.rules)"""
                elif "javascript" in prompt_lower and "function" in prompt_lower:
                    return """function validateForm(form) {
    const fields = form.querySelectorAll('input');
    return Array.from(fields).every(field => field.value.trim() !== '');
}"""
                elif "blog" in prompt_lower and "energy" in prompt_lower:
                    return """# The Future of Sustainable Energy

Sustainable energy solutions are becoming increasingly important as we face environmental challenges.

## Key Technologies
- Solar power innovations
- Wind energy advancements  
- Energy storage solutions

These technologies offer promising paths toward a cleaner future."""
                elif "climate" in prompt_lower and "change" in prompt_lower:
                    return """# Climate Change: Understanding the Impacts

Climate change represents one of the most significant challenges of our time.

## Major Impacts
- Rising global temperatures
- Extreme weather events
- Sea level rise
- Ecosystem disruption

Research shows the urgent need for comprehensive action."""
                else:
                    return "Generic response for the given task."
            
            mock_llm_instance.predict.side_effect = dynamic_llm_response
            mock_search_instance.run.return_value = "Research results for the topic"
            
            for test_case in test_cases:
                response = api_client.post(
                    "/v1/agent/execute",
                    json={"task": test_case["task"]}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify correct agent routing
                assert data["success"] is True
                assert data["agent_type"] == test_case["expected_agent"]
                
                # Verify result structure
                result = data["result"]
                if test_case["expected_agent"] == "code":
                    assert validate_code_response(result, test_case.get("expected_language"))
                elif test_case["expected_agent"] == "content":
                    assert validate_content_response(result, test_case.get("expected_topics"))
                    assert validate_search_integration(result)


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
            assert "error" in data
            assert isinstance(data["error"], str)
            
            # Should still log interaction
            mock_mongodb_connected.save_interaction.assert_called_once()
    
    @pytest.mark.integration
    def test_search_api_error_integration(self, api_client, mock_mongodb_connected):
        """Test integration when search API fails."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_search:
            
            # Setup search to fail, LLM to succeed
            mock_llm_instance = Mock()
            mock_search_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_search.return_value = mock_search_instance
            
            mock_search_instance.run.side_effect = Exception("Search API Error")
            mock_llm_instance.predict.return_value = """# Technology Trends

Even without search results, I can provide insights about technology trends.

## Key Areas
- Artificial intelligence development
- Cloud computing growth
- Cybersecurity improvements

These trends shape the future of technology."""
            
            # Make request that should trigger search
            response = api_client.post(
                "/v1/agent/execute",
                json={"task": "Research and write about technology trends"}
            )
            
            # Should succeed despite search failure
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["agent_type"] == "content"
            
            result = data["result"]
            assert validate_content_response(result, expected_topics=["technology", "trends"])
            assert validate_search_integration(result)
            
            # Search should have been attempted but failed
            assert result["search_performed"] is True
            assert result["search_results_count"] == 0
    
    @pytest.mark.integration
    def test_database_error_integration(self, api_client, mock_mongodb_disconnected):
        """Test integration when database is unavailable."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            # Setup successful LLM response
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.predict.return_value = "print('Hello, Database Error!')"
            
            # Make request
            response = api_client.post(
                "/v1/agent/execute",
                json={"task": "Write Python hello world"}
            )
            
            # Should succeed despite database being unavailable
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["agent_type"] == "code"
            
            # Result should be valid
            result = data["result"]
            assert validate_code_response(result, expected_language="python")
            
            # Database interaction should have been attempted
            mock_mongodb_disconnected.save_interaction.assert_called_once()
    
    @pytest.mark.integration
    def test_cascading_error_recovery(self, api_client, mock_mongodb_connected):
        """Test system recovery from cascading errors."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_search:
            
            # Setup multiple failure points
            mock_llm_instance = Mock()
            mock_search_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_search.return_value = mock_search_instance
            
            # First call fails, second succeeds
            mock_llm_instance.predict.side_effect = [
                Exception("First API call failed"),
                Exception("Second API call failed"),
                "print('Finally working!')"  # Third call succeeds
            ]
            
            # Make multiple requests to test recovery
            responses = []
            for i in range(3):
                response = api_client.post(
                    "/v1/agent/execute",
                    json={"task": f"Write Python code attempt {i+1}"}
                )
                responses.append(response)
            
            # First two should fail
            assert responses[0].status_code == 500
            assert responses[1].status_code == 500
            
            # Third should succeed
            assert responses[2].status_code == 200
            data = responses[2].json()
            assert data["success"] is True
            assert validate_code_response(data["result"], expected_language="python")
            
            # All should be logged
            assert mock_mongodb_connected.save_interaction.call_count == 3


class TestPerformanceIntegration:
    """Test performance characteristics of the integration."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_concurrent_requests_handling(self, api_client, mock_mongodb_connected):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            # Setup mock with slight delay to simulate processing
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            
            def delayed_response(prompt):
                time.sleep(0.1)  # Small delay
                return f"# Response for: {prompt[:50]}...\n\nProcessed successfully."
            
            mock_llm_instance.predict.side_effect = delayed_response
            
            # Function to make request
            results = []
            def make_request(task_num):
                response = api_client.post(
                    "/v1/agent/execute",
                    json={"task": f"Write about topic {task_num}"}
                )
                results.append(response)
            
            # Create multiple threads
            threads = []
            for i in range(3):  # Small number for test reliability
                thread = threading.Thread(target=make_request, args=(i,))
                threads.append(thread)
            
            # Start all threads
            for thread in threads:
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify all requests succeeded
            assert len(results) == 3
            for response in results:
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["agent_type"] == "content"
                assert validate_content_response(data["result"])
    
    @pytest.mark.integration
    def test_large_task_handling(self, api_client, mock_mongodb_connected):
        """Test handling of large tasks."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.predict.return_value = """# Large Task Response

This is a comprehensive response to a large task request.

## Content Structure
- Detailed analysis
- Multiple sections
- Extensive coverage

The system can handle substantial task requests and generate appropriate responses."""
            
            # Create a large task
            large_task = """Please write a comprehensive analysis of modern software development practices, including:
            1. Agile methodologies and their variations
            2. DevOps practices and CI/CD pipelines
            3. Cloud computing architectures
            4. Microservices design patterns
            5. Security best practices
            6. Testing strategies and automation
            7. Performance optimization techniques
            8. Code quality and maintainability
            9. Team collaboration tools
            10. Future trends in software development"""
            
            response = api_client.post(
                "/v1/agent/execute",
                json={"task": large_task}
            )
            
            # Should handle large task successfully
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["agent_type"] == "content"
            
            result = data["result"]
            assert validate_content_response(result, expected_topics=["software", "development"])
            
            # Response should be substantial
            content = result["content"]
            assert len(content) > 100  # Should be comprehensive
    
    @pytest.mark.integration
    def test_response_time_consistency(self, api_client, mock_mongodb_connected):
        """Test that response times are consistent and reasonable."""
        import time
        
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.predict.return_value = "print('Timing test')"
            
            # Make multiple requests and measure timing
            response_times = []
            for i in range(5):
                start_time = time.time()
                response = api_client.post(
                    "/v1/agent/execute",
                    json={"task": f"Write Python code test {i}"}
                )
                end_time = time.time()
                
                assert response.status_code == 200
                response_times.append(end_time - start_time)
            
            # Response times should be reasonable
            max_time = max(response_times)
            min_time = min(response_times)
            avg_time = sum(response_times) / len(response_times)
            
            # Basic performance assertions
            assert max_time < 5.0  # Should complete within 5 seconds
            assert min_time > 0.0  # Should take some time
            assert avg_time < 2.0  # Average should be reasonable


class TestDataFlowIntegration:
    """Test data flow through the complete system."""
    
    @pytest.mark.integration
    def test_request_response_data_integrity(self, api_client, mock_mongodb_connected):
        """Test that data maintains integrity through the pipeline."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.predict.return_value = """def calculate_average(numbers):
    '''Calculate the average of a list of numbers'''
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

# Example usage
data = [1, 2, 3, 4, 5]
result = calculate_average(data)
print(f"Average: {result}")"""
            
            original_task = "Write a Python function to calculate the average of a list"
            session_id = "test-session-12345678-1234-5678-9012-123456789abc"
            
            response = api_client.post(
                "/v1/agent/execute",
                json={
                    "task": original_task,
                    "session_id": session_id
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify data integrity
            assert data["success"] is True
            assert data["session_id"] == session_id
            
            result = data["result"]
            assert result["task"] == original_task
            assert result["agent"] == "code"
            assert result["language"] == "python"
            
            # Verify the function code contains expected elements
            code = result["code"]
            assert "def calculate_average" in code
            assert "numbers" in code
            assert "return" in code
    
    @pytest.mark.integration
    def test_search_data_integration(self, api_client, mock_mongodb_connected):
        """Test that search data is properly integrated into responses."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_search:
            
            mock_llm_instance = Mock()
            mock_search_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_search.return_value = mock_search_instance
            
            # Setup search results
            search_results = "Quantum Computing Advances\nLatest quantum computing breakthroughs\nQuantum algorithms research\nQuantum hardware developments"
            mock_search_instance.run.return_value = search_results
            
            # Setup LLM to incorporate search results
            mock_llm_instance.predict.return_value = """# Quantum Computing: Current State and Future Prospects

Based on recent research, quantum computing continues to advance rapidly.

## Key Developments
- Quantum algorithm improvements
- Hardware stability enhancements
- Error correction advances
- Practical applications emerging

## Research Sources
This analysis incorporates the latest findings from quantum computing research communities."""
            
            response = api_client.post(
                "/v1/agent/execute",
                json={"task": "Research and write about quantum computing advances"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify search integration
            result = data["result"]
            assert validate_search_integration(result)
            assert result["search_performed"] is True
            assert result["search_results_count"] > 0
            
            # Verify content incorporates search context
            content = result["content"]
            quantum_concepts = ["quantum", "computing", "algorithm", "hardware", "research"]
            content_lower = content.lower()
            assert any(concept in content_lower for concept in quantum_concepts)
    
    @pytest.mark.integration
    def test_error_data_propagation(self, api_client, mock_mongodb_connected):
        """Test that error information is properly propagated."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm:
            
            # Setup LLM to fail with specific error
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.predict.side_effect = Exception("Specific API Error Message")
            
            response = api_client.post(
                "/v1/agent/execute",
                json={"task": "Write Python code"}
            )
            
            assert response.status_code == 500
            data = response.json()
            
            # Verify error propagation
            assert data["success"] is False
            assert "error" in data
            assert isinstance(data["error"], str)
            assert len(data["error"]) > 0
            
            # Should maintain request context
            assert "timestamp" in data
            assert assert_valid_iso_datetime(data["timestamp"])


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.integration
    def test_developer_workflow(self, api_client, mock_mongodb_connected):
        """Test a complete developer workflow."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            
            # Simulate developer workflow: function -> test -> debug
            workflow_responses = [
                # 1. Write function
                """def validate_email(email):
    '''Validate email address format'''
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))""",
                
                # 2. Write tests
                """import unittest

class TestEmailValidation(unittest.TestCase):
    def test_valid_emails(self):
        valid_emails = ['test@example.com', 'user@domain.org']
        for email in valid_emails:
            self.assertTrue(validate_email(email))
    
    def test_invalid_emails(self):
        invalid_emails = ['invalid', 'test@', '@domain.com']
        for email in invalid_emails:
            self.assertFalse(validate_email(email))""",
                
                # 3. Debug issue
                """# Fixed email validation with better regex
def validate_email(email):
    '''Validate email address format with improved pattern'''
    import re
    # Updated pattern to handle edge cases
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))"""
            ]
            
            mock_llm_instance.predict.side_effect = workflow_responses
            
            workflow_tasks = [
                "Write a Python function to validate email addresses",
                "Write unit tests for the email validation function",
                "Debug and fix the email validation function"
            ]
            
            session_id = "dev-workflow-12345678-1234-5678-9012-123456789abc"
            
            for i, task in enumerate(workflow_tasks):
                response = api_client.post(
                    "/v1/agent/execute",
                    json={
                        "task": task,
                        "session_id": session_id
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify workflow continuity
                assert data["success"] is True
                assert data["agent_type"] == "code"
                assert data["session_id"] == session_id
                
                result = data["result"]
                assert validate_code_response(result, expected_language="python")
                
                # Verify appropriate code patterns
                code = result["code"]
                if i == 0:  # Function
                    assert "def validate_email" in code
                elif i == 1:  # Tests
                    assert "unittest" in code or "test" in code.lower()
                elif i == 2:  # Debug
                    assert "fixed" in code.lower() or "debug" in code.lower()
    
    @pytest.mark.integration
    def test_content_creator_workflow(self, api_client, mock_mongodb_connected):
        """Test a complete content creator workflow."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_search:
            
            mock_llm_instance = Mock()
            mock_search_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_search.return_value = mock_search_instance
            
            # Setup search results
            mock_search_instance.run.return_value = "Sustainable Technology News\nGreen energy innovations\nEco-friendly tech solutions\nEnvironmental impact studies"
            
            # Simulate content workflow: research -> outline -> write
            workflow_responses = [
                # 1. Research
                """# Research Summary: Sustainable Technology

## Key Findings
- Renewable energy adoption increasing
- Battery technology improvements
- Electric vehicle growth
- Smart grid implementations

## Sources
- Recent industry reports
- Academic research papers
- Government sustainability initiatives""",
                
                # 2. Create outline
                """# Article Outline: The Future of Sustainable Technology

## Introduction
- Current environmental challenges
- Role of technology in solutions

## Main Sections
1. Renewable Energy Revolution
2. Transportation Transformation
3. Smart City Innovations
4. Consumer Technology Impact

## Conclusion
- Future outlook
- Call to action""",
                
                # 3. Write full article
                """# The Future of Sustainable Technology: Building a Greener Tomorrow

As environmental challenges intensify, sustainable technology emerges as our most powerful ally in creating a cleaner, more efficient world.

## The Renewable Energy Revolution

Solar and wind power have transformed from alternative energy sources to mainstream solutions. Recent advances in battery storage and smart grid technology are making renewable energy more reliable and cost-effective than ever before.

## Transportation Transformation

The automotive industry is undergoing a fundamental shift toward electric vehicles. With improving battery technology and expanding charging infrastructure, electric transportation is becoming accessible to consumers worldwide.

## Smart City Innovations

Urban areas are implementing smart technologies to reduce energy consumption and improve quality of life. From intelligent traffic systems to efficient waste management, cities are becoming more sustainable through technology.

## Consumer Technology Impact

Personal devices are becoming more energy-efficient while maintaining performance. Sustainable manufacturing practices and circular economy principles are reshaping how we design and use technology.

## Conclusion

The future of sustainable technology holds immense promise. By continuing to innovate and invest in green solutions, we can build a more sustainable world for future generations."""
            ]
            
            mock_llm_instance.predict.side_effect = workflow_responses
            
            workflow_tasks = [
                "Research recent developments in sustainable technology",
                "Create an outline for an article about sustainable technology",
                "Write a comprehensive article about the future of sustainable technology"
            ]
            
            session_id = "content-workflow-12345678-1234-5678-9012-123456789abc"
            
            for i, task in enumerate(workflow_tasks):
                response = api_client.post(
                    "/v1/agent/execute",
                    json={
                        "task": task,
                        "session_id": session_id
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify workflow continuity
                assert data["success"] is True
                assert data["agent_type"] == "content"
                assert data["session_id"] == session_id
                
                result = data["result"]
                assert validate_content_response(result, expected_topics=["sustainable", "technology"])
                assert validate_search_integration(result)
                
                # Verify progression complexity
                content = result["content"]
                if i == 0:  # Research
                    assert "research" in content.lower()
                    assert "findings" in content.lower()
                elif i == 1:  # Outline
                    assert "outline" in content.lower()
                    assert any(marker in content for marker in ["##", "1.", "2."])
                elif i == 2:  # Full article
                    assert len(content) > 500  # Should be comprehensive
                    assert "sustainable" in content.lower()
                    assert "technology" in content.lower()
    
    @pytest.mark.integration
    def test_mixed_workflow_interaction(self, api_client, mock_mongodb_connected):
        """Test workflow with mixed content and code tasks."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_search:
            
            mock_llm_instance = Mock()
            mock_search_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_search.return_value = mock_search_instance
            
            mock_search_instance.run.return_value = "API Development Best Practices\nRESTful API design\nAPI documentation tools\nTesting strategies"
            
            # Mixed workflow: research -> design -> implement
            workflow_responses = [
                # 1. Research content
                """# API Development Best Practices

## Key Principles
- RESTful design patterns
- Proper HTTP status codes
- Comprehensive documentation
- Thorough testing strategies

## Implementation Guidelines
- Version control
- Error handling
- Security considerations
- Performance optimization""",
                
                # 2. Code implementation
                """from flask import Flask, jsonify, request
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

class UserAPI(Resource):
    def get(self, user_id):
        # Retrieve user data
        return jsonify({'user_id': user_id, 'status': 'active'})
    
    def post(self):
        # Create new user
        data = request.get_json()
        return jsonify({'message': 'User created', 'data': data}), 201

api.add_resource(UserAPI, '/api/users/<int:user_id>', '/api/users')

if __name__ == '__main__':
    app.run(debug=True)"""
            ]
            
            mock_llm_instance.predict.side_effect = workflow_responses
            
            mixed_tasks = [
                "Research best practices for API development",
                "Write Python code for a RESTful API using Flask"
            ]
            
            session_id = "mixed-workflow-12345678-1234-5678-9012-123456789abc"
            
            for i, task in enumerate(mixed_tasks):
                response = api_client.post(
                    "/v1/agent/execute",
                    json={
                        "task": task,
                        "session_id": session_id
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify appropriate agent routing
                assert data["success"] is True
                assert data["session_id"] == session_id
                
                result = data["result"]
                
                if i == 0:  # Research task
                    assert data["agent_type"] == "content"
                    assert validate_content_response(result, expected_topics=["api", "development"])
                    assert validate_search_integration(result)
                elif i == 1:  # Code task
                    assert data["agent_type"] == "code"
                    assert validate_code_response(result, expected_language="python")
                    
                    # Verify API-related code
                    code = result["code"]
                    api_patterns = ["flask", "api", "resource", "def", "return"]
                    code_lower = code.lower()
                    assert any(pattern in code_lower for pattern in api_patterns) 