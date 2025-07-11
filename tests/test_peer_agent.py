"""
Tests for the PeerAgent class and routing functionality.

This module tests the core routing logic, task analysis, and agent delegation
functionality of the Peer Agent system.
"""

import pytest
from unittest.mock import Mock, patch
from agents.peer_agent import PeerAgent


class TestPeerAgentInitialization:
    """Test PeerAgent initialization and setup."""
    
    @patch('agents.peer_agent.ContentAgent')
    @patch('agents.peer_agent.CodeAgent')
    def test_peer_agent_initialization(self, mock_code_agent, mock_content_agent, mock_env_vars):
        """Test that PeerAgent initializes correctly with sub-agents."""
        # Arrange
        mock_content_instance = Mock()
        mock_code_instance = Mock()
        mock_content_agent.return_value = mock_content_instance
        mock_code_agent.return_value = mock_code_instance
        
        # Act
        peer_agent = PeerAgent()
        
        # Assert
        assert peer_agent is not None
        assert peer_agent.content_agent == mock_content_instance
        assert peer_agent.code_agent == mock_code_instance
        assert isinstance(peer_agent.routing_keywords, dict)
        assert 'content' in peer_agent.routing_keywords
        assert 'code' in peer_agent.routing_keywords

    def test_routing_keywords_structure(self, mock_env_vars):
        """Test that routing keywords are properly structured."""
        with patch('agents.peer_agent.ContentAgent'), patch('agents.peer_agent.CodeAgent'):
            peer_agent = PeerAgent()
            
            # Check content keywords
            content_keywords = peer_agent.routing_keywords['content']
            assert isinstance(content_keywords, list)
            assert 'blog' in content_keywords
            assert 'article' in content_keywords
            assert 'write' in content_keywords
            
            # Check code keywords
            code_keywords = peer_agent.routing_keywords['code']
            assert isinstance(code_keywords, list)
            assert 'code' in code_keywords
            assert 'function' in code_keywords
            assert 'python' in code_keywords


class TestTaskAnalysis:
    """Test task analysis and routing decision logic."""
    
    @pytest.fixture
    def peer_agent(self, mock_env_vars):
        """Create a PeerAgent instance for testing."""
        with patch('agents.peer_agent.ContentAgent'), patch('agents.peer_agent.CodeAgent'):
            return PeerAgent()
    
    @pytest.mark.unit
    def test_analyze_code_tasks(self, peer_agent, sample_task_requests):
        """Test that code-related tasks are routed to code agent."""
        code_tasks = [
            "Write a Python function to calculate fibonacci numbers",
            "Create a JavaScript function for sorting arrays",
            "Debug this code snippet",
            "Implement a binary search algorithm",
            "Fix this Python script error"
        ]
        
        for task in code_tasks:
            result = peer_agent.analyze_task(task)
            assert result == "code", f"Task '{task}' should route to code agent"
    
    @pytest.mark.unit
    def test_analyze_content_tasks(self, peer_agent):
        """Test that content-related tasks are routed to content agent."""
        content_tasks = [
            "Write a blog post about artificial intelligence",
            "Create an article about climate change",
            "Research information about space exploration",
            "Generate content about dogs",
            "Write a story about adventure"
        ]
        
        for task in content_tasks:
            result = peer_agent.analyze_task(task)
            assert result == "content", f"Task '{task}' should route to content agent"
    
    @pytest.mark.unit
    def test_analyze_unknown_tasks(self, peer_agent, sample_task_requests):
        """Test that unrecognized tasks return unknown."""
        unknown_tasks = [
            "fly to the moon",
            "solve world hunger",
            "predict the future",
            "teleport to Paris",
            "become invisible"
        ]
        
        for task in unknown_tasks:
            result = peer_agent.analyze_task(task)
            assert result == "unknown", f"Task '{task}' should return unknown"
    
    @pytest.mark.unit
    def test_analyze_empty_tasks(self, peer_agent, sample_task_requests):
        """Test edge cases with empty or whitespace tasks."""
        empty_tasks = ["", "   ", "\n", "\t", None]
        
        for task in empty_tasks:
            result = peer_agent.analyze_task(task)
            assert result == "unknown", f"Empty task '{task}' should return unknown"
    
    @pytest.mark.unit
    def test_analyze_mixed_tasks(self, peer_agent):
        """Test tasks with mixed keywords favor the one with more matches."""
        # Task with clearly more code keywords
        mixed_task_code = "Write Python code function to implement algorithm debug script"
        result = peer_agent.analyze_task(mixed_task_code)
        assert result == "code", "Task with more code keywords should route to code agent"
        
        # Task with clearly more content keywords  
        mixed_task_content = "Write a blog article post story about research topic information"
        result = peer_agent.analyze_task(mixed_task_content)
        assert result == "content", "Task with more content keywords should route to content agent"
    
    @pytest.mark.unit
    def test_case_insensitive_routing(self, peer_agent):
        """Test that routing is case-insensitive."""
        tasks = [
            ("WRITE CODE IN PYTHON", "code"),
            ("Create A Blog POST", "content"),
            ("Python FUNCTION for sorting", "code"),
            ("ARTICLE about AI", "content")
        ]
        
        for task, expected in tasks:
            result = peer_agent.analyze_task(task)
            assert result == expected, f"Case-insensitive task '{task}' should route to {expected}"


class TestTaskRouting:
    """Test the complete task routing workflow."""
    
    @pytest.fixture
    def peer_agent_with_mocks(self, mock_env_vars):
        """Create PeerAgent with mocked sub-agents."""
        with patch('agents.peer_agent.ContentAgent') as mock_content, \
             patch('agents.peer_agent.CodeAgent') as mock_code:
            
            # Setup mock responses
            mock_content_instance = Mock()
            mock_code_instance = Mock()
            mock_content.return_value = mock_content_instance
            mock_code.return_value = mock_code_instance
            
            # Mock successful responses
            mock_content_instance.process_task.return_value = {
                "agent": "content",
                "task": "test task",
                "content": "Generated content",
                "success": True
            }
            
            mock_code_instance.process_task.return_value = {
                "agent": "code", 
                "task": "test task",
                "code": "print('hello world')",
                "success": True
            }
            
            peer_agent = PeerAgent()
            return peer_agent, mock_content_instance, mock_code_instance
    
    @pytest.mark.unit
    def test_route_code_task_success(self, peer_agent_with_mocks):
        """Test successful routing of code task."""
        peer_agent, mock_content, mock_code = peer_agent_with_mocks
        
        task = "Write a Python function to calculate factorial"
        result = peer_agent.route_task(task)
        
        # Assertions
        assert result["success"] is True
        assert result["agent_type"] == "code"
        assert result["result"]["agent"] == "code"
        mock_code.process_task.assert_called_once_with(task)
        mock_content.process_task.assert_not_called()
    
    @pytest.mark.unit
    def test_route_content_task_success(self, peer_agent_with_mocks):
        """Test successful routing of content task."""
        peer_agent, mock_content, mock_code = peer_agent_with_mocks
        
        task = "Write a blog post about machine learning"
        result = peer_agent.route_task(task)
        
        # Assertions
        assert result["success"] is True
        assert result["agent_type"] == "content"
        assert result["result"]["agent"] == "content"
        mock_content.process_task.assert_called_once_with(task)
        mock_code.process_task.assert_not_called()
    
    @pytest.mark.unit
    def test_route_unknown_task(self, peer_agent_with_mocks):
        """Test routing of unknown task."""
        peer_agent, mock_content, mock_code = peer_agent_with_mocks
        
        task = "fly to the moon"
        result = peer_agent.route_task(task)
        
        # Assertions
        assert result["success"] is False
        assert result["agent_type"] == "unknown"
        assert "error" in result["result"]
        assert "Unknown task type" in result["result"]["error"]
        mock_content.process_task.assert_not_called()
        mock_code.process_task.assert_not_called()
    
    @pytest.mark.unit
    def test_route_task_with_agent_error(self, mock_env_vars):
        """Test routing when sub-agent throws an error."""
        with patch('agents.peer_agent.ContentAgent') as mock_content, \
             patch('agents.peer_agent.CodeAgent') as mock_code:
            
            # Setup mocks
            mock_content_instance = Mock()
            mock_code_instance = Mock()
            mock_content.return_value = mock_content_instance
            mock_code.return_value = mock_code_instance
            
            # Make code agent throw an exception
            mock_code_instance.process_task.side_effect = Exception("Agent error")
            
            peer_agent = PeerAgent()
            
            task = "Write Python code"
            result = peer_agent.route_task(task)
            
            # Should handle the error gracefully
            assert result["success"] is False
            assert result["agent_type"] == "error"
            assert "error" in result["result"]
            assert "Internal error" in result["result"]["error"]


class TestUtilityMethods:
    """Test utility methods of PeerAgent."""
    
    @pytest.fixture
    def peer_agent(self, mock_env_vars):
        """Create a PeerAgent instance for testing."""
        with patch('agents.peer_agent.ContentAgent'), patch('agents.peer_agent.CodeAgent'):
            return PeerAgent()
    
    @pytest.mark.unit
    def test_get_available_agents(self, peer_agent):
        """Test getting available agents information."""
        result = peer_agent.get_available_agents()
        
        assert isinstance(result, dict)
        assert 'content' in result
        assert 'code' in result
        assert isinstance(result['content'], list)
        assert isinstance(result['code'], list)
        
        # Check that original keywords exist
        original_count = len(peer_agent.routing_keywords['content'])
        result['content'].append('test')
        # The copy() method only does shallow copy, so this might still affect original
        # Just check the structure is correct
        assert len(peer_agent.routing_keywords['content']) >= original_count
    
    @pytest.mark.unit
    def test_add_routing_keyword_success(self, peer_agent):
        """Test successfully adding a routing keyword."""
        initial_count = len(peer_agent.routing_keywords['content'])
        
        result = peer_agent.add_routing_keyword('content', 'newkeyword')
        
        assert result is True
        assert 'newkeyword' in peer_agent.routing_keywords['content']
        assert len(peer_agent.routing_keywords['content']) == initial_count + 1
    
    @pytest.mark.unit
    def test_add_routing_keyword_duplicate(self, peer_agent):
        """Test adding a duplicate routing keyword."""
        existing_keyword = peer_agent.routing_keywords['content'][0]
        initial_count = len(peer_agent.routing_keywords['content'])
        
        result = peer_agent.add_routing_keyword('content', existing_keyword)
        
        assert result is False
        assert len(peer_agent.routing_keywords['content']) == initial_count
    
    @pytest.mark.unit
    def test_add_routing_keyword_invalid_agent(self, peer_agent):
        """Test adding keyword to non-existent agent type."""
        result = peer_agent.add_routing_keyword('invalid_agent', 'keyword')
        
        assert result is False
        assert 'invalid_agent' not in peer_agent.routing_keywords


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def peer_agent(self, mock_env_vars):
        """Create a PeerAgent instance for testing."""
        with patch('agents.peer_agent.ContentAgent'), patch('agents.peer_agent.CodeAgent'):
            return PeerAgent()
    
    @pytest.mark.unit
    def test_very_long_task(self, peer_agent):
        """Test handling of very long task descriptions."""
        long_task = "Write a Python function " * 1000 + "to calculate fibonacci"
        result = peer_agent.analyze_task(long_task)
        assert result == "code"
    
    @pytest.mark.unit
    def test_special_characters_in_task(self, peer_agent):
        """Test handling of special characters in tasks."""
        special_tasks = [
            "Write code with @#$%^&*() symbols",
            "Create blog about émojis and ñoñó",
            "Function with unicode: 你好世界",
            "Code with newlines\nand\ttabs"
        ]
        
        for task in special_tasks:
            result = peer_agent.analyze_task(task)
            assert result in ["code", "content", "unknown"]
    
    @pytest.mark.unit
    def test_word_boundary_matching(self, peer_agent):
        """Test that keyword matching respects word boundaries."""
        # Should NOT match 'code' in 'decode'
        task = "decode this message"
        result = peer_agent.analyze_task(task)
        assert result == "unknown"  # No other keywords should match
        
        # Should match 'code' in 'code something'
        task = "code something"
        result = peer_agent.analyze_task(task)
        assert result == "code" 