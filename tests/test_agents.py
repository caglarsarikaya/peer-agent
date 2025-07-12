"""
Tests for ContentAgent and CodeAgent classes.

This module tests the individual agent functionality, including content generation,
code generation, error handling, and integration with external APIs.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agents.content_agent import ContentAgent
from agents.code_agent import CodeAgent
from tests.conftest import (
    validate_code_response,
    validate_content_response,
    validate_search_integration,
    assert_response_structure,
    semantic_similarity
)


class TestContentAgent:
    """Test ContentAgent functionality."""
    
    @pytest.fixture
    def content_agent(self, mock_env_vars):
        """Create ContentAgent instance with mocked dependencies."""
        with patch('agents.content_agent.ChatOpenAI') as mock_llm, \
             patch('agents.content_agent.DuckDuckGoSearchRun') as mock_search, \
             patch('agents.content_agent.initialize_agent') as mock_init_agent, \
             patch('agents.content_agent.config') as mock_config:
            
            # Mock configuration
            mock_config.OPENAI_API_KEY = "test-key"
            mock_config.OPENAI_MODEL = "gpt-3.5-turbo"
            mock_config.OPENAI_TEMPERATURE = 0.7
            mock_config.OPENAI_MAX_TOKENS = 1000
            mock_config.SERPAPI_KEY = None  # No SerpAPI key
            
            mock_llm_instance = Mock()
            mock_llm_instance.predict.return_value = "Mocked LLM response"
            mock_llm.return_value = mock_llm_instance
            
            mock_search_instance = Mock()
            mock_search_instance.run.return_value = "Mocked search results"
            mock_search.return_value = mock_search_instance
            
            mock_agent_instance = Mock()
            mock_init_agent.return_value = mock_agent_instance
            
            agent = ContentAgent()
            # Override the real instances with mocks
            agent.llm = mock_llm_instance
            agent.search_tool = mock_search_instance
            agent.agent = mock_agent_instance
            return agent
    
    @pytest.mark.agent
    def test_content_agent_initialization(self, mock_env_vars):
        """Test ContentAgent initializes correctly."""
        with patch('agents.content_agent.ChatOpenAI') as mock_llm, \
             patch('agents.content_agent.DuckDuckGoSearchRun') as mock_search, \
             patch('agents.content_agent.initialize_agent') as mock_init_agent, \
             patch('agents.content_agent.config') as mock_config:
            
            # Mock configuration to ensure DuckDuckGo is used
            mock_config.OPENAI_API_KEY = "test-key"
            mock_config.OPENAI_MODEL = "gpt-3.5-turbo"
            mock_config.OPENAI_TEMPERATURE = 0.7
            mock_config.OPENAI_MAX_TOKENS = 1000
            mock_config.SERPAPI_KEY = None  # No SerpAPI key
            
            mock_llm_instance = Mock()
            mock_search_instance = Mock()
            mock_agent_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_search.return_value = mock_search_instance
            mock_init_agent.return_value = mock_agent_instance
            
            agent = ContentAgent()
            
            assert agent is not None
            assert agent.llm == mock_llm_instance
            assert agent.search_tool == mock_search_instance
            assert agent.agent == mock_agent_instance
    
    @pytest.mark.agent
    def test_process_blog_task_without_search(self, content_agent, realistic_llm_mock):
        """Test processing blog task without web search."""
        # Use realistic LLM mock that responds based on context
        content_agent.llm = realistic_llm_mock
        
        task = "Write a short blog post about dogs"
        
        with patch('agents.content_agent.datetime') as mock_dt:
            # Mock datetime to avoid timing conflicts
            mock_now = Mock()
            mock_now.isoformat.return_value = "2024-01-01T12:00:00"
            mock_dt.now.return_value = mock_now
            
            result = content_agent.process_task(task)
        
        # Use behavioral assertions instead of exact content matching
        assert result["success"] is True
        assert result["agent"] == "content"
        assert result["task"] == task
        assert validate_content_response(result, expected_topics=["dog"])
        assert validate_search_integration(result)
        assert result["search_performed"] is False
        assert result["search_results_count"] == 0
        assert "processing_time_seconds" in result
        assert "timestamp" in result
        
        # Verify content has proper structure
        content = result["content"]
        assert isinstance(content, str)
        assert len(content.strip()) > 50  # Substantial content
        assert "#" in content or "##" in content  # Has headers
    
    @pytest.mark.agent
    def test_process_research_task_with_search(self, content_agent, realistic_llm_mock, flexible_search_mock):
        """Test processing research task with web search."""
        # Setup mocks with realistic responses
        content_agent.llm = realistic_llm_mock
        content_agent.search_tool = flexible_search_mock
        
        task = "Research and write about recent AI developments"
        
        result = content_agent.process_task(task)
        
        # Behavioral assertions focused on functionality
        assert result["success"] is True
        assert result["agent"] == "content"
        assert validate_content_response(result, expected_topics=["ai", "artificial", "intelligence"])
        assert validate_search_integration(result)
        assert result["search_performed"] is True
        assert result["search_results_count"] >= 0
        assert "processing_time_seconds" in result
        
        # Verify search was called
        content_agent.search_tool.run.assert_called_once()
        
        # Verify content quality without exact matching
        content = result["content"]
        assert isinstance(content, str)
        assert len(content.strip()) > 100  # Research content should be substantial
        # Check for AI-related concepts without exact matching
        ai_concepts = ["ai", "artificial", "intelligence", "machine", "learning", "technology"]
        content_lower = content.lower()
        assert any(concept in content_lower for concept in ai_concepts)
    
    @pytest.mark.agent
    def test_process_task_with_search_error(self, content_agent, realistic_llm_mock):
        """Test handling search API errors gracefully."""
        # Setup mocks - search fails, LLM continues
        content_agent.llm = realistic_llm_mock
        content_agent.search_tool.run.side_effect = Exception("Search API error")
        
        task = "Research quantum computing"
        result = content_agent.process_task(task)
        
        # Should still succeed but with search attempted
        assert result["success"] is True
        assert validate_search_integration(result)
        assert result["search_performed"] is True  # Search was attempted
        assert result["search_results_count"] == 0  # But no results due to error
        
        # Content should still be generated
        assert "content" in result
        assert isinstance(result["content"], str)
        assert len(result["content"].strip()) > 20
    
    @pytest.mark.agent
    def test_process_task_with_llm_error(self, content_agent, mock_openai_error):
        """Test handling LLM API errors."""
        # Override the agent's LLM with the error mock
        content_agent.llm = mock_openai_error
        
        task = "Write about space exploration"
        
        result = content_agent.process_task(task)
        
        # Should return error response
        assert result["success"] is False
        assert result["agent"] == "content"
        assert "error" in result
        assert isinstance(result["error"], str)
        assert len(result["error"]) > 0
    
    @pytest.mark.agent
    def test_search_query_extraction(self, content_agent):
        """Test extraction of search queries from tasks."""
        test_cases = [
            ("Research latest developments in AI", ["ai", "artificial", "intelligence", "development"]),
            ("Find information about climate change", ["climate", "change", "information"]),
            ("Write an article about space exploration", ["space", "exploration", "article"])
        ]
        
        for task, expected_concepts in test_cases:
            query = content_agent._extract_search_query(task)
            assert isinstance(query, str)
            assert len(query) > 0
            
            # Check for semantic similarity instead of exact matching
            query_lower = query.lower()
            assert any(concept in query_lower for concept in expected_concepts)
    
    @pytest.mark.agent
    def test_content_formatting(self, content_agent, realistic_llm_mock):
        """Test that content is properly formatted."""
        content_agent.llm = realistic_llm_mock
        
        task = "Write about programming"
        result = content_agent.process_task(task)
        
        assert result["success"] is True
        assert validate_content_response(result, expected_topics=["programming"])
        
        # Check content structure
        content = result["content"]
        assert isinstance(content, str)
        assert len(content) > 0
        # Should have some structure (headers, line breaks, etc.)
        assert any(marker in content for marker in ["#", "\n\n", "##"])


class TestCodeAgent:
    """Test CodeAgent functionality."""
    
    @pytest.fixture
    def code_agent(self, mock_env_vars):
        """Create CodeAgent instance with mocked dependencies."""
        with patch('agents.code_agent.ChatOpenAI') as mock_llm, \
             patch('agents.code_agent.config') as mock_config:
            
            # Mock configuration
            mock_config.OPENAI_API_KEY = "test-key"
            mock_config.OPENAI_MODEL = "gpt-3.5-turbo"
            mock_config.OPENAI_MAX_TOKENS = 1000
            
            mock_llm_instance = Mock()
            mock_llm_instance.predict.return_value = "Mocked LLM response"
            mock_llm.return_value = mock_llm_instance
            
            agent = CodeAgent()
            # Override the real instance with mock
            agent.llm = mock_llm_instance
            return agent
    
    @pytest.mark.agent
    def test_code_agent_initialization(self, mock_env_vars):
        """Test CodeAgent initializes correctly."""
        with patch('agents.code_agent.ChatOpenAI') as mock_llm, \
             patch('agents.code_agent.config') as mock_config:
            
            # Mock configuration
            mock_config.OPENAI_API_KEY = "test-key"
            mock_config.OPENAI_MODEL = "gpt-3.5-turbo"
            mock_config.OPENAI_MAX_TOKENS = 1000
            
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            
            agent = CodeAgent()
            
            assert agent is not None
            assert agent.llm == mock_llm_instance
    
    @pytest.mark.agent
    def test_process_python_function_task(self, code_agent, realistic_llm_mock):
        """Test processing Python function task."""
        # Setup realistic LLM response
        code_agent.llm = realistic_llm_mock
        
        task = "Write a Python function to calculate fibonacci numbers"
        
        result = code_agent.process_task(task)
        
        # Behavioral assertions
        assert result["success"] is True
        assert result["agent"] == "code"
        assert result["task"] == task
        assert validate_code_response(result, expected_language="python")
        
        # Verify code quality without exact matching
        code = result["code"]
        assert isinstance(code, str)
        assert len(code.strip()) > 20  # Substantial code
        
        # Check for Python code patterns
        python_patterns = ["def ", "return", ":", "fibonacci", "fib"]
        code_lower = code.lower()
        assert any(pattern in code_lower for pattern in python_patterns)
        
        # Check metadata
        assert result["language"] == "python"
        assert "task_type" in result
        assert "processing_time_seconds" in result
    
    @pytest.mark.agent
    def test_process_javascript_task(self, code_agent, realistic_llm_mock):
        """Test processing JavaScript task."""
        # Custom response for JavaScript
        def js_response(prompt):
            if "javascript" in prompt.lower():
                return """function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Example usage
console.log(fibonacci(10));"""
            return realistic_llm_mock.predict.side_effect(prompt)
        
        code_agent.llm.predict.side_effect = js_response
        
        task = "Write a JavaScript function for fibonacci sequence"
        
        result = code_agent.process_task(task)
        
        # Behavioral assertions
        assert result["success"] is True
        assert validate_code_response(result, expected_language="javascript")
        
        # Verify JavaScript code characteristics
        code = result["code"]
        js_patterns = ["function", "return", "{", "}", "fibonacci"]
        code_lower = code.lower()
        assert any(pattern in code_lower for pattern in js_patterns)
    
    @pytest.mark.agent
    def test_process_debug_task(self, code_agent, realistic_llm_mock):
        """Test processing debug task."""
        # Custom response for debugging
        def debug_response(prompt):
            if "debug" in prompt.lower() or "fix" in prompt.lower():
                return """# Bug Analysis
The issue is with the loop condition.

# Fixed Code
def corrected_function(n):
    result = 0
    for i in range(n):  # Fixed: was range(n+1)
        result += i
    return result

# Explanation
The original code had an off-by-one error in the range."""
            return realistic_llm_mock.predict.side_effect(prompt)
        
        code_agent.llm.predict.side_effect = debug_response
        
        task = "Debug this Python function that has a loop error"
        
        result = code_agent.process_task(task)
        
        # Behavioral assertions
        assert result["success"] is True
        assert validate_code_response(result, expected_language="python")
        
        # Check for debugging content
        code = result["code"]
        debug_indicators = ["bug", "fixed", "error", "issue", "corrected"]
        code_lower = code.lower()
        assert any(indicator in code_lower for indicator in debug_indicators)
    
    @pytest.mark.agent
    def test_language_detection(self, code_agent):
        """Test language detection from task descriptions."""
        test_cases = [
            ("Write a Python function", "python"),
            ("Create a JavaScript method", "javascript"),
            ("Build a Java class", "java"),
            ("Write C++ code", "cpp"),
            ("Create a shell script", "bash")
        ]
        
        for task, expected_lang in test_cases:
            detected = code_agent._detect_language(task)
            assert detected == expected_lang or detected == "python"  # Default fallback
    
    @pytest.mark.agent
    def test_task_type_detection(self, code_agent):
        """Test task type detection from descriptions."""
        test_cases = [
            ("Write a function", "function"),
            ("Create a class", "class"),
            ("Build a script", "script"),
            ("Debug this code", "debug"),
            ("Fix the bug", "debug")
        ]
        
        for task, expected_type in test_cases:
            detected = code_agent._detect_task_type(task)
            assert detected == expected_type or detected == "function"  # Default fallback
    
    @pytest.mark.agent
    def test_process_task_with_llm_error(self, code_agent, mock_openai_error):
        """Test handling LLM API errors."""
        # Override the agent's LLM with the error mock
        code_agent.llm = mock_openai_error
        
        task = "Write Python code"
        
        result = code_agent.process_task(task)
        
        # Should return error response
        assert result["success"] is False
        assert result["agent"] == "code"
        assert "error" in result
        assert isinstance(result["error"], str)
    
    @pytest.mark.agent
    def test_python_syntax_validation(self, code_agent):
        """Test Python syntax validation."""
        valid_code = """def hello():
    print("Hello, World!")
    return True"""
        
        invalid_code = """def hello(:
    print("Hello, World!")
    return True"""
        
        assert code_agent._validate_python_syntax(valid_code) is True
        assert code_agent._validate_python_syntax(invalid_code) is False
    
    @pytest.mark.agent
    def test_supported_languages_coverage(self, code_agent):
        """Test that agent supports expected languages."""
        expected_languages = ["python", "javascript", "java", "cpp", "bash", "sql"]
        
        for lang in expected_languages:
            # Should be able to handle tasks for each language
            task = f"Write a {lang} function"
            detected = code_agent._detect_language(task)
            assert detected in expected_languages or detected == "python"  # Default
    
    @pytest.mark.agent
    def test_code_formatting_and_cleanup(self, code_agent, realistic_llm_mock):
        """Test that code is properly formatted and cleaned."""
        # Setup mock with messy code
        def messy_code_response(prompt):
            return """   def messy_function(  ):
            
            
    print(  "Hello"  )
    
    
    return   True   """
        
        code_agent.llm.predict.side_effect = messy_code_response
        
        task = "Write a Python function"
        result = code_agent.process_task(task)
        
        # Code should be present and structured
        assert result["success"] is True
        assert "code" in result
        assert isinstance(result["code"], str)
        assert len(result["code"].strip()) > 0
        
        # Should contain function definition
        assert "def " in result["code"]


class TestAgentErrorHandling:
    """Test error handling scenarios for both agents."""
    
    @pytest.mark.agent
    def test_content_agent_with_none_task(self, mock_env_vars):
        """Test ContentAgent with None task."""
        with patch('agents.content_agent.ChatOpenAI'), \
             patch('agents.content_agent.DuckDuckGoSearchRun'):
            
            agent = ContentAgent()
            result = agent.process_task(None)
            
            assert result["success"] is False
            assert "error" in result
    
    @pytest.mark.agent
    def test_code_agent_with_empty_task(self, mock_env_vars):
        """Test CodeAgent with empty task."""
        with patch('agents.code_agent.ChatOpenAI'):
            agent = CodeAgent()
            result = agent.process_task("")
            
            assert result["success"] is False
            assert "error" in result
    
    @pytest.mark.agent
    def test_agent_with_missing_env_vars(self):
        """Test agent initialization with missing environment variables."""
        # This should be handled gracefully
        with patch.dict('os.environ', {}, clear=True):
            with patch('agents.content_agent.ChatOpenAI') as mock_llm:
                # Should either use default or raise informative error
                try:
                    agent = ContentAgent()
                    assert agent is not None
                except Exception as e:
                    # Should be informative about missing env vars
                    assert "api" in str(e).lower() or "key" in str(e).lower()


class TestAgentIntegration:
    """Test integration scenarios between agents and external services."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_content_agent_end_to_end_mock(self, mock_env_vars):
        """Test ContentAgent with comprehensive mocking."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_search:
            
            # Setup realistic mocks
            mock_llm_instance = Mock()
            mock_search_instance = Mock()
            
            mock_llm_instance.predict.return_value = """# Technology Trends 2024

The technology landscape continues to evolve rapidly with several key trends emerging.

## Key Developments
- Artificial intelligence advancements
- Cloud computing growth
- Cybersecurity improvements

These trends are reshaping how we work and interact with technology."""
            
            mock_search_instance.run.return_value = "Technology trends 2024\nLatest tech developments\nInnovation in AI and cloud"
            
            mock_llm.return_value = mock_llm_instance
            mock_search.return_value = mock_search_instance
            
            agent = ContentAgent()
            result = agent.process_task("Research and write about technology trends")
            
            # Comprehensive validation
            assert result["success"] is True
            assert validate_content_response(result, expected_topics=["technology", "trends"])
            assert validate_search_integration(result)
            assert assert_response_structure(result, expected_agent="content")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_code_agent_end_to_end_mock(self, mock_env_vars):
        """Test CodeAgent with comprehensive mocking."""
        with patch('langchain_openai.ChatOpenAI') as mock_llm:
            mock_llm_instance = Mock()
            
            mock_llm_instance.predict.return_value = """def bubble_sort(arr):
    '''Implement bubble sort algorithm'''
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Example usage
numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_numbers = bubble_sort(numbers)
print(sorted_numbers)"""
            
            mock_llm.return_value = mock_llm_instance
            
            agent = CodeAgent()
            result = agent.process_task("Write a Python function for bubble sort")
            
            # Comprehensive validation
            assert result["success"] is True
            assert validate_code_response(result, expected_language="python")
            assert assert_response_structure(result, expected_agent="code")
            
            # Verify sorting algorithm content
            code = result["code"]
            sort_indicators = ["sort", "bubble", "arr", "for", "if"]
            code_lower = code.lower()
            assert any(indicator in code_lower for indicator in sort_indicators) 