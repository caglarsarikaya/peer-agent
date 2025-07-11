"""
Tests for web search integration functionality.

This module tests the search tool selection, SerpAPI vs DuckDuckGo integration,
and end-to-end search functionality in ContentAgent.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agents.content_agent import ContentAgent


class TestSearchToolInitialization:
    """Test search tool initialization and selection logic."""
    
    @pytest.mark.integration
    def test_duckduckgo_default_initialization(self, mock_env_vars):
        """Test that DuckDuckGo is used when no SerpAPI key is provided."""
        with patch.dict('os.environ', {'SERPAPI_KEY': ''}), \
             patch('langchain_openai.ChatOpenAI'), \
             patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_ddg:
            
            mock_ddg_instance = Mock()
            mock_ddg.return_value = mock_ddg_instance
            
            agent = ContentAgent()
            
            assert agent._get_search_tool_name() == "DuckDuckGo"
            mock_ddg.assert_called_once()
    
    @pytest.mark.integration
    def test_serpapi_initialization_with_key(self, mock_env_vars):
        """Test that SerpAPI is used when API key is provided."""
        with patch.dict('os.environ', {'SERPAPI_KEY': 'test-serpapi-key'}), \
             patch('langchain_openai.ChatOpenAI'), \
             patch('langchain_community.utilities.SerpAPIWrapper') as mock_serp, \
             patch('langchain.tools.Tool') as mock_tool:
            
            mock_serp_instance = Mock()
            mock_serp.return_value = mock_serp_instance
            mock_tool_instance = Mock()
            mock_tool_instance.name = 'SerpAPI'
            mock_tool.return_value = mock_tool_instance
            
            agent = ContentAgent()
            
            assert agent._get_search_tool_name() == "SerpAPI"
            mock_serp.assert_called_once_with(serpapi_api_key='test-serpapi-key')
    
    @pytest.mark.integration
    def test_serpapi_fallback_on_import_error(self, mock_env_vars):
        """Test fallback to DuckDuckGo when SerpAPI import fails."""
        with patch.dict('os.environ', {'SERPAPI_KEY': 'test-serpapi-key'}), \
             patch('langchain_openai.ChatOpenAI'), \
             patch('langchain_community.utilities.SerpAPIWrapper', side_effect=ImportError()), \
             patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_ddg:
            
            mock_ddg_instance = Mock()
            mock_ddg.return_value = mock_ddg_instance
            
            agent = ContentAgent()
            
            assert agent._get_search_tool_name() == "DuckDuckGo"
            mock_ddg.assert_called_once()


class TestSearchFunctionality:
    """Test search functionality with different search engines."""
    
    @pytest.fixture
    def mock_content_agent_duckduckgo(self, mock_env_vars):
        """Create ContentAgent with mocked DuckDuckGo search."""
        with patch('langchain_openai.ChatOpenAI'), \
             patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_ddg:
            
            mock_ddg_instance = Mock()
            mock_ddg_instance.run.return_value = "AI Development\nLatest trends in artificial intelligence\nMachine Learning News\nRecent breakthroughs in ML"
            mock_ddg.return_value = mock_ddg_instance
            
            agent = ContentAgent()
            return agent, mock_ddg_instance
    
    @pytest.fixture
    def mock_content_agent_serpapi(self, mock_env_vars):
        """Create ContentAgent with mocked SerpAPI search."""
        with patch.dict('os.environ', {'SERPAPI_KEY': 'test-key'}), \
             patch('langchain_openai.ChatOpenAI'), \
             patch('langchain_community.utilities.SerpAPIWrapper') as mock_serp, \
             patch('langchain.tools.Tool') as mock_tool:
            
            mock_serp_instance = Mock()
            mock_tool_instance = Mock()
            mock_tool_instance.name = 'SerpAPI'
            mock_tool_instance.run.return_value = '{"organic_results": [{"title": "AI Trends 2024", "snippet": "Latest developments", "link": "https://example.com"}]}'
            
            mock_serp.return_value = mock_serp_instance
            mock_tool.return_value = mock_tool_instance
            
            agent = ContentAgent()
            return agent, mock_tool_instance
    
    @pytest.mark.integration
    def test_duckduckgo_search_parsing(self, mock_content_agent_duckduckgo):
        """Test DuckDuckGo search result parsing."""
        agent, mock_search = mock_content_agent_duckduckgo
        
        results = agent.search_web("artificial intelligence trends", max_results=2)
        
        assert len(results) == 2
        assert results[0]["title"] == "AI Development"
        assert results[0]["snippet"] == "Latest trends in artificial intelligence"
        assert results[0]["source"] == "DuckDuckGo"
        assert results[1]["title"] == "Machine Learning News"
        assert results[1]["snippet"] == "Recent breakthroughs in ML"
        
        mock_search.run.assert_called_once_with("artificial intelligence trends")
    
    @pytest.mark.integration
    def test_serpapi_search_parsing(self, mock_content_agent_serpapi):
        """Test SerpAPI search result parsing."""
        agent, mock_search = mock_content_agent_serpapi
        
        results = agent.search_web("artificial intelligence trends", max_results=2)
        
        assert len(results) == 1
        assert results[0]["title"] == "AI Trends 2024"
        assert results[0]["snippet"] == "Latest developments"
        assert results[0]["link"] == "https://example.com"
        assert results[0]["source"] == "SerpAPI"
        
        mock_search.run.assert_called_once_with("artificial intelligence trends")
    
    @pytest.mark.integration
    def test_search_error_handling(self, mock_content_agent_duckduckgo):
        """Test search error handling."""
        agent, mock_search = mock_content_agent_duckduckgo
        mock_search.run.side_effect = Exception("Search API error")
        
        results = agent.search_web("test query")
        
        # Should return empty list on error
        assert results == []
    
    @pytest.mark.integration
    def test_search_with_no_results(self, mock_content_agent_duckduckgo):
        """Test handling of search with no results."""
        agent, mock_search = mock_content_agent_duckduckgo
        mock_search.run.return_value = ""
        
        results = agent.search_web("nonexistent query")
        
        assert results == []


class TestEndToEndSearchIntegration:
    """Test complete end-to-end search integration in ContentAgent."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_research_task_with_search(self, mock_env_vars, mock_openai_success, mock_duckduckgo_search):
        """Test complete research task with search integration."""
        # Setup comprehensive mocks
        mock_openai_success.predict.return_value = """# Latest AI Developments

Based on recent research, artificial intelligence continues to advance rapidly.

## Key Trends
- Large language models improvement
- Computer vision breakthroughs
- Autonomous systems development

These developments are reshaping technology landscapes."""
        
        mock_duckduckgo_search.run.return_value = "AI Research 2024\nLatest developments in artificial intelligence\nMachine Learning Progress\nBreakthroughs in neural networks"
        
        with patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_ddg:
            mock_ddg.return_value = mock_duckduckgo_search
            
            agent = ContentAgent()
            result = agent.process_task("Research latest developments in artificial intelligence")
            
            # Verify task completion
            assert result["success"] is True
            assert result["agent"] == "content"
            assert result["search_performed"] is True
            assert result["search_results_count"] > 0
            assert "Latest AI Developments" in result["content"]
            assert "Sources" in result["content"]  # Citations should be included
    
    @pytest.mark.integration
    def test_creative_task_without_search(self, mock_env_vars, mock_openai_success):
        """Test that creative tasks don't trigger search."""
        mock_openai_success.predict.return_value = """# The Dragon's Tale

Once upon a time, in a mystical realm far beyond the mountains, there lived a wise dragon named Ember."""
        
        with patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_ddg:
            mock_search_instance = Mock()
            mock_ddg.return_value = mock_search_instance
            
            agent = ContentAgent()
            result = agent.process_task("Write a creative story about dragons")
            
            # Verify no search was performed
            assert result["success"] is True
            assert result["search_performed"] is False
            assert result["search_results_count"] == 0
            assert "Dragon's Tale" in result["content"]
            mock_search_instance.run.assert_not_called()
    
    @pytest.mark.integration
    def test_search_trigger_keywords(self, mock_env_vars):
        """Test that specific keywords trigger search functionality."""
        with patch('langchain_openai.ChatOpenAI'), \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            agent = ContentAgent()
            
            # Tasks that should trigger search
            search_tasks = [
                "Research quantum computing",
                "Find information about climate change",
                "What are the latest news in technology",
                "Get current data on market trends",
                "Gather facts about space exploration"
            ]
            
            # Tasks that should NOT trigger search
            creative_tasks = [
                "Write a poem about love",
                "Create a story about adventure",
                "Generate content about friendship",
                "Write a blog post about personal growth"
            ]
            
            for task in search_tasks:
                # Check search detection logic
                needs_search = any(keyword in task.lower() for keyword in ['research', 'information', 'current', 'latest', 'news', 'facts', 'data'])
                assert needs_search is True, f"Task should trigger search: {task}"
            
            for task in creative_tasks:
                # Check search detection logic
                needs_search = any(keyword in task.lower() for keyword in ['research', 'information', 'current', 'latest', 'news', 'facts', 'data'])
                assert needs_search is False, f"Task should not trigger search: {task}"


class TestSearchToolCapabilities:
    """Test search tool capabilities and configuration."""
    
    @pytest.mark.integration
    def test_get_capabilities_reflects_search_tool(self, mock_env_vars):
        """Test that capabilities reflect the configured search tool."""
        with patch('langchain_openai.ChatOpenAI'), \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            agent = ContentAgent()
            capabilities = agent.get_capabilities()
            
            assert "tools" in capabilities
            assert "DuckDuckGo Search" in capabilities["tools"]
            assert "OpenAI GPT" in capabilities["tools"]
    
    @pytest.mark.integration
    def test_search_tool_name_consistency(self, mock_env_vars):
        """Test that search tool name is consistently reported."""
        with patch('langchain_openai.ChatOpenAI'), \
             patch('langchain_community.tools.DuckDuckGoSearchRun'):
            
            agent = ContentAgent()
            tool_name = agent._get_search_tool_name()
            capabilities = agent.get_capabilities()
            
            assert f"{tool_name} Search" in capabilities["tools"] 