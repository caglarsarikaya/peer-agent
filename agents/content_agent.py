"""
Content Agent - Specialized AI agent for content creation and research tasks.

This module implements the ContentAgent that handles blog writing, research,
and content generation tasks using web search and OpenAI GPT integration.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# LangChain imports
from langchain.agents import AgentType, initialize_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.llms import OpenAI
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI

# Configuration
from config import config

logger = logging.getLogger(__name__)


class ContentAgent:
    """
    Content Agent specialized in research and content creation tasks.
    
    This agent can:
    - Search the web for relevant information
    - Generate blog posts and articles
    - Include citations and sources
    - Create well-structured content
    """
    
    def __init__(self):
        """Initialize the Content Agent with search tools and LLM."""
        try:
            # Initialize OpenAI chat model
            self.llm = ChatOpenAI(
                openai_api_key=config.OPENAI_API_KEY,
                model_name=config.OPENAI_MODEL,
                temperature=config.OPENAI_TEMPERATURE,
                max_tokens=config.OPENAI_MAX_TOKENS
            )
            
            # Initialize search tool (using DuckDuckGo as it doesn't require API key)
            self.search_tool = DuckDuckGoSearchRun()
            
            # Initialize agent with tools
            tools = [self.search_tool]
            self.agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True
            )
            
            logger.info("Content Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Content Agent: {str(e)}")
            raise
    
    def search_web(self, query: str, max_results: int = None) -> List[Dict[str, str]]:
        """
        Search the web for information related to the query.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, str]]: List of search results with title and snippet
        """
        try:
            if max_results is None:
                max_results = config.MAX_SEARCH_RESULTS
            
            logger.info(f"Searching web for: {query}")
            
            # Perform search using DuckDuckGo
            search_results = self.search_tool.run(query)
            
            # Parse and format results
            results = []
            if search_results:
                # Split results and take first few
                result_lines = search_results.split('\n')[:max_results * 2]  # Rough estimate
                
                current_result = {}
                for line in result_lines:
                    if line.strip():
                        if not current_result:
                            current_result = {"title": line.strip(), "snippet": ""}
                        else:
                            current_result["snippet"] = line.strip()
                            results.append(current_result)
                            current_result = {}
                        
                        if len(results) >= max_results:
                            break
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return []
    
    def generate_content(self, task: str, search_results: List[Dict[str, str]] = None) -> str:
        """
        Generate content using OpenAI GPT with optional search context.
        
        Args:
            task (str): Content generation task
            search_results (List[Dict]): Optional search results for context
            
        Returns:
            str: Generated content
        """
        try:
            # Prepare context from search results
            context = ""
            if search_results:
                context = "\n\nRelevant information from web search:\n"
                for i, result in enumerate(search_results, 1):
                    context += f"\n{i}. {result.get('title', 'No title')}\n"
                    context += f"   {result.get('snippet', 'No description')}\n"
            
            # Create a comprehensive prompt
            prompt = f"""You are a professional content writer and researcher. Your task is to {task}.

Guidelines:
- Create well-structured, engaging content
- Use a professional yet accessible tone
- Include relevant sections with clear headings
- If web search results are provided, incorporate the information naturally
- Add citations where appropriate using [1], [2], etc. format
- Ensure accuracy and provide valuable insights
- Make the content informative and actionable

{context}

Please create the content now:"""

            # Generate content using OpenAI
            response = self.llm.predict(prompt)
            
            # Add citations section if search results were used
            if search_results:
                response += "\n\n## Sources\n"
                for i, result in enumerate(search_results, 1):
                    response += f"[{i}] {result.get('title', 'Unknown source')}\n"
            
            logger.info("Content generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}")
            return f"I apologize, but I encountered an error while generating content: {str(e)}"
    
    def process_task(self, task: str) -> Dict[str, Any]:
        """
        Process a content creation task end-to-end.
        
        Args:
            task (str): The content task description
            
        Returns:
            Dict[str, Any]: Response with generated content and metadata
        """
        try:
            start_time = datetime.now()
            logger.info(f"Processing content task: {task[:100]}...")
            
            # Determine if we need web search
            search_keywords = ['research', 'information', 'current', 'latest', 'news', 'facts', 'data']
            needs_search = any(keyword in task.lower() for keyword in search_keywords)
            
            search_results = []
            if needs_search:
                # Extract search query from task
                search_query = self._extract_search_query(task)
                search_results = self.search_web(search_query)
            
            # Generate content
            content = self.generate_content(task, search_results)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = {
                "agent": "content",
                "task": task,
                "content": content,
                "search_performed": needs_search,
                "search_results_count": len(search_results),
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            logger.info(f"Content task completed in {processing_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Error processing content task: {str(e)}")
            return {
                "agent": "content",
                "task": task,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_search_query(self, task: str) -> str:
        """
        Extract a search query from the task description.
        
        Args:
            task (str): Task description
            
        Returns:
            str: Optimized search query
        """
        # Simple extraction - remove common task words and focus on main topic
        task_lower = task.lower()
        
        # Remove common instruction words
        remove_words = ['write', 'create', 'generate', 'blog', 'article', 'post', 'about', 'on', 'the']
        words = task.split()
        
        filtered_words = [word for word in words if word.lower() not in remove_words]
        
        # Take first 5-6 meaningful words for search
        search_query = ' '.join(filtered_words[:6])
        
        logger.info(f"Extracted search query: '{search_query}' from task: '{task}'")
        return search_query if search_query else task
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about the Content Agent's capabilities.
        
        Returns:
            Dict[str, Any]: Agent capabilities and configuration
        """
        return {
            "agent_type": "content",
            "capabilities": [
                "Blog and article writing",
                "Web research and information gathering",
                "Content creation with citations",
                "SEO-friendly content structure",
                "Multi-format content generation"
            ],
            "tools": ["DuckDuckGo Search", "OpenAI GPT"],
            "max_search_results": config.MAX_SEARCH_RESULTS,
            "model": config.OPENAI_MODEL,
            "enabled": config.CONTENT_AGENT_ENABLED
        } 