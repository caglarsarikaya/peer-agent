"""
Peer Agent - Central task router for delegating user tasks to specialized AI agents.

This module implements the main Peer Agent that receives user tasks, analyzes them,
and routes them to appropriate sub-agents based on keyword detection.
"""

from typing import Dict, Any, Optional
import logging
import re

# Import available agents
from .content_agent import ContentAgent
from .code_agent import CodeAgent

logger = logging.getLogger(__name__)


class PeerAgent:
    """
    Central Peer Agent that routes user tasks to specialized sub-agents.
    
    The Peer Agent uses keyword-based routing to identify task types and
    delegate them to the appropriate specialized agents.
    """
    
    def __init__(self):
        """Initialize the Peer Agent with routing rules and sub-agents."""
        # Define routing keywords for different agent types
        self.routing_keywords = {
            'content': [
                'blog', 'article', 'write', 'content', 'post', 'story',
                'research', 'search', 'information', 'news', 'topic'
            ],
            'code': [
                'code', 'program', 'script', 'function', 'class', 'api',
                'python', 'javascript', 'java', 'c++', 'html', 'css',
                'algorithm', 'debug', 'fix', 'implement', 'develop'
            ]
        }
        
        # Initialize sub-agents
        self.content_agent = ContentAgent()
        self.code_agent = CodeAgent()
        
        logger.info("Peer Agent initialized with routing keywords")
    
    def analyze_task(self, task: str) -> str:
        """
        Analyze the user task and determine which agent should handle it.
        
        Args:
            task (str): The user's task description
            
        Returns:
            str: The agent type ('content', 'code', or 'unknown')
        """
        if not task or not task.strip():
            return 'unknown'
        
        task_lower = task.lower()
        
        # Count keyword matches for each agent type
        agent_scores = {}
        
        for agent_type, keywords in self.routing_keywords.items():
            score = 0
            for keyword in keywords:
                # Use word boundaries to avoid partial matches
                if re.search(r'\b' + re.escape(keyword) + r'\b', task_lower):
                    score += 1
            agent_scores[agent_type] = score
        
        # Return the agent type with the highest score
        if max(agent_scores.values()) > 0:
            best_agent = max(agent_scores, key=agent_scores.get)
            logger.info(f"Task routed to {best_agent} agent. Scores: {agent_scores}")
            return best_agent
        
        logger.warning(f"No suitable agent found for task: {task[:50]}...")
        return 'unknown'
    
    def route_task(self, task: str) -> Dict[str, Any]:
        """
        Route the task to the appropriate agent and return the result.
        
        Args:
            task (str): The user's task description
            
        Returns:
            Dict[str, Any]: Response containing agent_type, success status, and result
        """
        try:
            agent_type = self.analyze_task(task)
            
            if agent_type == 'content':
                result = self.content_agent.process_task(task)
                
            elif agent_type == 'code':
                result = self.code_agent.process_task(task)
                
            else:
                result = {
                    "error": "Unknown task type",
                    "message": "I couldn't determine how to handle this task. Please try rephrasing with keywords like 'write', 'blog', 'code', or 'program'.",
                    "task": task
                }
                
            return {
                "agent_type": agent_type,
                "success": "error" not in result,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error routing task: {str(e)}")
            return {
                "agent_type": "error",
                "success": False,
                "result": {
                    "error": "Internal error",
                    "message": f"An error occurred while processing the task: {str(e)}"
                }
            }
    
    def get_available_agents(self) -> Dict[str, list]:
        """
        Get information about available agents and their capabilities.
        
        Returns:
            Dict[str, list]: Dictionary mapping agent types to their keywords
        """
        return self.routing_keywords.copy()
    
    def add_routing_keyword(self, agent_type: str, keyword: str) -> bool:
        """
        Add a new routing keyword for an agent type.
        
        Args:
            agent_type (str): The agent type ('content' or 'code')
            keyword (str): The keyword to add
            
        Returns:
            bool: True if successfully added, False otherwise
        """
        if agent_type in self.routing_keywords:
            if keyword.lower() not in self.routing_keywords[agent_type]:
                self.routing_keywords[agent_type].append(keyword.lower())
                logger.info(f"Added keyword '{keyword}' to {agent_type} agent")
                return True
        return False 