"""
Code Agent - Specialized AI agent for code generation and programming tasks.

This module implements the CodeAgent that handles code writing, debugging,
and programming tasks using OpenAI GPT integration.
"""

from typing import Dict, Any, List, Optional
import logging
import re
from datetime import datetime

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Configuration
from config import config

logger = logging.getLogger(__name__)


class CodeAgent:
    """
    Code Agent specialized in programming and software development tasks.
    
    This agent can:
    - Generate code in multiple programming languages
    - Debug and fix code issues
    - Explain code functionality
    - Create complete functions and classes
    - Provide best practices and optimization suggestions
    """
    
    def __init__(self):
        """Initialize the Code Agent with OpenAI LLM."""
        try:
            # Initialize OpenAI chat model with settings optimized for code
            self.llm = ChatOpenAI(
                openai_api_key=config.OPENAI_API_KEY,
                model_name=config.OPENAI_MODEL,
                temperature=0.1,  # Lower temperature for more deterministic code
                max_tokens=config.OPENAI_MAX_TOKENS
            )
            
            # Define supported programming languages
            self.supported_languages = [
                'python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'c',
                'go', 'rust', 'php', 'ruby', 'swift', 'kotlin', 'scala',
                'html', 'css', 'sql', 'bash', 'powershell', 'r', 'matlab'
            ]
            
            # Define code generation templates
            self.code_templates = {
                'function': 'Create a function that',
                'class': 'Create a class that',
                'script': 'Create a script that',
                'api': 'Create an API endpoint that',
                'algorithm': 'Implement an algorithm that',
                'debug': 'Debug and fix the following code',
                'explain': 'Explain how the following code works',
                'optimize': 'Optimize the following code'
            }
            
            logger.info("Code Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Code Agent: {str(e)}")
            raise
    
    def detect_language(self, task: str) -> str:
        """
        Detect the programming language from the task description.
        
        Args:
            task (str): Task description
            
        Returns:
            str: Detected programming language or 'python' as default
        """
        task_lower = task.lower()
        
        # Check for explicit language mentions
        for lang in self.supported_languages:
            if lang in task_lower:
                logger.info(f"Detected language: {lang}")
                return lang
        
        # Default to Python if no language specified
        logger.info("No specific language detected, defaulting to Python")
        return 'python'
    
    def detect_task_type(self, task: str) -> str:
        """
        Detect the type of coding task from the description.
        
        Args:
            task (str): Task description
            
        Returns:
            str: Task type (function, class, debug, etc.)
        """
        task_lower = task.lower()
        
        # Check for task type keywords
        if any(word in task_lower for word in ['debug', 'fix', 'error']):
            return 'debug'
        elif any(word in task_lower for word in ['explain', 'understand', 'how does']):
            return 'explain'
        elif any(word in task_lower for word in ['optimize', 'improve', 'performance']):
            return 'optimize'
        elif any(word in task_lower for word in ['class', 'object']):
            return 'class'
        elif any(word in task_lower for word in ['api', 'endpoint', 'route']):
            return 'api'
        elif any(word in task_lower for word in ['algorithm', 'sort', 'search']):
            return 'algorithm'
        elif any(word in task_lower for word in ['script', 'program']):
            return 'script'
        else:
            return 'function'  # Default to function
    
    def generate_code(self, task: str, language: str, task_type: str) -> str:
        """
        Generate code using OpenAI GPT based on the task requirements.
        
        Args:
            task (str): Code generation task
            language (str): Programming language
            task_type (str): Type of coding task
            
        Returns:
            str: Generated code with explanations
        """
        try:
            # Create a comprehensive prompt for code generation
            prompt = f"""You are a senior software engineer and coding expert. Your task is to {task}.

**Requirements:**
- Programming Language: {language}
- Task Type: {task_type}
- Write clean, well-documented, and efficient code
- Include comments explaining complex logic
- Follow best practices and conventions for {language}
- Add error handling where appropriate
- Provide a brief explanation of how the code works

**Code Generation Guidelines:**
- Start with a brief description of the solution
- Write the complete, runnable code
- Include proper imports/dependencies
- Add inline comments for clarity
- End with a usage example if applicable
- Mention any assumptions or limitations

Please generate the code now:"""

            # Generate code using OpenAI
            response = self.llm.predict(prompt)
            
            logger.info(f"Code generated successfully for {language} {task_type}")
            return response
            
        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}")
            return f"""// Error generating code: {str(e)}

I apologize, but I encountered an error while generating the code. 
Please check your API configuration and try again.

The error was: {str(e)}"""
    
    def process_task(self, task: str) -> Dict[str, Any]:
        """
        Process a code generation task end-to-end.
        
        Args:
            task (str): The coding task description
            
        Returns:
            Dict[str, Any]: Response with generated code and metadata
        """
        try:
            start_time = datetime.now()
            logger.info(f"Processing code task: {task[:100]}...")
            
            # Analyze the task
            language = self.detect_language(task)
            task_type = self.detect_task_type(task)
            
            # Generate code
            code_content = self.generate_code(task, language, task_type)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = {
                "agent": "code",
                "task": task,
                "code": code_content,
                "language": language,
                "task_type": task_type,
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            logger.info(f"Code task completed in {processing_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Error processing code task: {str(e)}")
            return {
                "agent": "code",
                "task": task,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def validate_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """
        Basic syntax validation for generated code (Python only for now).
        
        Args:
            code (str): Code to validate
            language (str): Programming language
            
        Returns:
            Dict[str, Any]: Validation results
        """
        if language.lower() != 'python':
            return {
                "validated": False,
                "message": f"Syntax validation not available for {language}"
            }
        
        try:
            # Extract Python code blocks
            import ast
            code_blocks = re.findall(r'```python\n(.*?)\n```', code, re.DOTALL)
            
            if not code_blocks:
                # Try to find code without markdown formatting
                code_blocks = [code]
            
            for block in code_blocks:
                ast.parse(block)
            
            return {
                "validated": True,
                "message": "Code syntax is valid",
                "blocks_checked": len(code_blocks)
            }
            
        except SyntaxError as e:
            return {
                "validated": False,
                "message": f"Syntax error: {str(e)}",
                "error_type": "SyntaxError"
            }
        except Exception as e:
            return {
                "validated": False,
                "message": f"Validation error: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about the Code Agent's capabilities.
        
        Returns:
            Dict[str, Any]: Agent capabilities and configuration
        """
        return {
            "agent_type": "code",
            "capabilities": [
                "Code generation in multiple languages",
                "Function and class creation",
                "Algorithm implementation",
                "Code debugging and optimization",
                "API endpoint development",
                "Script writing",
                "Code explanation and documentation"
            ],
            "supported_languages": self.supported_languages,
            "task_types": list(self.code_templates.keys()),
            "tools": ["OpenAI GPT"],
            "model": config.OPENAI_MODEL,
            "enabled": config.CODE_AGENT_ENABLED,
            "features": [
                "Syntax validation (Python)",
                "Language auto-detection",
                "Task type classification",
                "Best practices integration",
                "Error handling"
            ]
        }
    
    def get_language_examples(self) -> Dict[str, str]:
        """
        Get example tasks for different programming languages.
        
        Returns:
            Dict[str, str]: Language examples
        """
        return {
            "python": "Create a function to calculate fibonacci numbers",
            "javascript": "Write a function to validate email addresses",
            "java": "Create a class for a simple calculator",
            "c++": "Implement a binary search algorithm",
            "sql": "Write a query to find top customers by sales",
            "html": "Create a responsive navigation menu",
            "css": "Design a card component with hover effects"
        } 