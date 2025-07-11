"""
Peer Agent Setup Script

This script helps you set up the Peer Agent project by creating the necessary
configuration files and guiding you through the setup process.

Usage: python setup.py
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file with default configuration."""
    env_content = """# Peer Agent Directed API - Environment Configuration
# Update with your actual values

# =============================================================================
# REQUIRED - OpenAI API Configuration
# =============================================================================
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=1500

# =============================================================================
# OPTIONAL - MongoDB Configuration
# =============================================================================
# MongoDB connection string (app works without MongoDB)
MONGODB_URI=mongodb://localhost:27017/peer_agent
MONGODB_DATABASE=peer_agent

# =============================================================================
# OPTIONAL - API Configuration
# =============================================================================
# API server settings
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True

# =============================================================================
# OPTIONAL - Agent Configuration
# =============================================================================
# Enable/disable specific agents
CONTENT_AGENT_ENABLED=True
CODE_AGENT_ENABLED=True

# Content agent settings
MAX_SEARCH_RESULTS=5

# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================
# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Request timeout (seconds)
REQUEST_TIMEOUT=30
"""
    
    env_path = Path(".env")
    if env_path.exists():
        print("⚠️  .env file already exists!")
        overwrite = input("Do you want to overwrite it? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("✅ Keeping existing .env file")
            return False
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file")
    return True

def create_docker_env_file():
    """Create .env.docker file for Docker deployment."""
    docker_env_content = """# Docker Environment Configuration
# Update with your actual values for Docker deployment

OPENAI_API_KEY=your_openai_api_key_here
MONGODB_URI=mongodb://mongo:27017/peer_agent
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=False
LOG_LEVEL=INFO
"""
    
    docker_env_path = Path(".env.docker")
    if not docker_env_path.exists():
        with open(docker_env_path, 'w') as f:
            f.write(docker_env_content)
        print("✅ Created .env.docker file")
        return True
    
    print("⚠️  .env.docker file already exists")
    return False

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version OK: {sys.version}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import langchain
        import openai
        print("✅ Core dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

def main():
    """Main setup function."""
    print("🤖 Peer Agent Setup Script")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create configuration files
    env_created = create_env_file()
    create_docker_env_file()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    print("\n📋 Next Steps:")
    print("=" * 40)
    
    if env_created:
        print("1. 🔑 Edit .env file and add your OpenAI API key")
        print("   Get one from: https://platform.openai.com/api-keys")
    
    if not deps_ok:
        print("2. 📦 Install dependencies:")
        print("   pip install -r requirements.txt")
    
    print("3. 🧪 Run tests to verify setup:")
    print("   python -m pytest tests/test_peer_agent.py -v")
    
    print("4. 🚀 Start the development server:")
    print("   uvicorn api.main:app --reload")
    
    print("5. 🌐 Visit the API docs:")
    print("   http://localhost:8000/docs")
    
    print("\n🐳 For Docker deployment:")
    print("   docker-compose up --build")
    
    print("\n✅ Setup complete! Check README.md for full instructions.")

if __name__ == "__main__":
    main() 