# Copy this content to your .env.docker file

# OpenAI Configuration (REQUIRED - copy from your existing .env)
OPENAI_API_KEY=your_actual_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=1500
OPENAI_TEMPERATURE=0.7

# Search API Configuration (OPTIONAL - copy from your existing .env if you have it)
SERPAPI_KEY=your_serpapi_key_here
SEARCH_ENGINE=serpapi

# MongoDB Configuration for Docker (these are the Docker container settings)
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=adminpassword
MONGODB_DATABASE=peer_agent_db
MONGODB_COLLECTION=interactions

# FastAPI Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=False

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Agent Configuration
MAX_SEARCH_RESULTS=5
CONTENT_AGENT_ENABLED=True
CODE_AGENT_ENABLED=True

# Note: MONGODB_URI is automatically configured in docker-compose.yml
# It will be: mongodb://admin:adminpassword@mongo:27017/peer_agent_db?authSource=admin 