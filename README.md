# 🤖 Peer Agent Directed API

A sophisticated AI-powered task routing system where a central **Peer Agent** intelligently delegates user tasks to specialized AI agents. Built with **FastAPI**, **LangChain**, **MongoDB**, and **Docker**.

## 🌟 Features

- 🧠 **Intelligent Task Routing**: Keyword-based analysis routes tasks to appropriate AI agents
- 📝 **Content Generation**: Research, blog posts, articles with web search integration
- 💻 **Code Generation**: Multi-language code generation, debugging, and optimization
- 🔍 **Web Search Integration**: Real-time information gathering with DuckDuckGo
- 📊 **Session Management**: Track user interactions across multiple requests
- 🗄️ **MongoDB Logging**: Persistent storage of all interactions and analytics
- 🐳 **Docker Ready**: Fully containerized for easy deployment
- 🧪 **Comprehensive Testing**: 30+ tests ensuring reliability

## 🏗️ Architecture

```
User Request → Peer Agent → Task Analysis → Specialized Agent → Response
                    ↓
               MongoDB Logging
```

### Specialized Agents
- **ContentAgent**: Blog posts, articles, research with web search
- **CodeAgent**: Code generation in 20+ programming languages
- **PeerAgent**: Central router with keyword-based task classification

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.9+**
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **MongoDB** (optional - runs without it)
- **Docker** (optional - for containerized deployment)

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd peer-agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

**Quick Setup (Recommended):**

```bash
# Run the setup script to create configuration files
python setup.py
```

**Manual Setup:**

Create a `.env` file in the project root with your API keys:

```env
# Required - OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=1500

# Optional - MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/peer_agent
MONGODB_DATABASE=peer_agent

# Optional - API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True

# Optional - Agent Configuration
CONTENT_AGENT_ENABLED=True
CODE_AGENT_ENABLED=True
MAX_SEARCH_RESULTS=5

# Optional - Search Engine Configuration
# SerpAPI key for enhanced search results (leave empty to use DuckDuckGo)
# Get one from: https://serpapi.com/
SERPAPI_KEY=your_serpapi_key_here
```

> ⚠️ **Important**: Never commit your `.env` file with real API keys!

## 🧪 Testing

Run the comprehensive test suite to verify everything works:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_peer_agent.py -v    # Core routing tests
python -m pytest -m "unit" -v                  # Unit tests only
python -m pytest -m "not slow" -v              # Exclude slow tests

# Check test coverage
python -m pytest --cov=. --cov-report=html
```

**Expected Result**: Core functionality tests should pass, validating the system works correctly.

## 🖥️ Local Development

### Start the API Server

```bash
# Method 1: Using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Method 2: Using Python module
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Main Endpoint**: http://localhost:8000/v1/agent/execute

### MongoDB Setup (Optional)

If you want persistent logging:

```bash
# Install MongoDB locally, or use Docker:
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

## 🐳 Docker Deployment

### Quick Docker Setup

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Docker Configuration

Create `.env.docker` for container-specific settings:

```env
# Docker Environment Configuration
OPENAI_API_KEY=your_openai_api_key_here
MONGODB_URI=mongodb://mongo:27017/peer_agent
API_HOST=0.0.0.0
API_PORT=8000
```

### Services

- **API**: FastAPI application on port 8000
- **MongoDB**: Database on port 27017 with persistent volume
- **Health Checks**: Automatic service health monitoring

## 📡 API Usage

### Main Endpoint

**POST** `/v1/agent/execute`

```json
{
  "task": "Your task description here",
  "session_id": "optional-session-uuid"
}
```

### Example Requests

#### Content Generation

```bash
curl -X POST "http://localhost:8000/v1/agent/execute" \
     -H "Content-Type: application/json" \
     -d '{
       "task": "Write a blog post about artificial intelligence trends"
     }'
```

#### Code Generation

```bash
curl -X POST "http://localhost:8000/v1/agent/execute" \
     -H "Content-Type: application/json" \
     -d '{
       "task": "Write a Python function to calculate fibonacci numbers"
     }'
```

#### Research Task

```bash
curl -X POST "http://localhost:8000/v1/agent/execute" \
     -H "Content-Type: application/json" \
     -d '{
       "task": "Research latest developments in quantum computing"
     }'
```

### Response Format

```json
{
  "success": true,
  "agent_type": "content",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-01T12:00:00",
  "request_id": "req_123456",
  "result": {
    "agent": "content",
    "task": "Write a blog post about AI",
    "content": "# AI Blog Post\n\nContent here...",
    "processing_time_seconds": 2.5,
    "success": true
  }
}
```

## 🔍 Additional Endpoints

### Health Check
```bash
GET /health
```

### Agent Capabilities
```bash
GET /v1/agent/capabilities
```

### Routing Information
```bash
GET /v1/agent/routing-info
```

### Session History
```bash
GET /v1/agent/sessions/{session_id}/history
```

### Analytics
```bash
GET /v1/agent/stats
```

## 🎯 Task Routing Examples

### Content Tasks (→ ContentAgent)
- "Write a blog post about..."
- "Create an article about..."
- "Research information on..."
- "Generate content about..."

### Code Tasks (→ CodeAgent)  
- "Write a Python function to..."
- "Debug this JavaScript code..."
- "Create a REST API in..."
- "Implement a sorting algorithm..."

### Supported Programming Languages
Python, JavaScript, TypeScript, Java, C++, C#, C, Go, Rust, PHP, Ruby, Swift, Kotlin, Scala, HTML, CSS, SQL, Bash, PowerShell, R, MATLAB

## 🔍 Web Search Integration

**✅ STATUS: FULLY OPERATIONAL with SerpAPI**

The system features intelligent web search integration that automatically detects when tasks require current information and performs searches accordingly.

### 🎯 How Task Routing & Search Works

#### 1. **Task Analysis Flow**
```
User Task → PeerAgent → ContentAgent or CodeAgent
                ↓
        ContentAgent analyzes task for search keywords
                ↓
    🔍 Search Required? → Web Search + Content Generation
    📝 No Search? → Direct Content Generation
```

#### 2. **Search Trigger Keywords**
The ContentAgent automatically triggers web search when tasks contain these keywords:

**Search Keywords**: `research`, `information`, `current`, `latest`, `news`, `facts`, `data`

```bash
# ✅ TRIGGERS SEARCH - These will search the web:
"Research latest developments in AI"
"Find information about State College weather"  
"What's the current status of climate change?"
"Latest news about cryptocurrency"
"Get facts about renewable energy"
"Provide data on population growth"

# ❌ NO SEARCH - These generate content directly:
"Write a creative blog about happiness"
"Create a story about dragons"
"Draft an email template"
"Write a poem about nature"
```

#### 3. **Search Engine Selection (Automatic)**
The system intelligently selects the best available search engine:

1. **✅ SerpAPI** (Premium) - **CURRENTLY ACTIVE**
   - Structured JSON results with rich metadata
   - High-quality, relevant search results
   - Requires API key: `SERPAPI_KEY=your_serpapi_key`

2. **🔄 DuckDuckGo** (Fallback)
   - Free search, no API key required
   - Plain text results
   - Automatically used if SerpAPI unavailable

#### 4. **Search Result Integration**
When search is performed:
- **Results Found**: Web data is integrated into AI-generated content with citations
- **No Results**: AI generates content based on training data (with disclaimer)
- **Sources Added**: Automatic citation section with links

### 🔧 Search Configuration

```env
# Search API Configuration
SERPAPI_KEY=your_serpapi_key_here        # Optional but recommended
MAX_SEARCH_RESULTS=5                     # Number of results to include

# Search Engine Selection (automatic based on available keys)
SEARCH_ENGINE=serpapi                    # serpapi or duckduckgo
```

### 📊 Search vs Non-Search Examples

| Task Type | Example | Search Triggered? | Output Includes |
|-----------|---------|-------------------|-----------------|
| **Research** | "Research quantum computing trends" | ✅ Yes | Current web data + citations |
| **Current Info** | "Latest information about Tesla stock" | ✅ Yes | Real-time data + sources |
| **Creative** | "Write a story about space travel" | ❌ No | AI-generated creative content |
| **Educational** | "Explain machine learning concepts" | ❌ No | AI knowledge without web search |
| **News** | "What's the latest news in tech?" | ✅ Yes | Current news articles + links |

### 🔍 Search Response Format

When search is triggered, responses include:

```json
{
  "result": {
    "content": "Generated content with web search data...",
    "search_performed": true,
    "search_results_count": 5,
    "sources": [
      {
        "title": "Article Title",
        "url": "https://example.com",
        "source": "SerpAPI"
      }
    ]
  }
}
```

### 🛠️ Troubleshooting Search Issues

#### Common Issues:
1. **Search not triggering**: Ensure task contains search keywords
2. **No results found**: SerpAPI/DuckDuckGo may have rate limits
3. **JSON parsing errors**: Check SerpAPI key validity
4. **Fallback to DuckDuckGo**: SerpAPI key missing/invalid

#### Debug Commands:
```bash
# Check search tool selection in logs:
python -m api.main
# Look for: "Search tool: SerpAPI selected" or "Search tool: DuckDuckGo"

# Test search functionality:
curl -X POST "http://localhost:8000/v1/agent/execute" \
     -H "Content-Type: application/json" \
     -d '{"task": "research current AI trends"}'
```

## 📊 Project Structure

```
peer-agent/
├── agents/                 # AI Agent implementations
│   ├── peer_agent.py      # Central task router
│   ├── content_agent.py   # Content generation
│   └── code_agent.py      # Code generation
├── api/                   # FastAPI application
│   └── main.py           # API endpoints
├── models/               # Data models
│   └── schemas.py        # Pydantic schemas
├── database/            # Database integration
│   └── mongo.py         # MongoDB operations
├── tests/               # Test suite
│   ├── test_peer_agent.py
│   ├── test_agents.py
│   ├── test_api.py
│   └── test_integration.py
├── docker-compose.yml   # Docker orchestration
├── Dockerfile          # Container definition
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🐛 Troubleshooting

### Common Issues

**1. OpenAI API Key Issues**
```bash
# Error: "No API key provided"
# Solution: Check your .env file has OPENAI_API_KEY set
```

**2. MongoDB Connection Issues**
```bash
# Error: "Database connection failed" 
# Solution: The app works without MongoDB, but check connection string
```

**3. Port Already in Use**
```bash
# Error: "Port 8000 is already in use"
# Solution: Change port in .env or stop other services
uvicorn api.main:app --port 8001
```

**4. Import Errors**
```bash
# Error: "Module not found"
# Solution: Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

**5. Search Functionality Issues**
```bash
# Error: "SerpAPI module not found"
# Solution: Install optional search dependencies
pip install google-search-results

# Warning: "Falling back to DuckDuckGo"
# This is normal if no SerpAPI key is provided - DuckDuckGo works fine

# Error: "Failed to parse SerpAPI results: Expecting property name..."
# Solution: This is handled automatically - system falls back to DuckDuckGo parsing

# Issue: "Search not triggering when expected"
# Solution: Include search keywords: research, information, current, latest, news, facts, data

# Issue: "OpenAI API error with placeholder key"
# Solution: Replace placeholder OPENAI_API_KEY with actual API key from platform.openai.com
```

### Development Tips

- **API Documentation**: Visit `/docs` for interactive Swagger UI
- **Debug Mode**: Set `API_DEBUG=True` in `.env` for detailed logs
- **Testing**: Run tests before making changes to ensure stability
- **Logs**: Check application logs for detailed error information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `python -m pytest tests/ -v`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain** for AI agent framework
- **FastAPI** for high-performance API framework
- **OpenAI** for language model capabilities
- **MongoDB** for document storage
- **Docker** for containerization

---

**🚀 Ready to delegate your tasks to AI agents? Get started now!** 