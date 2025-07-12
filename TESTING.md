# Testing Strategy for LLM Applications

This document outlines the comprehensive testing strategy implemented for the Peer Agent system, specifically designed to handle the challenges of testing non-deterministic Large Language Model (LLM) applications.

## Testing Philosophy

### Traditional Testing Challenges with LLMs

LLM applications present unique testing challenges:
- **Non-deterministic outputs**: Same input can produce different valid outputs
- **Variable search results**: Web search APIs return different results over time
- **Context-dependent responses**: LLM responses vary based on prompt context
- **Semantic equivalence**: Different text can convey the same meaning

### Our Solution: Behavioral Testing

Instead of exact content matching, we use **behavioral testing** that focuses on:
- **Response structure validation**: Ensuring outputs have correct format
- **Semantic content validation**: Checking for key concepts rather than exact text
- **Functional behavior verification**: Testing that components work as intended
- **Integration flow validation**: Ensuring data flows correctly through the system

## Testing Utilities

### Core Validation Functions

Located in `tests/conftest.py`:

```python
def validate_code_response(response, expected_language=None):
    """Validate code response structure and content patterns"""
    
def validate_content_response(response, expected_topics=None):
    """Validate content response structure and semantic topics"""
    
def validate_search_integration(response):
    """Validate search integration fields and consistency"""
    
def semantic_similarity(text1, text2, threshold=0.3):
    """Check semantic similarity between texts"""
```

### Flexible Mock Fixtures

- **`realistic_llm_mock`**: Returns context-appropriate responses
- **`flexible_search_mock`**: Adapts search results to query content
- **Dynamic response generation**: Mocks that respond intelligently to input

## Test Categories

### 1. Unit Tests (`@pytest.mark.unit`)
Test individual components in isolation:
- Agent initialization
- Task routing logic
- Response formatting
- Error handling

### 2. Integration Tests (`@pytest.mark.integration`)
Test component interactions:
- Agent communication
- Search integration
- Database operations
- API endpoint behavior

### 3. API Tests (`@pytest.mark.api`)
Test HTTP API endpoints:
- Request/response validation
- Error handling
- Authentication
- Rate limiting

### 4. Agent Tests (`@pytest.mark.agent`)
Test LLM agent functionality:
- Task processing
- Content generation
- Code generation
- Search integration

### 5. Search Tests (`@pytest.mark.search`)
Test search functionality:
- Search tool selection
- Result processing
- Error handling
- Performance

### 6. LLM Tests (`@pytest.mark.llm`)
Test LLM-dependent functionality:
- Response quality
- Behavioral consistency
- Error recovery
- Performance

## Testing Patterns

### 1. Behavioral Assertions

Instead of:
```python
assert result["content"] == "Expected exact content"
```

Use:
```python
assert validate_content_response(result, expected_topics=["ai", "technology"])
assert len(result["content"]) > 100  # Substantial content
assert any(concept in result["content"].lower() for concept in ["ai", "artificial"])
```

### 2. Structure Validation

Focus on response structure rather than exact content:
```python
assert result["success"] is True
assert "code" in result
assert result["language"] == "python"
assert bool(re.search(r'def \w+\(', result["code"]))  # Contains function definition
```

### 3. Semantic Content Checking

Check for semantic concepts rather than exact strings:
```python
# Extract key concepts from response
concepts = extract_key_concepts(result["content"])
assert any(concept in ["artificial", "intelligence", "machine"] for concept in concepts)
```

### 4. Flexible Mock Responses

Use dynamic mocks that respond to context:
```python
def dynamic_llm_response(prompt):
    if "python" in prompt.lower():
        return generate_python_code()
    elif "article" in prompt.lower():
        return generate_article()
    return "Generic response"
```

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m api
pytest -m agent

# Run with coverage
pytest --cov=agents --cov=api --cov-report=html
```

### Test Configuration
```bash
# Fast tests only (exclude slow integration tests)
pytest -m "not slow"

# LLM-dependent tests
pytest -m llm

# Search-dependent tests
pytest -m search
```

### Environment Setup
```bash
# Set test environment variables
export OPENAI_API_KEY="test-key"
export MONGODB_URI="mongodb://localhost:27017/test_db"

# Run tests in test environment
pytest --env=test
```

## Test Data Management

### Mock Data Strategy
- **Realistic mock responses**: Use varied, realistic example outputs
- **Response variants**: Multiple valid responses for the same input
- **Context-aware mocks**: Mocks that respond appropriately to input

### Test Data Files
- `conftest.py`: Shared fixtures and utilities
- Mock response variants for different scenarios
- Utility functions for validation

## Performance Testing

### Response Time Validation
```python
def test_response_time_consistency():
    """Test that response times are reasonable and consistent"""
    response_times = []
    for i in range(5):
        start = time.time()
        result = agent.process_task(f"Test task {i}")
        end = time.time()
        response_times.append(end - start)
    
    assert max(response_times) < 5.0  # Max 5 seconds
    assert sum(response_times) / len(response_times) < 2.0  # Avg < 2 seconds
```

### Concurrency Testing
```python
def test_concurrent_requests():
    """Test handling of concurrent requests"""
    import threading
    
    results = []
    def make_request(i):
        response = client.post("/execute", json={"task": f"Task {i}"})
        results.append(response)
    
    threads = [threading.Thread(target=make_request, args=(i,)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    assert all(r.status_code == 200 for r in results)
```

## Error Testing

### Graceful Error Handling
```python
def test_llm_api_error_handling():
    """Test graceful handling of LLM API errors"""
    with patch('openai.ChatCompletion.create') as mock_llm:
        mock_llm.side_effect = Exception("API Error")
        
        result = agent.process_task("Test task")
        
        assert result["success"] is False
        assert "error" in result
        assert isinstance(result["error"], str)
```

### Cascading Error Recovery
```python
def test_error_recovery():
    """Test system recovery from multiple failures"""
    # First call fails, second succeeds
    mock_llm.side_effect = [Exception("Error"), "Success response"]
    
    # First request fails
    result1 = agent.process_task("Task 1")
    assert result1["success"] is False
    
    # Second request succeeds
    result2 = agent.process_task("Task 2")
    assert result2["success"] is True
```

## Best Practices

### 1. Test Behavior, Not Implementation
- Focus on what the system does, not how it does it
- Test outcomes and side effects
- Validate functional requirements

### 2. Use Semantic Validation
- Check for key concepts rather than exact text
- Use similarity thresholds for flexible matching
- Validate response structure and format

### 3. Mock Realistically
- Use varied, realistic mock responses
- Include error scenarios and edge cases
- Make mocks context-aware

### 4. Test Error Scenarios
- Test all failure modes
- Verify graceful error handling
- Test recovery mechanisms

### 5. Performance Considerations
- Test response times
- Validate concurrent request handling
- Monitor resource usage

## Common Testing Antipatterns to Avoid

### ❌ Exact String Matching
```python
# BAD: Brittle and unrealistic
assert result["content"] == "Artificial Intelligence is a technology..."
```

### ❌ Hardcoded Expected Values
```python
# BAD: Non-deterministic systems don't produce identical outputs
assert result["search_results_count"] == 5
```

### ❌ Over-specific Mocks
```python
# BAD: Too specific, doesn't reflect real variability
mock_llm.return_value = "exact expected response"
```

### ✅ Behavioral Validation
```python
# GOOD: Flexible and realistic
assert validate_content_response(result, expected_topics=["ai", "technology"])
assert result["search_results_count"] >= 0
assert len(result["content"]) > 50
```

## Test Maintenance

### Regular Test Review
- Review test failures for patterns
- Update mocks to reflect real-world changes
- Refactor tests for better maintainability

### Test Documentation
- Document test intent and expected behavior
- Explain complex validation logic
- Provide examples of valid responses

### Continuous Improvement
- Monitor test reliability
- Add new test cases for edge cases
- Improve test performance and coverage

## Conclusion

This testing strategy provides a robust framework for testing LLM applications that:
- Handles non-deterministic outputs gracefully
- Focuses on functional behavior rather than exact content
- Provides comprehensive coverage of all system components
- Scales well with system complexity

The key is to test **what the system should do** rather than **exactly what it outputs**, making tests both reliable and maintainable while ensuring comprehensive coverage of the application's functionality. 