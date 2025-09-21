# Azure OpenAI and Storage Connection Tests

This directory contains comprehensive pytest tests for testing Azure OpenAI and Azure Storage connections in the Intelligent Document Analysis System.

## 🧪 Test Structure

### Test Files

- **`conftest.py`** - Pytest configuration and shared fixtures
- **`test_azure_openai.py`** - Azure OpenAI connection and functionality tests
- **`test_azure_storage.py`** - Azure Storage connection and functionality tests
- **`test_azure_integration.py`** - Integration tests for both Azure services
- **`test_document_processor.py`** - Document processing tests (existing)
- **`test_runner.py`** - Test runner script with various options

### Test Categories

#### 🔗 Connection Tests
- Azure OpenAI client initialization
- Azure Storage client initialization
- Connection health checks
- Authentication validation
- Error handling for connection failures

#### 🔧 Functionality Tests
- Document analysis with Azure OpenAI
- File upload/download with Azure Storage
- Model selection and configuration
- Text chunking and processing
- Error handling and recovery

#### 🔄 Integration Tests
- End-to-end document processing workflows
- Batch processing operations
- Concurrent operations
- Performance testing
- Security and monitoring

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# Azure Storage Configuration
AZURE_STORAGE_ACCOUNT_NAME=yourstorageaccount
AZURE_STORAGE_ACCOUNT_KEY=your-storage-key
AZURE_STORAGE_CONTAINER_NAME=documents

# Azure Document Intelligence Configuration
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=your-doc-intel-key

# Test Configuration
DEBUG=True
```

### 3. Run Tests

#### Using the Test Runner (Recommended)

```bash
# Run all tests
python tests/test_runner.py

# Run only Azure OpenAI tests
python tests/test_runner.py --type openai

# Run only Azure Storage tests
python tests/test_runner.py --type storage

# Run only integration tests
python tests/test_runner.py --type integration

# Run with coverage
python tests/test_runner.py --coverage

# Run only mock tests (no Azure services required)
python tests/test_runner.py --mock-only

# Check environment variables
python tests/test_runner.py --check-env
```

#### Using Pytest Directly

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_azure_openai.py

# Run with markers
pytest -m "azure and not slow"

# Run with coverage
pytest --cov=src --cov-report=html

# Run in parallel
pytest -n auto
```

## 📋 Test Markers

### Available Markers

- **`@pytest.mark.azure`** - Tests requiring Azure services
- **`@pytest.mark.integration`** - Integration tests
- **`@pytest.mark.slow`** - Slow-running tests
- **`@pytest.mark.unit`** - Unit tests
- **`@pytest.mark.mock`** - Tests using mocks only

### Using Markers

```bash
# Run only Azure tests
pytest -m azure

# Run integration tests
pytest -m integration

# Run fast tests only
pytest -m "not slow"

# Run mock tests only
pytest -m "not azure"
```

## 🔧 Test Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | Yes |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Yes |
| `AZURE_OPENAI_API_VERSION` | API version | No (default: 2023-12-01-preview) |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Deployment name | No (default: gpt-4o) |
| `AZURE_STORAGE_ACCOUNT_NAME` | Storage account name | Yes |
| `AZURE_STORAGE_ACCOUNT_KEY` | Storage account key | Yes |
| `AZURE_STORAGE_CONTAINER_NAME` | Container name | No (default: documents) |
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | Document Intelligence endpoint | No |
| `AZURE_DOCUMENT_INTELLIGENCE_API_KEY` | Document Intelligence key | No |

### Test Settings

The tests use the `test_settings` fixture which provides mock Azure credentials for testing. Real Azure credentials are only required for integration tests marked with `@pytest.mark.azure`.

## 📊 Test Coverage

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Generate terminal coverage report
pytest --cov=src --cov-report=term

# Generate XML coverage report
pytest --cov=src --cov-report=xml
```

### Coverage Targets

- **Overall Coverage**: > 80%
- **Azure OpenAI Module**: > 90%
- **Azure Storage Module**: > 90%
- **Integration Tests**: > 85%

## 🐛 Debugging Tests

### Verbose Output

```bash
pytest -v
```

### Debug Mode

```bash
pytest --pdb
```

### Logging

```bash
pytest --log-cli-level=DEBUG
```

### Specific Test

```bash
pytest tests/test_azure_openai.py::TestAzureOpenAIConnection::test_openai_client_initialization_success -v
```

## 🔒 Security Testing

### Bandit Security Scanner

```bash
bandit -r src/
```

### Safety Dependency Check

```bash
safety check
```

## 📈 Performance Testing

### Benchmark Tests

```bash
pytest --benchmark-only
```

### Memory Profiling

```bash
pytest --profile
```

## 🚨 Common Issues

### 1. Import Errors

**Problem**: `ModuleNotFoundError` when running tests

**Solution**: Ensure the `src` directory is in the Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

### 2. Azure Authentication Errors

**Problem**: `AuthenticationError` when running Azure tests

**Solution**: 
- Check environment variables are set correctly
- Verify Azure credentials are valid
- Use `--mock-only` flag for testing without Azure services

### 3. Test Timeouts

**Problem**: Tests timing out

**Solution**:
- Use `@pytest.mark.slow` for long-running tests
- Run with `-m "not slow"` to skip slow tests
- Increase timeout in test configuration

### 4. Mock Issues

**Problem**: Mocks not working as expected

**Solution**:
- Check mock patches are applied correctly
- Verify mock return values match expected types
- Use `pytest-mock` for better mock management

## 📝 Writing New Tests

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

class TestNewFeature:
    """Test cases for new feature."""
    
    def test_feature_success(self, mock_fixture):
        """Test successful feature execution."""
        # Arrange
        input_data = "test data"
        expected_result = "expected result"
        
        # Act
        result = feature_function(input_data)
        
        # Assert
        assert result == expected_result
    
    @pytest.mark.azure
    def test_feature_with_azure(self, mock_azure_client):
        """Test feature with Azure service."""
        # Test implementation
        pass
```

### Best Practices

1. **Use descriptive test names** that explain what is being tested
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Use appropriate markers** for test categorization
4. **Mock external dependencies** to ensure test isolation
5. **Test both success and failure scenarios**
6. **Use fixtures** for common test setup
7. **Keep tests focused** on a single behavior

## 🔄 Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      run: |
        python tests/test_runner.py --mock-only --coverage
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Azure OpenAI Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)
- [Azure Storage Documentation](https://docs.microsoft.com/en-us/azure/storage/)
- [Python Testing Best Practices](https://docs.python.org/3/library/unittest.html)
