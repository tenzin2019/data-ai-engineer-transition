"""
Pytest configuration and fixtures for the Intelligent Document Analysis System.
"""

import os
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, Generator
import tempfile
import json
from pathlib import Path

# Add the src directory to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import Settings
from core.ai_analyzer import AIAnalyzer
from utils.ai_utils import chunk_text, estimate_tokens, validate_analysis_results


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with mock Azure credentials."""
    return Settings(
        azure_openai_endpoint="https://test-openai.openai.azure.com/",
        azure_openai_api_key="test-api-key",
        azure_openai_api_version="2023-12-01-preview",
        azure_openai_deployment_name="gpt-4o",
        azure_document_intelligence_endpoint="https://test-doc-intel.cognitiveservices.azure.com/",
        azure_document_intelligence_api_key="test-doc-key",
        azure_storage_account_name="teststorageaccount",
        azure_storage_account_key="test-storage-key",
        azure_storage_container_name="test-documents",
        max_tokens=4000,
        temperature=0.3,
        max_document_length=10000,
        debug=True
    )


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "summary": "This is a test document about artificial intelligence and machine learning.",
                        "key_phrases": ["artificial intelligence", "machine learning", "neural networks"],
                        "entities": [
                            {"text": "OpenAI", "type": "ORGANIZATION", "confidence": 0.9},
                            {"text": "2024", "type": "DATE", "confidence": 0.8}
                        ],
                        "sentiment": {"score": 0.7, "label": "positive"},
                        "topics": ["AI", "technology", "innovation"],
                        "insights": ["Document shows positive outlook on AI"],
                        "recommendations": ["Consider implementing AI solutions"]
                    })
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300
        }
    }


@pytest.fixture
def mock_document_intelligence_response():
    """Mock Azure Document Intelligence response."""
    return {
        "analyzeResult": {
            "apiVersion": "2023-07-31",
            "modelId": "prebuilt-document",
            "content": "This is a sample document content extracted by Document Intelligence.",
            "pages": [
                {
                    "pageNumber": 1,
                    "width": 8.5,
                    "height": 11.0,
                    "unit": "inch",
                    "words": [
                        {
                            "content": "Sample",
                            "confidence": 0.99,
                            "boundingBox": [1, 2, 3, 4, 5, 6, 7, 8]
                        }
                    ]
                }
            ],
            "tables": [],
            "keyValuePairs": [],
            "entities": []
        }
    }


@pytest.fixture
def sample_document_text():
    """Sample document text for testing."""
    return """
    Artificial Intelligence and Machine Learning in Modern Business
    
    Artificial Intelligence (AI) and Machine Learning (ML) are revolutionizing the way businesses operate. 
    These technologies enable organizations to automate processes, gain insights from data, and make 
    better decisions.
    
    Key Benefits:
    1. Automation of repetitive tasks
    2. Enhanced data analysis capabilities
    3. Improved customer experience
    4. Cost reduction and efficiency gains
    
    Companies like OpenAI, Google, and Microsoft are leading the development of advanced AI systems.
    The future of business lies in the successful integration of AI technologies.
    """


@pytest.fixture
def sample_legal_document():
    """Sample legal document for testing."""
    return """
    SOFTWARE LICENSE AGREEMENT
    
    This Software License Agreement ("Agreement") is entered into on January 1, 2024, between 
    TechCorp Inc. ("Licensor") and ClientCompany LLC ("Licensee").
    
    TERMS AND CONDITIONS:
    1. Grant of License: Licensor grants Licensee a non-exclusive license to use the software.
    2. Restrictions: Licensee may not reverse engineer or distribute the software.
    3. Term: This agreement shall remain in effect for 2 years from the effective date.
    4. Termination: Either party may terminate with 30 days written notice.
    
    GOVERNING LAW: This agreement shall be governed by the laws of California.
    """


@pytest.fixture
def sample_financial_document():
    """Sample financial document for testing."""
    return """
    QUARTERLY FINANCIAL REPORT - Q4 2023
    
    REVENUE SUMMARY:
    - Total Revenue: $2.5M (15% increase from Q3)
    - Product Sales: $1.8M
    - Service Revenue: $700K
    
    EXPENSES:
    - Operating Expenses: $1.2M
    - Marketing: $300K
    - R&D: $400K
    
    PROFIT MARGIN: 52% (up from 48% in Q3)
    
    KEY METRICS:
    - Customer Acquisition Cost: $150
    - Customer Lifetime Value: $2,400
    - Monthly Recurring Revenue: $800K
    """


@pytest.fixture
def mock_azure_storage_client():
    """Mock Azure Storage client."""
    mock_client = Mock()
    mock_container = Mock()
    mock_blob = Mock()
    
    # Mock blob operations
    mock_blob.upload_blob.return_value = None
    mock_blob.download_blob.return_value = Mock(readall=lambda: b"test content")
    mock_blob.delete_blob.return_value = None
    mock_blob.exists.return_value = True
    
    # Mock container operations
    mock_container.get_blob_client.return_value = mock_blob
    mock_container.list_blobs.return_value = [
        Mock(name="test-doc1.pdf", size=1024),
        Mock(name="test-doc2.docx", size=2048)
    ]
    
    # Mock client operations
    mock_client.get_container_client.return_value = mock_container
    
    return mock_client


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("This is a test document content.")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def mock_ai_analyzer(test_settings, mock_openai_response):
    """Create a mock AI analyzer with mocked OpenAI client."""
    with patch('core.ai_analyzer.AzureOpenAI') as mock_openai_class, \
         patch('core.ai_analyzer.DocumentIntelligenceClient') as mock_doc_intel_class:
        
        # Mock OpenAI client
        mock_openai_client = Mock()
        mock_openai_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content=json.dumps(mock_openai_response["choices"][0]["message"]["content"])))]
        )
        mock_openai_class.return_value = mock_openai_client
        
        # Mock Document Intelligence client
        mock_doc_intel_client = Mock()
        mock_doc_intel_class.return_value = mock_doc_intel_client
        
        # Create analyzer instance
        analyzer = AIAnalyzer()
        analyzer.openai_client = mock_openai_client
        analyzer.document_intelligence_client = mock_doc_intel_client
        
        yield analyzer


@pytest.fixture
def mock_azure_storage_operations():
    """Mock Azure Storage operations."""
    with patch('azure.storage.blob.BlobServiceClient') as mock_blob_service:
        mock_client = Mock()
        mock_container = Mock()
        mock_blob = Mock()
        
        # Configure mock responses
        mock_blob.upload_blob.return_value = None
        mock_blob.download_blob.return_value = Mock(readall=lambda: b"test content")
        mock_blob.delete_blob.return_value = None
        mock_blob.exists.return_value = True
        
        mock_container.get_blob_client.return_value = mock_blob
        mock_container.list_blobs.return_value = [
            Mock(name="test-doc1.pdf", size=1024),
            Mock(name="test-doc2.docx", size=2048)
        ]
        
        mock_client.get_container_client.return_value = mock_container
        mock_blob_service.from_connection_string.return_value = mock_client
        
        yield {
            'blob_service': mock_blob_service,
            'client': mock_client,
            'container': mock_container,
            'blob': mock_blob
        }


@pytest.fixture
def test_environment_variables():
    """Set up test environment variables."""
    env_vars = {
        'AZURE_OPENAI_ENDPOINT': 'https://test-openai.openai.azure.com/',
        'AZURE_OPENAI_API_KEY': 'test-api-key',
        'AZURE_OPENAI_API_VERSION': '2023-12-01-preview',
        'AZURE_OPENAI_DEPLOYMENT_NAME': 'gpt-4o',
        'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'https://test-doc-intel.cognitiveservices.azure.com/',
        'AZURE_DOCUMENT_INTELLIGENCE_API_KEY': 'test-doc-key',
        'AZURE_STORAGE_ACCOUNT_NAME': 'teststorageaccount',
        'AZURE_STORAGE_ACCOUNT_KEY': 'test-storage-key',
        'AZURE_STORAGE_CONTAINER_NAME': 'test-documents',
        'DEBUG': 'True'
    }
    
    # Store original values
    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield env_vars
    
    # Restore original values
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def mock_http_responses():
    """Mock HTTP responses for testing."""
    return {
        'success': Mock(
            status_code=200,
            json=lambda: {"status": "success", "data": "test data"},
            text="success response"
        ),
        'unauthorized': Mock(
            status_code=401,
            json=lambda: {"error": "Unauthorized"},
            text="unauthorized"
        ),
        'not_found': Mock(
            status_code=404,
            json=lambda: {"error": "Not Found"},
            text="not found"
        ),
        'server_error': Mock(
            status_code=500,
            json=lambda: {"error": "Internal Server Error"},
            text="server error"
        )
    }


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"


@pytest.fixture
def mock_analysis_result():
    """Mock analysis result for testing."""
    return {
        "summary": "This is a comprehensive analysis of the document.",
        "key_phrases": ["artificial intelligence", "machine learning", "business automation"],
        "entities": [
            {"text": "OpenAI", "type": "ORGANIZATION", "confidence": 0.95},
            {"text": "2024", "type": "DATE", "confidence": 0.90}
        ],
        "sentiment": {"score": 0.8, "label": "positive"},
        "topics": ["AI", "technology", "business", "innovation"],
        "insights": ["Document shows strong positive sentiment towards AI adoption"],
        "recommendations": ["Consider implementing AI solutions", "Invest in ML training"],
        "confidence_score": 0.85
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "azure: mark test as requiring Azure services"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test names
        if "azure" in item.name.lower():
            item.add_marker(pytest.mark.azure)
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        if "slow" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)
