"""
Test configuration and fixtures for RAG Conversational AI Assistant
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from typing import AsyncGenerator, Generator
import uuid

# Test imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.document_processor import DocumentProcessor, DocumentChunk
from core.vector_store import VectorStore
from core.rag_engine import RAGEngine
from orchestration.llm_orchestrator import LLMOrchestrator, OpenAIProvider, AnthropicProvider
from utils.text_utils import TextPreprocessor, TextSplitter


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return """
    This is a sample document for testing the RAG conversational AI assistant.
    It contains multiple paragraphs and various types of content.
    
    The system should be able to process this text, chunk it appropriately,
    and create embeddings for semantic search.
    
    This document discusses artificial intelligence, machine learning,
    and natural language processing concepts that are relevant for testing.
    
    The RAG system combines retrieval and generation to provide accurate
    answers based on the document content.
    """


@pytest.fixture
def sample_chunks(sample_text):
    """Sample document chunks for testing"""
    chunks = []
    text_splitter = TextSplitter(chunk_size=200, chunk_overlap=50)
    text_chunks = text_splitter.split_text(sample_text)
    
    for i, chunk_text in enumerate(text_chunks):
        metadata = {
            'document_id': 'test_doc_123',
            'filename': 'test_document.txt',
            'chunk_number': i,
            'total_chunks': len(text_chunks),
            'chunk_size': len(chunk_text)
        }
        chunks.append(DocumentChunk(chunk_text, metadata))
    
    return chunks


@pytest.fixture
def sample_legal_document():
    """Sample legal document for testing"""
    return """
    TERMS AND CONDITIONS OF SERVICE
    
    1. ACCEPTANCE OF TERMS
    By accessing and using this service, you accept and agree to be bound by the terms
    and provision of this agreement.
    
    2. LIABILITY
    The company shall not be liable for any direct, indirect, incidental, special,
    consequential or exemplary damages resulting from the use of this service.
    
    3. TERMINATION
    This agreement may be terminated by either party without notice at any time
    for any reason whatsoever.
    
    4. GOVERNING LAW
    This agreement shall be governed by and construed in accordance with the laws
    of the State of California.
    """


@pytest.fixture
def sample_financial_document():
    """Sample financial document for testing"""
    return """
    QUARTERLY EARNINGS REPORT - Q3 2024
    
    EXECUTIVE SUMMARY
    Revenue for Q3 2024 increased by 15% year-over-year to $2.5 billion.
    Net income rose to $450 million, representing an 18% increase.
    
    FINANCIAL HIGHLIGHTS
    - Total Revenue: $2.5B (+15% YoY)
    - Gross Profit: $1.2B (+12% YoY)
    - Operating Income: $600M (+20% YoY)
    - Net Income: $450M (+18% YoY)
    - Earnings Per Share: $2.25 (+16% YoY)
    
    BUSINESS OUTLOOK
    We expect continued growth in Q4 2024, with projected revenue
    between $2.6B and $2.8B.
    """


@pytest.fixture
async def temp_directory():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
async def test_document_processor():
    """Create a test document processor"""
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    yield processor
    await processor.cleanup()


@pytest.fixture
async def test_vector_store(temp_directory):
    """Create a test vector store"""
    persist_dir = os.path.join(temp_directory, "test_chroma_db")
    vector_store = VectorStore(persist_directory=persist_dir)
    yield vector_store
    # Cleanup
    vector_store.reset_collection()


@pytest.fixture
def mock_llm_orchestrator():
    """Create a mock LLM orchestrator for testing"""
    class MockLLMOrchestrator:
        def __init__(self):
            self.providers = []
        
        async def generate_response(self, prompt: str, max_tokens: int = 500, 
                                  temperature: float = 0.7, conversation_id: str = None,
                                  **kwargs) -> str:
            return f"Mock response to: {prompt[:50]}..."
        
        async def get_provider_status(self):
            return {
                'total_providers': 0,
                'available_providers': 0,
                'providers': []
            }
        
        async def reset_provider_errors(self, provider_name: str = None):
            pass
    
    return MockLLMOrchestrator()


@pytest.fixture
async def test_rag_engine(test_vector_store, mock_llm_orchestrator):
    """Create a test RAG engine"""
    engine = RAGEngine(
        vector_store=test_vector_store,
        llm_orchestrator=mock_llm_orchestrator
    )
    yield engine


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    class MockResponse:
        def __init__(self):
            self.choices = [MockChoice()]
    
    class MockChoice:
        def __init__(self):
            self.message = MockMessage()
    
    class MockMessage:
        def __init__(self):
            self.content = "This is a mock response from OpenAI API for testing purposes."
    
    return MockResponse()


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response"""
    class MockResponse:
        def __init__(self):
            self.content = [MockContent()]
    
    class MockContent:
        def __init__(self):
            self.text = "This is a mock response from Anthropic API for testing purposes."
    
    return MockResponse()


@pytest.fixture
async def sample_pdf_file(temp_directory):
    """Create a sample PDF file for testing"""
    # This would create a sample PDF file in a real implementation
    pdf_path = os.path.join(temp_directory, "sample.pdf")
    
    # For testing, we'll create a dummy file
    with open(pdf_path, "w") as f:
        f.write("This is a mock PDF file content for testing.")
    
    yield pdf_path


@pytest.fixture
async def sample_text_file(temp_directory, sample_text):
    """Create a sample text file for testing"""
    text_path = os.path.join(temp_directory, "sample.txt")
    
    with open(text_path, "w") as f:
        f.write(sample_text)
    
    yield text_path


@pytest.fixture
def test_conversation_id():
    """Generate a test conversation ID"""
    return str(uuid.uuid4())


@pytest.fixture
def text_preprocessor():
    """Create a text preprocessor for testing"""
    return TextPreprocessor()


@pytest.fixture
def text_splitter():
    """Create a text splitter for testing"""
    return TextSplitter(chunk_size=1000, chunk_overlap=200)


# Test data fixtures
@pytest.fixture
def sample_queries():
    """Sample queries for testing"""
    return [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain natural language processing",
        "What are the benefits of RAG systems?",
        "How do embeddings work in vector databases?"
    ]


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history for testing"""
    return [
        {
            "timestamp": "2024-01-01T10:00:00",
            "query": "What is AI?",
            "response": "AI is artificial intelligence...",
            "sources": []
        },
        {
            "timestamp": "2024-01-01T10:01:00",
            "query": "How does it work?",
            "response": "AI works by processing data...",
            "sources": []
        }
    ]


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing"""
    test_env = {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-azure-key",
        "DATABASE_URL": "sqlite:///test.db",
        "REDIS_URL": "redis://localhost:6379/1"
    }
    
    # Store original values
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_env
    
    # Restore original values
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# Async test helpers
@pytest.fixture
def async_test_timeout():
    """Timeout for async tests"""
    return 30  # seconds


# Performance testing fixtures
@pytest.fixture
def performance_test_data():
    """Large dataset for performance testing"""
    return {
        "large_text": "This is a large text document. " * 1000,
        "multiple_documents": [f"Document {i} content" for i in range(100)],
        "complex_queries": [
            "What are the implications of artificial intelligence on society?",
            "How do large language models impact natural language processing?",
            "What are the ethical considerations of AI development?",
            "How does retrieval-augmented generation improve AI responses?",
            "What are the challenges in scaling AI systems?"
        ]
    }


# Configuration for different test environments
@pytest.fixture(params=["development", "testing", "production"])
def test_environment(request):
    """Test different environment configurations"""
    return request.param


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add unit marker to tests in unit test directories
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration test directories
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that might be slow
        if "performance" in str(item.fspath) or "slow" in item.name:
            item.add_marker(pytest.mark.slow)
