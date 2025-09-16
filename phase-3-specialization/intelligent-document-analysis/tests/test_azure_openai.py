"""
Tests for Azure OpenAI connection and functionality.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from openai import AzureOpenAI
from azure.core.exceptions import AzureError

from core.ai_analyzer import AIAnalyzer
from utils.ai_utils import chunk_text, estimate_tokens, validate_analysis_results


class TestAzureOpenAIConnection:
    """Test Azure OpenAI connection and basic functionality."""
    
    def test_openai_client_initialization_success(self, test_settings):
        """Test successful Azure OpenAI client initialization."""
        with patch('core.ai_analyzer.AzureOpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            analyzer = AIAnalyzer()
            
            assert analyzer.openai_client is not None
            mock_openai.assert_called_once_with(
                azure_endpoint=test_settings.azure_openai_endpoint,
                api_key=test_settings.azure_openai_api_key,
                api_version=test_settings.azure_openai_api_version
            )
    
    def test_openai_client_initialization_missing_credentials(self):
        """Test OpenAI client initialization with missing credentials."""
        with patch.dict('os.environ', {}, clear=True):
            analyzer = AIAnalyzer()
            assert analyzer.openai_client is None
    
    def test_openai_client_initialization_invalid_endpoint(self, test_settings):
        """Test OpenAI client initialization with invalid endpoint."""
        with patch('core.ai_analyzer.AzureOpenAI') as mock_openai:
            mock_openai.side_effect = Exception("Invalid endpoint")
            
            with pytest.raises(Exception, match="Invalid endpoint"):
                AIAnalyzer()
    
    @pytest.mark.azure
    def test_openai_connection_health_check(self, mock_ai_analyzer):
        """Test OpenAI connection health check."""
        # Mock successful health check response
        mock_ai_analyzer.openai_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Health check successful"))]
        )
        
        # Test health check
        response = mock_ai_analyzer.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        assert response is not None
        assert response.choices[0].message.content == "Health check successful"
    
    @pytest.mark.azure
    def test_openai_connection_failure(self, test_settings):
        """Test OpenAI connection failure handling."""
        with patch('core.ai_analyzer.AzureOpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("Connection failed")
            mock_openai.return_value = mock_client
            
            analyzer = AIAnalyzer()
            
            with pytest.raises(Exception, match="Connection failed"):
                analyzer.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=10
                )


class TestAzureOpenAIAnalysis:
    """Test Azure OpenAI document analysis functionality."""
    
    def test_analyze_document_success(self, mock_ai_analyzer, sample_document_text):
        """Test successful document analysis."""
        result = mock_ai_analyzer.analyze_document(sample_document_text, "general")
        
        assert isinstance(result, dict)
        assert "summary" in result
        assert "key_phrases" in result
        assert "entities" in result
        assert "sentiment" in result
        assert "topics" in result
        assert "insights" in result
        assert "recommendations" in result
        assert "confidence_score" in result
    
    def test_analyze_document_no_client(self, sample_document_text):
        """Test document analysis without OpenAI client."""
        analyzer = AIAnalyzer()
        analyzer.openai_client = None
        
        with pytest.raises(ValueError, match="Azure OpenAI client not initialized"):
            analyzer.analyze_document(sample_document_text)
    
    def test_analyze_document_empty_text(self, mock_ai_analyzer):
        """Test document analysis with empty text."""
        result = mock_ai_analyzer.analyze_document("", "general")
        
        assert isinstance(result, dict)
        assert result["summary"] == ""
        assert result["key_phrases"] == []
        assert result["entities"] == []
    
    def test_analyze_document_legal_type(self, mock_ai_analyzer, sample_legal_document):
        """Test analysis of legal document type."""
        result = mock_ai_analyzer.analyze_document(sample_legal_document, "legal")
        
        assert isinstance(result, dict)
        assert "summary" in result
        # Legal documents should have entities like dates and organizations
        entities = result.get("entities", [])
        assert any(entity.get("type") == "ORGANIZATION" for entity in entities)
    
    def test_analyze_document_financial_type(self, mock_ai_analyzer, sample_financial_document):
        """Test analysis of financial document type."""
        result = mock_ai_analyzer.analyze_document(sample_financial_document, "financial")
        
        assert isinstance(result, dict)
        assert "summary" in result
        # Financial documents should have monetary values and metrics
        key_phrases = result.get("key_phrases", [])
        assert any("revenue" in phrase.lower() or "profit" in phrase.lower() for phrase in key_phrases)
    
    def test_analyze_document_chunking(self, mock_ai_analyzer):
        """Test document analysis with text chunking."""
        # Create a very long document that needs chunking
        long_text = "This is a test sentence. " * 1000  # ~25,000 characters
        
        with patch.object(mock_ai_analyzer, '_analyze_text_chunk') as mock_chunk:
            mock_chunk.return_value = {
                "summary": "Test summary",
                "key_phrases": ["test"],
                "entities": [],
                "sentiment": {"score": 0.0, "label": "neutral"},
                "topics": ["test"]
            }
            
            result = mock_ai_analyzer.analyze_document(long_text, "general")
            
            # Should call chunk analysis multiple times
            assert mock_chunk.call_count > 1
            assert isinstance(result, dict)
    
    def test_analyze_text_chunk_success(self, mock_ai_analyzer, sample_document_text):
        """Test successful text chunk analysis."""
        result = mock_ai_analyzer._analyze_text_chunk(sample_document_text, "general", "gpt-4o")
        
        assert isinstance(result, dict)
        assert "summary" in result
        assert "key_phrases" in result
        assert "entities" in result
        assert "sentiment" in result
        assert "topics" in result
    
    def test_analyze_text_chunk_api_error(self, mock_ai_analyzer, sample_document_text):
        """Test text chunk analysis with API error."""
        # Mock API error
        mock_ai_analyzer.openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        result = mock_ai_analyzer._analyze_text_chunk(sample_document_text, "general", "gpt-4o")
        
        assert isinstance(result, dict)
        assert "error" in result
        assert result["error"] == "API Error"
    
    def test_create_analysis_prompt(self, mock_ai_analyzer, sample_document_text):
        """Test analysis prompt creation."""
        prompt = mock_ai_analyzer._create_analysis_prompt(sample_document_text, "legal")
        
        assert isinstance(prompt, str)
        assert "legal" in prompt.lower()
        assert "json" in prompt.lower()
        assert sample_document_text[:4000] in prompt
    
    def test_estimate_complexity(self, mock_ai_analyzer):
        """Test document complexity estimation."""
        # Simple document
        simple_text = "This is a simple document."
        complexity = mock_ai_analyzer._estimate_complexity(simple_text, "general")
        assert 0.0 <= complexity <= 1.0
        
        # Complex document
        complex_text = "This is a complex technical document with algorithms, implementation details, and architectural frameworks. It contains legal terms, compliance requirements, and medical terminology."
        complexity = mock_ai_analyzer._estimate_complexity(complex_text, "technical")
        assert complexity > 0.5
    
    def test_combine_chunk_results(self, mock_ai_analyzer):
        """Test combining results from multiple chunks."""
        chunk_results = [
            {
                "summary": "First chunk summary",
                "key_phrases": ["AI", "technology"],
                "entities": [{"text": "OpenAI", "type": "ORGANIZATION", "confidence": 0.9}],
                "sentiment": {"score": 0.5, "label": "positive"},
                "topics": ["technology"]
            },
            {
                "summary": "Second chunk summary",
                "key_phrases": ["machine learning", "automation"],
                "entities": [{"text": "2024", "type": "DATE", "confidence": 0.8}],
                "sentiment": {"score": 0.7, "label": "positive"},
                "topics": ["AI"]
            }
        ]
        
        result = mock_ai_analyzer._combine_chunk_results(chunk_results)
        
        assert isinstance(result, dict)
        assert "summary" in result
        assert len(result["key_phrases"]) >= 2
        assert len(result["entities"]) >= 2
        assert result["sentiment"]["score"] > 0.5  # Average of 0.5 and 0.7


class TestAzureOpenAIModelSelection:
    """Test model selection functionality."""
    
    def test_model_selection_primary(self, mock_ai_analyzer):
        """Test selection of primary model for complex documents."""
        with patch('utils.model_selector.model_selector') as mock_selector:
            mock_selector.select_model.return_value = "gpt-4o"
            
            result = mock_ai_analyzer.analyze_document("Complex technical document", "technical")
            
            mock_selector.select_model.assert_called_once()
            assert isinstance(result, dict)
    
    def test_model_selection_budget(self, mock_ai_analyzer):
        """Test selection of budget model for simple documents."""
        with patch('utils.model_selector.model_selector') as mock_selector:
            mock_selector.select_model.return_value = "gpt-3.5-turbo"
            
            result = mock_ai_analyzer.analyze_document("Simple document", "general")
            
            mock_selector.select_model.assert_called_once()
            assert isinstance(result, dict)


class TestAzureOpenAIErrorHandling:
    """Test error handling in Azure OpenAI operations."""
    
    def test_network_timeout_error(self, mock_ai_analyzer, sample_document_text):
        """Test handling of network timeout errors."""
        mock_ai_analyzer.openai_client.chat.completions.create.side_effect = Exception("Request timeout")
        
        result = mock_ai_analyzer._analyze_text_chunk(sample_document_text, "general", "gpt-4o")
        
        assert "error" in result
        assert result["error"] == "Request timeout"
    
    def test_rate_limit_error(self, mock_ai_analyzer, sample_document_text):
        """Test handling of rate limit errors."""
        from openai import RateLimitError
        
        mock_ai_analyzer.openai_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded", response=None, body=None
        )
        
        result = mock_ai_analyzer._analyze_text_chunk(sample_document_text, "general", "gpt-4o")
        
        assert "error" in result
        assert "rate limit" in result["error"].lower()
    
    def test_invalid_api_key_error(self, mock_ai_analyzer, sample_document_text):
        """Test handling of invalid API key errors."""
        from openai import AuthenticationError
        
        mock_ai_analyzer.openai_client.chat.completions.create.side_effect = AuthenticationError(
            "Invalid API key", response=None, body=None
        )
        
        result = mock_ai_analyzer._analyze_text_chunk(sample_document_text, "general", "gpt-4o")
        
        assert "error" in result
        assert "authentication" in result["error"].lower() or "api key" in result["error"].lower()
    
    def test_quota_exceeded_error(self, mock_ai_analyzer, sample_document_text):
        """Test handling of quota exceeded errors."""
        from openai import APIConnectionError
        
        mock_ai_analyzer.openai_client.chat.completions.create.side_effect = APIConnectionError(
            "Quota exceeded", request=None
        )
        
        result = mock_ai_analyzer._analyze_text_chunk(sample_document_text, "general", "gpt-4o")
        
        assert "error" in result
        assert "quota" in result["error"].lower() or "connection" in result["error"].lower()


class TestAzureOpenAIPerformance:
    """Test performance-related functionality."""
    
    @pytest.mark.slow
    def test_concurrent_analysis_requests(self, mock_ai_analyzer, sample_document_text):
        """Test handling of concurrent analysis requests."""
        import asyncio
        import concurrent.futures
        
        def analyze_document():
            return mock_ai_analyzer.analyze_document(sample_document_text, "general")
        
        # Submit multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(analyze_document) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should complete successfully
        assert len(results) == 5
        for result in results:
            assert isinstance(result, dict)
            assert "summary" in result
    
    def test_large_document_processing(self, mock_ai_analyzer):
        """Test processing of large documents."""
        # Create a large document
        large_text = "This is a test sentence. " * 5000  # ~125,000 characters
        
        with patch.object(mock_ai_analyzer, '_analyze_text_chunk') as mock_chunk:
            mock_chunk.return_value = {
                "summary": "Test summary",
                "key_phrases": ["test"],
                "entities": [],
                "sentiment": {"score": 0.0, "label": "neutral"},
                "topics": ["test"]
            }
            
            result = mock_ai_analyzer.analyze_document(large_text, "general")
            
            # Should process multiple chunks
            assert mock_chunk.call_count > 1
            assert isinstance(result, dict)
    
    def test_memory_usage_optimization(self, mock_ai_analyzer):
        """Test memory usage optimization for large documents."""
        # This test would require memory profiling in a real scenario
        # For now, we'll test that chunking works correctly
        large_text = "Test sentence. " * 10000
        
        chunks = chunk_text(large_text, max_length=1000)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 1000 for chunk in chunks)


class TestAzureOpenAIConfiguration:
    """Test Azure OpenAI configuration and settings."""
    
    def test_model_configuration(self, test_settings):
        """Test model configuration settings."""
        assert test_settings.azure_openai_deployment_name == "gpt-4o"
        assert test_settings.azure_openai_api_version == "2023-12-01-preview"
        assert test_settings.max_tokens == 4000
        assert test_settings.temperature == 0.3
    
    def test_endpoint_configuration(self, test_settings):
        """Test endpoint configuration."""
        assert test_settings.azure_openai_endpoint == "https://test-openai.openai.azure.com/"
        assert test_settings.azure_openai_api_key == "test-api-key"
    
    def test_model_selection_criteria(self, test_settings):
        """Test model selection criteria configuration."""
        assert test_settings.primary_model == "gpt-4o"
        assert test_settings.secondary_model == "gpt-4o-mini"
        assert test_settings.budget_model == "gpt-3.5-turbo"
        assert test_settings.max_tokens_budget_threshold == 2000
    
    def test_complex_document_types(self, test_settings):
        """Test complex document types configuration."""
        expected_types = ["legal", "financial", "technical", "medical"]
        assert test_settings.complex_document_types == expected_types


class TestAzureOpenAIUtilities:
    """Test utility functions for Azure OpenAI operations."""
    
    def test_chunk_text_function(self):
        """Test text chunking utility function."""
        text = "This is a test sentence. " * 100
        chunks = chunk_text(text, max_length=100)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)
    
    def test_estimate_tokens_function(self):
        """Test token estimation utility function."""
        text = "This is a test sentence with some words."
        tokens = estimate_tokens(text, "gpt-4")
        
        assert isinstance(tokens, int)
        assert tokens > 0
    
    def test_validate_analysis_results_function(self, mock_analysis_result):
        """Test analysis results validation function."""
        validated = validate_analysis_results(mock_analysis_result)
        
        assert isinstance(validated, dict)
        assert "confidence_score" in validated
        assert 0.0 <= validated["confidence_score"] <= 1.0
    
    def test_validate_analysis_results_invalid_input(self):
        """Test analysis results validation with invalid input."""
        invalid_result = {
            "summary": 123,  # Should be string
            "key_phrases": "not a list",  # Should be list
            "entities": "invalid",  # Should be list
            "sentiment": "not a dict"  # Should be dict
        }
        
        validated = validate_analysis_results(invalid_result)
        
        assert isinstance(validated, dict)
        assert validated["summary"] == ""  # Should be cleaned
        assert validated["key_phrases"] == []  # Should be empty list
        assert validated["entities"] == []  # Should be empty list
        assert validated["sentiment"]["label"] == "neutral"  # Should be default


@pytest.mark.integration
class TestAzureOpenAIIntegration:
    """Integration tests for Azure OpenAI functionality."""
    
    @pytest.mark.azure
    def test_end_to_end_document_analysis(self, mock_ai_analyzer, sample_document_text):
        """Test end-to-end document analysis workflow."""
        # Test the complete analysis workflow
        result = mock_ai_analyzer.analyze_document(sample_document_text, "general")
        
        # Verify all expected fields are present
        expected_fields = [
            "summary", "key_phrases", "entities", "sentiment", 
            "topics", "insights", "recommendations", "confidence_score"
        ]
        
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"
        
        # Verify data types
        assert isinstance(result["summary"], str)
        assert isinstance(result["key_phrases"], list)
        assert isinstance(result["entities"], list)
        assert isinstance(result["sentiment"], dict)
        assert isinstance(result["topics"], list)
        assert isinstance(result["insights"], list)
        assert isinstance(result["recommendations"], list)
        assert isinstance(result["confidence_score"], float)
    
    @pytest.mark.azure
    def test_different_document_types_analysis(self, mock_ai_analyzer):
        """Test analysis of different document types."""
        document_types = ["legal", "financial", "technical", "medical", "business"]
        
        for doc_type in document_types:
            result = mock_ai_analyzer.analyze_document(f"Sample {doc_type} document", doc_type)
            
            assert isinstance(result, dict)
            assert "summary" in result
            assert "confidence_score" in result
    
    @pytest.mark.azure
    def test_batch_document_processing(self, mock_ai_analyzer):
        """Test batch processing of multiple documents."""
        documents = [
            ("Legal contract", "legal"),
            ("Financial report", "financial"),
            ("Technical specification", "technical"),
            ("Medical record", "medical")
        ]
        
        results = []
        for text, doc_type in documents:
            result = mock_ai_analyzer.analyze_document(text, doc_type)
            results.append(result)
        
        assert len(results) == len(documents)
        for result in results:
            assert isinstance(result, dict)
            assert "summary" in result
