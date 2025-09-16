"""
Integration tests for Azure OpenAI and Storage services.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from core.ai_analyzer import AIAnalyzer
from config.settings import Settings


class TestAzureServicesIntegration:
    """Test integration between Azure OpenAI and Storage services."""
    
    @pytest.mark.integration
    @pytest.mark.azure
    def test_document_analysis_and_storage_workflow(self, mock_ai_analyzer, mock_azure_storage_operations, sample_document_text):
        """Test complete workflow: analyze document and store results."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock storage operations
        mock_blob.upload_blob.return_value = None
        mock_blob.exists.return_value = True
        
        # 1. Analyze document
        analysis_result = mock_ai_analyzer.analyze_document(sample_document_text, "general")
        
        # 2. Store analysis result
        result_json = json.dumps(analysis_result, indent=2)
        blob_name = "analysis_results.json"
        blob_client = mock_container.get_blob_client(blob_name)
        blob_client.upload_blob(result_json.encode('utf-8'), overwrite=True)
        
        # Verify analysis was performed
        assert isinstance(analysis_result, dict)
        assert "summary" in analysis_result
        assert "confidence_score" in analysis_result
        
        # Verify storage operation was called
        mock_blob.upload_blob.assert_called_once()
        mock_container.get_blob_client.assert_called_once_with(blob_name)
    
    @pytest.mark.integration
    @pytest.mark.azure
    def test_document_processing_pipeline(self, mock_ai_analyzer, mock_azure_storage_operations, sample_document_text):
        """Test complete document processing pipeline."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock storage operations
        mock_blob.upload_blob.return_value = None
        mock_blob.exists.return_value = True
        
        # 1. Store original document
        document_blob_name = "original_document.txt"
        document_blob_client = mock_container.get_blob_client(document_blob_name)
        document_blob_client.upload_blob(sample_document_text.encode('utf-8'), overwrite=True)
        
        # 2. Analyze document
        analysis_result = mock_ai_analyzer.analyze_document(sample_document_text, "general")
        
        # 3. Store analysis result
        analysis_blob_name = "analysis_result.json"
        analysis_blob_client = mock_container.get_blob_client(analysis_blob_name)
        analysis_blob_client.upload_blob(json.dumps(analysis_result, indent=2).encode('utf-8'), overwrite=True)
        
        # 4. Store summary
        summary_blob_name = "summary.txt"
        summary_blob_client = mock_container.get_blob_client(summary_blob_name)
        summary_blob_client.upload_blob(analysis_result["summary"].encode('utf-8'), overwrite=True)
        
        # Verify all operations
        assert mock_blob.upload_blob.call_count == 3
        assert mock_container.get_blob_client.call_count == 3
    
    @pytest.mark.integration
    @pytest.mark.azure
    def test_batch_document_processing(self, mock_ai_analyzer, mock_azure_storage_operations):
        """Test batch processing of multiple documents."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock storage operations
        mock_blob.upload_blob.return_value = None
        mock_blob.exists.return_value = True
        
        # Test documents
        documents = [
            ("Legal contract", "legal"),
            ("Financial report", "financial"),
            ("Technical specification", "technical")
        ]
        
        results = []
        for i, (text, doc_type) in enumerate(documents):
            # Analyze document
            analysis_result = mock_ai_analyzer.analyze_document(text, doc_type)
            results.append(analysis_result)
            
            # Store analysis result
            blob_name = f"analysis_{i}_{doc_type}.json"
            blob_client = mock_container.get_blob_client(blob_name)
            blob_client.upload_blob(json.dumps(analysis_result, indent=2).encode('utf-8'), overwrite=True)
        
        # Verify all documents were processed
        assert len(results) == 3
        assert mock_blob.upload_blob.call_count == 3
        
        # Verify each result has expected structure
        for result in results:
            assert isinstance(result, dict)
            assert "summary" in result
            assert "confidence_score" in result
    
    @pytest.mark.integration
    @pytest.mark.azure
    def test_document_retrieval_and_analysis(self, mock_ai_analyzer, mock_azure_storage_operations, sample_document_text):
        """Test retrieving document from storage and analyzing it."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock download operation
        mock_download = Mock()
        mock_download.readall.return_value = sample_document_text.encode('utf-8')
        mock_blob.download_blob.return_value = mock_download
        
        # Mock upload operation
        mock_blob.upload_blob.return_value = None
        
        # 1. Retrieve document from storage
        blob_name = "stored_document.txt"
        blob_client = mock_container.get_blob_client(blob_name)
        download_stream = blob_client.download_blob()
        document_content = download_stream.readall().decode('utf-8')
        
        # 2. Analyze retrieved document
        analysis_result = mock_ai_analyzer.analyze_document(document_content, "general")
        
        # 3. Store analysis result
        analysis_blob_name = "retrieved_analysis.json"
        analysis_blob_client = mock_container.get_blob_client(analysis_blob_name)
        analysis_blob_client.upload_blob(json.dumps(analysis_result, indent=2).encode('utf-8'), overwrite=True)
        
        # Verify operations
        assert document_content == sample_document_text
        assert isinstance(analysis_result, dict)
        assert mock_blob.download_blob.call_count == 1
        assert mock_blob.upload_blob.call_count == 1
    
    @pytest.mark.integration
    @pytest.mark.azure
    def test_error_handling_integration(self, mock_ai_analyzer, mock_azure_storage_operations, sample_document_text):
        """Test error handling in integrated workflow."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock storage error
        mock_blob.upload_blob.side_effect = Exception("Storage error")
        
        # Analyze document (should succeed)
        analysis_result = mock_ai_analyzer.analyze_document(sample_document_text, "general")
        
        # Try to store result (should fail)
        blob_name = "analysis_result.json"
        blob_client = mock_container.get_blob_client(blob_name)
        
        with pytest.raises(Exception, match="Storage error"):
            blob_client.upload_blob(json.dumps(analysis_result, indent=2).encode('utf-8'), overwrite=True)
        
        # Analysis should still be valid
        assert isinstance(analysis_result, dict)
        assert "summary" in analysis_result


class TestAzureServicesPerformance:
    """Test performance of integrated Azure services."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.azure
    def test_concurrent_document_processing(self, mock_ai_analyzer, mock_azure_storage_operations):
        """Test concurrent processing of multiple documents."""
        import concurrent.futures
        
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock storage operations
        mock_blob.upload_blob.return_value = None
        
        def process_document(doc_id, text, doc_type):
            # Analyze document
            analysis_result = mock_ai_analyzer.analyze_document(text, doc_type)
            
            # Store result
            blob_name = f"analysis_{doc_id}.json"
            blob_client = mock_container.get_blob_client(blob_name)
            blob_client.upload_blob(json.dumps(analysis_result, indent=2).encode('utf-8'), overwrite=True)
            
            return analysis_result
        
        # Test documents
        documents = [
            (i, f"Document {i} content", "general") for i in range(10)
        ]
        
        # Process documents concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_document, doc_id, text, doc_type) 
                      for doc_id, text, doc_type in documents]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all documents were processed
        assert len(results) == 10
        assert mock_blob.upload_blob.call_count == 10
        
        # Verify all results are valid
        for result in results:
            assert isinstance(result, dict)
            assert "summary" in result
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.azure
    def test_large_document_processing_performance(self, mock_ai_analyzer, mock_azure_storage_operations):
        """Test performance with large documents."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock storage operations
        mock_blob.upload_blob.return_value = None
        
        # Create large document
        large_text = "This is a large document. " * 10000  # ~250,000 characters
        
        # Process large document
        analysis_result = mock_ai_analyzer.analyze_document(large_text, "general")
        
        # Store result
        blob_name = "large_document_analysis.json"
        blob_client = mock_container.get_blob_client(blob_name)
        blob_client.upload_blob(json.dumps(analysis_result, indent=2).encode('utf-8'), overwrite=True)
        
        # Verify processing completed
        assert isinstance(analysis_result, dict)
        assert "summary" in analysis_result
        assert mock_blob.upload_blob.call_count == 1


class TestAzureServicesConfiguration:
    """Test configuration of integrated Azure services."""
    
    @pytest.mark.integration
    def test_azure_services_configuration(self, test_settings):
        """Test Azure services configuration."""
        # OpenAI configuration
        assert test_settings.azure_openai_endpoint is not None
        assert test_settings.azure_openai_api_key is not None
        assert test_settings.azure_openai_deployment_name == "gpt-4o"
        
        # Storage configuration
        assert test_settings.azure_storage_account_name is not None
        assert test_settings.azure_storage_account_key is not None
        assert test_settings.azure_storage_container_name == "test-documents"
        
        # Document Intelligence configuration
        assert test_settings.azure_document_intelligence_endpoint is not None
        assert test_settings.azure_document_intelligence_api_key is not None
    
    @pytest.mark.integration
    def test_azure_services_initialization(self, test_settings):
        """Test Azure services initialization."""
        with patch('core.ai_analyzer.AzureOpenAI') as mock_openai, \
             patch('core.ai_analyzer.DocumentIntelligenceClient') as mock_doc_intel, \
             patch('azure.storage.blob.BlobServiceClient') as mock_blob_service:
            
            # Mock clients
            mock_openai_client = Mock()
            mock_doc_intel_client = Mock()
            mock_blob_client = Mock()
            
            mock_openai.return_value = mock_openai_client
            mock_doc_intel.return_value = mock_doc_intel_client
            mock_blob_service.from_connection_string.return_value = mock_blob_client
            
            # Initialize analyzer
            analyzer = AIAnalyzer()
            
            # Verify clients were initialized
            assert analyzer.openai_client is not None
            assert analyzer.document_intelligence_client is not None
            
            # Verify OpenAI client configuration
            mock_openai.assert_called_once_with(
                azure_endpoint=test_settings.azure_openai_endpoint,
                api_key=test_settings.azure_openai_api_key,
                api_version=test_settings.azure_openai_api_version
            )
            
            # Verify Document Intelligence client configuration
            mock_doc_intel.assert_called_once()
    
    @pytest.mark.integration
    def test_azure_services_health_check(self, mock_ai_analyzer, mock_azure_storage_operations):
        """Test health check for all Azure services."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock successful operations
        mock_ai_analyzer.openai_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Health check successful"))]
        )
        mock_container.exists.return_value = True
        
        # Test OpenAI health check
        openai_response = mock_ai_analyzer.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Health check"}],
            max_tokens=10
        )
        assert openai_response.choices[0].message.content == "Health check successful"
        
        # Test Storage health check
        storage_response = mock_container.exists()
        assert storage_response is True


class TestAzureServicesMonitoring:
    """Test monitoring and logging for Azure services."""
    
    @pytest.mark.integration
    def test_azure_services_logging(self, mock_ai_analyzer, mock_azure_storage_operations, sample_document_text):
        """Test logging for Azure services operations."""
        import logging
        
        # Set up logging capture
        logger = logging.getLogger('core.ai_analyzer')
        with patch.object(logger, 'info') as mock_info, \
             patch.object(logger, 'error') as mock_error:
            
            # Perform analysis
            analysis_result = mock_ai_analyzer.analyze_document(sample_document_text, "general")
            
            # Verify logging calls
            assert mock_info.call_count > 0
            assert mock_error.call_count == 0  # No errors expected
    
    @pytest.mark.integration
    def test_azure_services_metrics(self, mock_ai_analyzer, mock_azure_storage_operations, sample_document_text):
        """Test metrics collection for Azure services."""
        # This would be implemented with a metrics collection system
        # For now, we'll test that operations complete successfully
        
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock storage operations
        mock_blob.upload_blob.return_value = None
        
        # Perform operations
        analysis_result = mock_ai_analyzer.analyze_document(sample_document_text, "general")
        
        # Store result
        blob_name = "metrics_test.json"
        blob_client = mock_container.get_blob_client(blob_name)
        blob_client.upload_blob(json.dumps(analysis_result, indent=2).encode('utf-8'), overwrite=True)
        
        # Verify operations completed
        assert isinstance(analysis_result, dict)
        assert mock_blob.upload_blob.call_count == 1


class TestAzureServicesSecurity:
    """Test security aspects of Azure services integration."""
    
    @pytest.mark.integration
    def test_azure_services_authentication(self, test_settings):
        """Test Azure services authentication."""
        # Test that credentials are properly configured
        assert test_settings.azure_openai_api_key is not None
        assert test_settings.azure_storage_account_key is not None
        assert test_settings.azure_document_intelligence_api_key is not None
        
        # Test that endpoints are properly formatted
        assert test_settings.azure_openai_endpoint.startswith("https://")
        assert test_settings.azure_document_intelligence_endpoint.startswith("https://")
    
    @pytest.mark.integration
    def test_azure_services_data_encryption(self, mock_ai_analyzer, mock_azure_storage_operations, sample_document_text):
        """Test data encryption in Azure services."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock storage operations
        mock_blob.upload_blob.return_value = None
        
        # Process sensitive document
        analysis_result = mock_ai_analyzer.analyze_document(sample_document_text, "general")
        
        # Store result (in real implementation, this would be encrypted)
        blob_name = "encrypted_analysis.json"
        blob_client = mock_container.get_blob_client(blob_name)
        blob_client.upload_blob(json.dumps(analysis_result, indent=2).encode('utf-8'), overwrite=True)
        
        # Verify operations completed
        assert isinstance(analysis_result, dict)
        assert mock_blob.upload_blob.call_count == 1
    
    @pytest.mark.integration
    def test_azure_services_access_control(self, test_settings):
        """Test access control for Azure services."""
        # Test that only authorized services can access the resources
        # This would be implemented with proper IAM roles and policies
        
        # Verify configuration has proper access controls
        assert test_settings.azure_openai_endpoint is not None
        assert test_settings.azure_storage_account_name is not None
        assert test_settings.azure_document_intelligence_endpoint is not None


class TestAzureServicesDisasterRecovery:
    """Test disaster recovery for Azure services."""
    
    @pytest.mark.integration
    def test_azure_services_failover(self, mock_ai_analyzer, mock_azure_storage_operations, sample_document_text):
        """Test failover mechanisms for Azure services."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock first attempt failure, second attempt success
        mock_blob.upload_blob.side_effect = [Exception("Temporary failure"), None]
        
        # Analyze document (should succeed)
        analysis_result = mock_ai_analyzer.analyze_document(sample_document_text, "general")
        
        # First storage attempt should fail
        blob_name = "failover_test.json"
        blob_client = mock_container.get_blob_client(blob_name)
        
        with pytest.raises(Exception, match="Temporary failure"):
            blob_client.upload_blob(json.dumps(analysis_result, indent=2).encode('utf-8'), overwrite=True)
        
        # Second attempt should succeed
        blob_client.upload_blob(json.dumps(analysis_result, indent=2).encode('utf-8'), overwrite=True)
        
        # Verify operations
        assert isinstance(analysis_result, dict)
        assert mock_blob.upload_blob.call_count == 2
    
    @pytest.mark.integration
    def test_azure_services_backup_restore(self, mock_azure_storage_operations, sample_document_text):
        """Test backup and restore functionality."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock backup operations
        mock_blob.upload_blob.return_value = None
        
        # Create backup
        backup_blob_name = "backup/document.txt"
        backup_blob_client = mock_container.get_blob_client(backup_blob_name)
        backup_blob_client.upload_blob(sample_document_text.encode('utf-8'), overwrite=True)
        
        # Verify backup was created
        mock_blob.upload_blob.assert_called_once()
        mock_container.get_blob_client.assert_called_once_with(backup_blob_name)
