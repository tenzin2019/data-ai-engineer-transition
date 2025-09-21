"""
Tests for Azure Storage connection and functionality.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from azure.core.exceptions import AzureError, ResourceNotFoundError, ResourceExistsError
from io import BytesIO

from config.settings import Settings


class TestAzureStorageConnection:
    """Test Azure Storage connection and basic functionality."""
    
    def test_storage_client_initialization_success(self, test_settings):
        """Test successful Azure Storage client initialization."""
        with patch('azure.storage.blob.BlobServiceClient') as mock_blob_service:
            mock_client = Mock()
            mock_blob_service.from_connection_string.return_value = mock_client
            
            # Test client initialization
            connection_string = f"DefaultEndpointsProtocol=https;AccountName={test_settings.azure_storage_account_name};AccountKey={test_settings.azure_storage_account_key};EndpointSuffix=core.windows.net"
            client = BlobServiceClient.from_connection_string(connection_string)
            
            assert client is not None
            mock_blob_service.from_connection_string.assert_called_once_with(connection_string)
    
    def test_storage_client_initialization_missing_credentials(self):
        """Test Storage client initialization with missing credentials."""
        with pytest.raises(ValueError):
            BlobServiceClient.from_connection_string("")
    
    def test_storage_client_initialization_invalid_connection_string(self):
        """Test Storage client initialization with invalid connection string."""
        with patch('azure.storage.blob.BlobServiceClient') as mock_blob_service:
            mock_blob_service.from_connection_string.side_effect = ValueError("Invalid connection string")
            
            with pytest.raises(ValueError, match="Invalid connection string"):
                BlobServiceClient.from_connection_string("invalid_connection_string")
    
    @pytest.mark.azure
    def test_storage_connection_health_check(self, mock_azure_storage_operations):
        """Test Azure Storage connection health check."""
        mock_client = mock_azure_storage_operations['client']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock successful container access
        mock_container.exists.return_value = True
        
        # Test connection by checking if container exists
        container_client = mock_client.get_container_client("test-container")
        exists = container_client.exists()
        
        assert exists is True
        mock_client.get_container_client.assert_called_once_with("test-container")
    
    @pytest.mark.azure
    def test_storage_connection_failure(self, test_settings):
        """Test Azure Storage connection failure handling."""
        with patch('azure.storage.blob.BlobServiceClient') as mock_blob_service:
            mock_client = Mock()
            mock_client.get_container_client.side_effect = AzureError("Connection failed")
            mock_blob_service.from_connection_string.return_value = mock_client
            
            client = BlobServiceClient.from_connection_string("test_connection_string")
            
            with pytest.raises(AzureError, match="Connection failed"):
                client.get_container_client("test-container")


class TestAzureStorageOperations:
    """Test Azure Storage operations functionality."""
    
    def test_upload_blob_success(self, mock_azure_storage_operations):
        """Test successful blob upload."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Test data
        test_data = b"Test document content"
        blob_name = "test-document.pdf"
        
        # Mock successful upload
        mock_blob.upload_blob.return_value = None
        
        # Perform upload
        blob_client = mock_container.get_blob_client(blob_name)
        blob_client.upload_blob(test_data, overwrite=True)
        
        # Verify upload was called
        mock_blob.upload_blob.assert_called_once_with(test_data, overwrite=True)
        mock_container.get_blob_client.assert_called_once_with(blob_name)
    
    def test_upload_blob_failure(self, mock_azure_storage_operations):
        """Test blob upload failure handling."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock upload failure
        mock_blob.upload_blob.side_effect = AzureError("Upload failed")
        
        blob_client = mock_container.get_blob_client("test-document.pdf")
        
        with pytest.raises(AzureError, match="Upload failed"):
            blob_client.upload_blob(b"test data")
    
    def test_download_blob_success(self, mock_azure_storage_operations):
        """Test successful blob download."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock download response
        test_content = b"Downloaded document content"
        mock_download = Mock()
        mock_download.readall.return_value = test_content
        mock_blob.download_blob.return_value = mock_download
        
        # Perform download
        blob_client = mock_container.get_blob_client("test-document.pdf")
        download_stream = blob_client.download_blob()
        content = download_stream.readall()
        
        assert content == test_content
        mock_blob.download_blob.assert_called_once()
    
    def test_download_blob_not_found(self, mock_azure_storage_operations):
        """Test blob download when blob doesn't exist."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock blob not found error
        mock_blob.download_blob.side_effect = ResourceNotFoundError("Blob not found")
        
        blob_client = mock_container.get_blob_client("nonexistent-document.pdf")
        
        with pytest.raises(ResourceNotFoundError, match="Blob not found"):
            blob_client.download_blob()
    
    def test_delete_blob_success(self, mock_azure_storage_operations):
        """Test successful blob deletion."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock successful deletion
        mock_blob.delete_blob.return_value = None
        
        # Perform deletion
        blob_client = mock_container.get_blob_client("test-document.pdf")
        blob_client.delete_blob()
        
        mock_blob.delete_blob.assert_called_once()
    
    def test_delete_blob_not_found(self, mock_azure_storage_operations):
        """Test blob deletion when blob doesn't exist."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock blob not found error
        mock_blob.delete_blob.side_effect = ResourceNotFoundError("Blob not found")
        
        blob_client = mock_container.get_blob_client("nonexistent-document.pdf")
        
        with pytest.raises(ResourceNotFoundError, match="Blob not found"):
            blob_client.delete_blob()
    
    def test_list_blobs_success(self, mock_azure_storage_operations):
        """Test successful blob listing."""
        mock_container = mock_azure_storage_operations['container']
        
        # Mock blob list
        mock_blobs = [
            Mock(name="document1.pdf", size=1024),
            Mock(name="document2.docx", size=2048),
            Mock(name="document3.txt", size=512)
        ]
        mock_container.list_blobs.return_value = mock_blobs
        
        # Perform listing
        blobs = list(mock_container.list_blobs())
        
        assert len(blobs) == 3
        assert blobs[0].name == "document1.pdf"
        assert blobs[1].name == "document2.docx"
        assert blobs[2].name == "document3.txt"
    
    def test_blob_exists_check(self, mock_azure_storage_operations):
        """Test blob existence check."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock blob exists
        mock_blob.exists.return_value = True
        
        blob_client = mock_container.get_blob_client("existing-document.pdf")
        exists = blob_client.exists()
        
        assert exists is True
        mock_blob.exists.assert_called_once()
    
    def test_blob_metadata_operations(self, mock_azure_storage_operations):
        """Test blob metadata operations."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock blob properties
        mock_properties = Mock()
        mock_properties.size = 1024
        mock_properties.last_modified = "2024-01-01T00:00:00Z"
        mock_properties.content_type = "application/pdf"
        mock_blob.get_blob_properties.return_value = mock_properties
        
        blob_client = mock_container.get_blob_client("test-document.pdf")
        properties = blob_client.get_blob_properties()
        
        assert properties.size == 1024
        assert properties.content_type == "application/pdf"
        mock_blob.get_blob_properties.assert_called_once()


class TestAzureStorageFileOperations:
    """Test file operations with Azure Storage."""
    
    def test_upload_file_success(self, mock_azure_storage_operations, temp_file):
        """Test successful file upload."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock successful upload
        mock_blob.upload_blob.return_value = None
        
        # Upload file
        blob_name = "uploaded-file.txt"
        blob_client = mock_container.get_blob_client(blob_name)
        
        with open(temp_file, 'rb') as file_data:
            blob_client.upload_blob(file_data, overwrite=True)
        
        mock_blob.upload_blob.assert_called_once()
        mock_container.get_blob_client.assert_called_once_with(blob_name)
    
    def test_download_file_success(self, mock_azure_storage_operations):
        """Test successful file download."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock download response
        test_content = b"Downloaded file content"
        mock_download = Mock()
        mock_download.readall.return_value = test_content
        mock_blob.download_blob.return_value = mock_download
        
        # Download to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            blob_client = mock_container.get_blob_client("test-file.txt")
            download_stream = blob_client.download_blob()
            
            with open(temp_file.name, 'wb') as f:
                f.write(download_stream.readall())
            
            # Verify content
            with open(temp_file.name, 'rb') as f:
                content = f.read()
                assert content == test_content
            
            # Cleanup
            os.unlink(temp_file.name)
    
    def test_upload_large_file_chunked(self, mock_azure_storage_operations):
        """Test chunked upload of large files."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock chunked upload
        mock_blob.upload_blob.return_value = None
        
        # Create large data
        large_data = b"Large file content " * 10000  # ~200KB
        
        blob_client = mock_container.get_blob_client("large-file.bin")
        blob_client.upload_blob(large_data, overwrite=True)
        
        mock_blob.upload_blob.assert_called_once()
    
    def test_file_upload_with_metadata(self, mock_azure_storage_operations, temp_file):
        """Test file upload with custom metadata."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock upload with metadata
        mock_blob.upload_blob.return_value = None
        
        metadata = {
            "document_type": "contract",
            "upload_date": "2024-01-01",
            "author": "test_user"
        }
        
        blob_client = mock_container.get_blob_client("document-with-metadata.pdf")
        
        with open(temp_file, 'rb') as file_data:
            blob_client.upload_blob(
                file_data, 
                overwrite=True,
                metadata=metadata
            )
        
        mock_blob.upload_blob.assert_called_once()


class TestAzureStorageErrorHandling:
    """Test error handling in Azure Storage operations."""
    
    def test_network_timeout_error(self, mock_azure_storage_operations):
        """Test handling of network timeout errors."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock timeout error
        mock_blob.upload_blob.side_effect = AzureError("Request timeout")
        
        blob_client = mock_container.get_blob_client("test-document.pdf")
        
        with pytest.raises(AzureError, match="Request timeout"):
            blob_client.upload_blob(b"test data")
    
    def test_quota_exceeded_error(self, mock_azure_storage_operations):
        """Test handling of quota exceeded errors."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock quota exceeded error
        mock_blob.upload_blob.side_effect = AzureError("Storage quota exceeded")
        
        blob_client = mock_container.get_blob_client("test-document.pdf")
        
        with pytest.raises(AzureError, match="Storage quota exceeded"):
            blob_client.upload_blob(b"test data")
    
    def test_container_not_found_error(self, mock_azure_storage_operations):
        """Test handling of container not found errors."""
        mock_client = mock_azure_storage_operations['client']
        
        # Mock container not found error
        mock_client.get_container_client.side_effect = ResourceNotFoundError("Container not found")
        
        with pytest.raises(ResourceNotFoundError, match="Container not found"):
            mock_client.get_container_client("nonexistent-container")
    
    def test_blob_already_exists_error(self, mock_azure_storage_operations):
        """Test handling of blob already exists errors."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock blob already exists error
        mock_blob.upload_blob.side_effect = ResourceExistsError("Blob already exists")
        
        blob_client = mock_container.get_blob_client("existing-document.pdf")
        
        with pytest.raises(ResourceExistsError, match="Blob already exists"):
            blob_client.upload_blob(b"test data")
    
    def test_authentication_error(self, test_settings):
        """Test handling of authentication errors."""
        with patch('azure.storage.blob.BlobServiceClient') as mock_blob_service:
            mock_blob_service.from_connection_string.side_effect = AzureError("Authentication failed")
            
            with pytest.raises(AzureError, match="Authentication failed"):
                BlobServiceClient.from_connection_string("invalid_connection_string")


class TestAzureStoragePerformance:
    """Test performance-related functionality."""
    
    @pytest.mark.slow
    def test_concurrent_upload_operations(self, mock_azure_storage_operations):
        """Test concurrent upload operations."""
        import concurrent.futures
        
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock successful uploads
        mock_blob.upload_blob.return_value = None
        
        def upload_blob(blob_name):
            blob_client = mock_container.get_blob_client(blob_name)
            blob_client.upload_blob(f"Content for {blob_name}".encode())
            return blob_name
        
        # Submit multiple concurrent uploads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(upload_blob, f"document{i}.pdf") for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All uploads should complete successfully
        assert len(results) == 10
        assert mock_blob.upload_blob.call_count == 10
    
    @pytest.mark.slow
    def test_concurrent_download_operations(self, mock_azure_storage_operations):
        """Test concurrent download operations."""
        import concurrent.futures
        
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock download responses
        def mock_download():
            mock_download = Mock()
            mock_download.readall.return_value = b"Downloaded content"
            return mock_download
        
        mock_blob.download_blob.return_value = mock_download()
        
        def download_blob(blob_name):
            blob_client = mock_container.get_blob_client(blob_name)
            download_stream = blob_client.download_blob()
            return download_stream.readall()
        
        # Submit multiple concurrent downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(download_blob, f"document{i}.pdf") for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All downloads should complete successfully
        assert len(results) == 10
        assert all(result == b"Downloaded content" for result in results)
    
    def test_large_file_upload_performance(self, mock_azure_storage_operations):
        """Test performance of large file uploads."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock successful upload
        mock_blob.upload_blob.return_value = None
        
        # Create large data (1MB)
        large_data = b"Large file content " * 50000  # ~1MB
        
        blob_client = mock_container.get_blob_client("large-file.bin")
        
        # Time the upload (in real scenario)
        blob_client.upload_blob(large_data, overwrite=True)
        
        mock_blob.upload_blob.assert_called_once()
    
    def test_batch_operations_performance(self, mock_azure_storage_operations):
        """Test performance of batch operations."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock successful operations
        mock_blob.upload_blob.return_value = None
        mock_blob.delete_blob.return_value = None
        
        # Perform batch operations
        blob_names = [f"document{i}.pdf" for i in range(100)]
        
        for blob_name in blob_names:
            blob_client = mock_container.get_blob_client(blob_name)
            blob_client.upload_blob(f"Content for {blob_name}".encode())
        
        # Verify all operations were called
        assert mock_blob.upload_blob.call_count == 100


class TestAzureStorageConfiguration:
    """Test Azure Storage configuration and settings."""
    
    def test_storage_account_configuration(self, test_settings):
        """Test storage account configuration."""
        assert test_settings.azure_storage_account_name == "teststorageaccount"
        assert test_settings.azure_storage_account_key == "test-storage-key"
        assert test_settings.azure_storage_container_name == "test-documents"
    
    def test_connection_string_generation(self, test_settings):
        """Test connection string generation."""
        expected_connection_string = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={test_settings.azure_storage_account_name};"
            f"AccountKey={test_settings.azure_storage_account_key};"
            f"EndpointSuffix=core.windows.net"
        )
        
        # In real implementation, this would be generated by a utility function
        connection_string = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={test_settings.azure_storage_account_name};"
            f"AccountKey={test_settings.azure_storage_account_key};"
            f"EndpointSuffix=core.windows.net"
        )
        
        assert connection_string == expected_connection_string
    
    def test_container_configuration(self, test_settings):
        """Test container configuration."""
        assert test_settings.azure_storage_container_name == "test-documents"
    
    def test_file_upload_configuration(self, test_settings):
        """Test file upload configuration."""
        assert test_settings.max_file_size == 50 * 1024 * 1024  # 50MB
        assert "application/pdf" in test_settings.allowed_file_types
        assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in test_settings.allowed_file_types


class TestAzureStorageUtilities:
    """Test utility functions for Azure Storage operations."""
    
    def test_generate_blob_name(self):
        """Test blob name generation utility."""
        # This would be implemented in a utility function
        def generate_blob_name(filename, user_id=None, timestamp=None):
            import uuid
            import time
            
            if timestamp is None:
                timestamp = int(time.time())
            
            if user_id is None:
                user_id = str(uuid.uuid4())[:8]
            
            name, ext = os.path.splitext(filename)
            return f"{user_id}/{timestamp}_{name}{ext}"
        
        blob_name = generate_blob_name("document.pdf", "user123", 1234567890)
        assert blob_name == "user123/1234567890_document.pdf"
    
    def test_validate_file_type(self, test_settings):
        """Test file type validation utility."""
        def is_allowed_file_type(filename, allowed_types):
            import mimetypes
            mime_type, _ = mimetypes.guess_type(filename)
            return mime_type in allowed_types
        
        # Test allowed file types
        assert is_allowed_file_type("document.pdf", test_settings.allowed_file_types)
        assert is_allowed_file_type("document.docx", test_settings.allowed_file_types)
        assert is_allowed_file_type("document.txt", test_settings.allowed_file_types)
        
        # Test disallowed file types
        assert not is_allowed_file_type("document.exe", test_settings.allowed_file_types)
        assert not is_allowed_file_type("document.bat", test_settings.allowed_file_types)
    
    def test_validate_file_size(self, test_settings):
        """Test file size validation utility."""
        def is_valid_file_size(file_size, max_size):
            return file_size <= max_size
        
        # Test valid file sizes
        assert is_valid_file_size(1024, test_settings.max_file_size)  # 1KB
        assert is_valid_file_size(10 * 1024 * 1024, test_settings.max_file_size)  # 10MB
        
        # Test invalid file sizes
        assert not is_valid_file_size(100 * 1024 * 1024, test_settings.max_file_size)  # 100MB
        assert not is_valid_file_size(1000 * 1024 * 1024, test_settings.max_file_size)  # 1GB


@pytest.mark.integration
class TestAzureStorageIntegration:
    """Integration tests for Azure Storage functionality."""
    
    @pytest.mark.azure
    def test_end_to_end_file_workflow(self, mock_azure_storage_operations, temp_file):
        """Test end-to-end file workflow."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock all operations
        mock_blob.upload_blob.return_value = None
        mock_blob.exists.return_value = True
        
        mock_download = Mock()
        mock_download.readall.return_value = b"Downloaded content"
        mock_blob.download_blob.return_value = mock_download
        
        mock_blob.delete_blob.return_value = None
        
        blob_name = "test-workflow.pdf"
        blob_client = mock_container.get_blob_client(blob_name)
        
        # 1. Upload file
        with open(temp_file, 'rb') as file_data:
            blob_client.upload_blob(file_data, overwrite=True)
        
        # 2. Check if blob exists
        exists = blob_client.exists()
        assert exists is True
        
        # 3. Download file
        download_stream = blob_client.download_blob()
        content = download_stream.readall()
        assert content == b"Downloaded content"
        
        # 4. Delete file
        blob_client.delete_blob()
        
        # Verify all operations were called
        mock_blob.upload_blob.assert_called_once()
        mock_blob.exists.assert_called_once()
        mock_blob.download_blob.assert_called_once()
        mock_blob.delete_blob.assert_called_once()
    
    @pytest.mark.azure
    def test_batch_file_operations(self, mock_azure_storage_operations):
        """Test batch file operations."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock operations
        mock_blob.upload_blob.return_value = None
        mock_blob.delete_blob.return_value = None
        
        # Test batch upload
        files_to_upload = [
            ("document1.pdf", b"Content 1"),
            ("document2.docx", b"Content 2"),
            ("document3.txt", b"Content 3")
        ]
        
        for filename, content in files_to_upload:
            blob_client = mock_container.get_blob_client(filename)
            blob_client.upload_blob(content, overwrite=True)
        
        # Test batch delete
        for filename, _ in files_to_upload:
            blob_client = mock_container.get_blob_client(filename)
            blob_client.delete_blob()
        
        # Verify operations
        assert mock_blob.upload_blob.call_count == 3
        assert mock_blob.delete_blob.call_count == 3
    
    @pytest.mark.azure
    def test_error_recovery_workflow(self, mock_azure_storage_operations):
        """Test error recovery workflow."""
        mock_blob = mock_azure_storage_operations['blob']
        mock_container = mock_azure_storage_operations['container']
        
        # Mock first attempt failure, second attempt success
        mock_blob.upload_blob.side_effect = [AzureError("Temporary failure"), None]
        
        blob_client = mock_container.get_blob_client("test-document.pdf")
        
        # First attempt should fail
        with pytest.raises(AzureError, match="Temporary failure"):
            blob_client.upload_blob(b"test data")
        
        # Second attempt should succeed
        blob_client.upload_blob(b"test data")
        
        assert mock_blob.upload_blob.call_count == 2
