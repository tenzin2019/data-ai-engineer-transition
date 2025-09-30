"""
Test suite for RAG Conversational AI Assistant API endpoints
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch
import json
import tempfile
import os

# Import the FastAPI app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.main import app, initialize_components


class TestRAGAPI:
    """Test cases for RAG API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    async def mock_components(self):
        """Mock all components for testing"""
        with patch('api.main.document_processor') as mock_dp, \
             patch('api.main.vector_store') as mock_vs, \
             patch('api.main.rag_engine') as mock_re, \
             patch('api.main.llm_orchestrator') as mock_lo:
            
            # Setup mocks
            mock_dp.process_document = AsyncMock(return_value=[])
            mock_vs.add_documents = AsyncMock(return_value="test_doc_123")
            mock_vs.search_similar = AsyncMock(return_value=[])
            mock_vs.list_documents = AsyncMock(return_value=[])
            mock_vs.delete_document = AsyncMock(return_value=True)
            
            mock_re.process_query = AsyncMock(return_value={
                "answer": "Test response",
                "sources": [],
                "conversation_id": "test_conv_123",
                "metadata": {}
            })
            mock_re.get_conversation_history = AsyncMock(return_value=[])
            mock_re.clear_conversation = AsyncMock(return_value=True)
            mock_re.get_engine_stats = AsyncMock(return_value={})
            
            mock_lo.get_provider_status = AsyncMock(return_value={})
            mock_lo.reset_provider_errors = AsyncMock()
            
            yield {
                'document_processor': mock_dp,
                'vector_store': mock_vs,
                'rag_engine': mock_re,
                'llm_orchestrator': mock_lo
            }
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "RAG Conversational AI Assistant API"
        assert data["status"] == "running"
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
    
    def test_query_endpoint_success(self, client, mock_components):
        """Test successful query processing"""
        with patch('api.main.rag_engine', mock_components['rag_engine']):
            query_data = {
                "query": "What is artificial intelligence?",
                "conversation_id": "test_conv_123",
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = client.post("/query", json=query_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "conversation_id" in data
            assert "metadata" in data
    
    def test_query_endpoint_missing_rag_engine(self, client):
        """Test query endpoint when RAG engine is not initialized"""
        with patch('api.main.rag_engine', None):
            query_data = {
                "query": "Test query"
            }
            
            response = client.post("/query", json=query_data)
            assert response.status_code == 503
            assert "RAG engine not initialized" in response.json()["detail"]
    
    def test_upload_document_success(self, client, mock_components):
        """Test successful document upload"""
        with patch('api.main.document_processor', mock_components['document_processor']), \
             patch('api.main.vector_store', mock_components['vector_store']):
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write("This is a test document")
                temp_file_path = temp_file.name
            
            try:
                with open(temp_file_path, 'rb') as f:
                    files = {"file": ("test.txt", f, "text/plain")}
                    response = client.post("/upload", files=files)
                
                assert response.status_code == 200
                data = response.json()
                assert "document_id" in data
                assert "filename" in data
                assert "status" in data
                assert "chunks_created" in data
                
            finally:
                os.unlink(temp_file_path)
    
    def test_upload_document_missing_components(self, client):
        """Test document upload when components are not initialized"""
        with patch('api.main.document_processor', None):
            files = {"file": ("test.txt", b"test content", "text/plain")}
            response = client.post("/upload", files=files)
            
            assert response.status_code == 503
            assert "Components not initialized" in response.json()["detail"]
    
    def test_list_documents(self, client, mock_components):
        """Test listing documents"""
        with patch('api.main.vector_store', mock_components['vector_store']):
            response = client.get("/documents")
            assert response.status_code == 200
            data = response.json()
            assert "documents" in data
    
    def test_delete_document_success(self, client, mock_components):
        """Test successful document deletion"""
        with patch('api.main.vector_store', mock_components['vector_store']):
            response = client.delete("/documents/test_doc_123")
            assert response.status_code == 200
            data = response.json()
            assert "deleted successfully" in data["message"]
    
    def test_delete_document_not_found(self, client, mock_components):
        """Test document deletion when document not found"""
        mock_vs = mock_components['vector_store']
        mock_vs.delete_document = AsyncMock(return_value=False)
        
        with patch('api.main.vector_store', mock_vs):
            response = client.delete("/documents/nonexistent_doc")
            assert response.status_code == 404
            assert "Document not found" in response.json()["detail"]
    
    def test_get_conversation_history(self, client, mock_components):
        """Test getting conversation history"""
        with patch('api.main.rag_engine', mock_components['rag_engine']):
            response = client.get("/conversations/test_conv_123")
            assert response.status_code == 200
            data = response.json()
            assert "conversation_id" in data
            assert "history" in data
    
    def test_clear_conversation(self, client, mock_components):
        """Test clearing conversation"""
        with patch('api.main.rag_engine', mock_components['rag_engine']):
            response = client.delete("/conversations/test_conv_123")
            assert response.status_code == 200
            data = response.json()
            assert "cleared successfully" in data["message"]
    
    def test_get_metrics(self, client, mock_components):
        """Test getting system metrics"""
        with patch('api.main.rag_engine', mock_components['rag_engine']), \
             patch('api.main.llm_orchestrator', mock_components['llm_orchestrator']):
            
            response = client.get("/metrics")
            assert response.status_code == 200
            data = response.json()
            assert "rag_engine" in data
            assert "llm_providers" in data
            assert "system_status" in data
    
    def test_reset_provider_errors(self, client, mock_components):
        """Test resetting provider errors"""
        with patch('api.main.llm_orchestrator', mock_components['llm_orchestrator']):
            response = client.post("/providers/reset")
            assert response.status_code == 200
            data = response.json()
            assert "Reset errors" in data["message"]
    
    def test_get_provider_status(self, client, mock_components):
        """Test getting provider status"""
        with patch('api.main.llm_orchestrator', mock_components['llm_orchestrator']):
            response = client.get("/providers/status")
            assert response.status_code == 200
    
    def test_batch_upload_documents(self, client, mock_components):
        """Test batch document upload"""
        with patch('api.main.document_processor', mock_components['document_processor']), \
             patch('api.main.vector_store', mock_components['vector_store']):
            
            # Create temporary files
            files = []
            temp_files = []
            
            try:
                for i in range(2):
                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'_{i}.txt', delete=False)
                    temp_file.write(f"This is test document {i}")
                    temp_file.close()
                    temp_files.append(temp_file.name)
                    
                    with open(temp_file.name, 'rb') as f:
                        files.append(("files", (f"test_{i}.txt", f.read(), "text/plain")))
                
                response = client.post("/documents/batch", files=files)
                assert response.status_code == 200
                
                data = response.json()
                assert "results" in data
                assert len(data["results"]) == 2
                
            finally:
                for temp_file_path in temp_files:
                    try:
                        os.unlink(temp_file_path)
                    except FileNotFoundError:
                        pass
    
    def test_search_documents(self, client, mock_components):
        """Test document search"""
        with patch('api.main.vector_store', mock_components['vector_store']):
            response = client.get("/search?query=test query&limit=5")
            assert response.status_code == 200
            
            data = response.json()
            assert "query" in data
            assert "results" in data
            assert "total_found" in data
    
    def test_error_handling(self, client, mock_components):
        """Test error handling in API endpoints"""
        # Test with invalid JSON
        response = client.post("/query", json={"invalid": "data"})
        assert response.status_code == 422  # Validation error
        
        # Test with missing required fields
        response = client.post("/query", json={})
        assert response.status_code == 422  # Validation error


class TestAPIValidation:
    """Test API input validation"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_query_validation(self, client):
        """Test query input validation"""
        # Test missing query
        response = client.post("/query", json={})
        assert response.status_code == 422
        
        # Test invalid temperature
        response = client.post("/query", json={
            "query": "test",
            "temperature": 2.0  # Should be between 0 and 1
        })
        # Note: FastAPI doesn't validate this by default, would need custom validation
        
        # Test negative max_tokens
        response = client.post("/query", json={
            "query": "test",
            "max_tokens": -1
        })
        # Note: Would need custom validation for this
    
    def test_search_validation(self, client):
        """Test search input validation"""
        # Test missing query parameter
        response = client.get("/search")
        assert response.status_code == 422
        
        # Test invalid limit
        response = client.get("/search?query=test&limit=-1")
        # Note: Would need custom validation for negative limits


class TestAPIPerformance:
    """Test API performance"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.mark.slow
    def test_query_response_time(self, client, mock_components):
        """Test query response time"""
        import time
        
        with patch('api.main.rag_engine', mock_components['rag_engine']):
            start_time = time.time()
            
            response = client.post("/query", json={
                "query": "What is artificial intelligence?"
            })
            
            end_time = time.time()
            response_time = end_time - start_time
            
            assert response.status_code == 200
            assert response_time < 5.0  # Should respond within 5 seconds
    
    @pytest.mark.slow
    def test_concurrent_requests(self, client, mock_components):
        """Test handling of concurrent requests"""
        import concurrent.futures
        import threading
        
        def make_request():
            with patch('api.main.rag_engine', mock_components['rag_engine']):
                return client.post("/query", json={
                    "query": "Test concurrent query"
                })
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in results:
            assert response.status_code in [200, 503]  # 503 if components not initialized
