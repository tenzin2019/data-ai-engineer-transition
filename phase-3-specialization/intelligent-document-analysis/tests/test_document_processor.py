"""
Tests for the Document Processor module.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.core.document_processor import DocumentProcessor
from src.models.document import DocumentType


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
        self.test_data_dir = Path(__file__).parent.parent / "data" / "sample_documents"
    
    def test_processor_initialization(self):
        """Test DocumentProcessor initialization."""
        assert self.processor is not None
        assert hasattr(self.processor, 'supported_types')
        assert len(self.processor.supported_types) > 0
    
    def test_supported_file_types(self):
        """Test supported file types."""
        expected_types = {
            'application/pdf': DocumentType.PDF,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocumentType.DOCX,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': DocumentType.XLSX,
            'text/plain': DocumentType.TXT,
        }
        
        assert self.processor.supported_types == expected_types
    
    def test_process_txt_file(self):
        """Test processing of plain text files."""
        # Create a temporary text file
        test_content = "This is a test document.\nIt contains multiple lines.\nFor testing purposes."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file = f.name
        
        try:
            result = self.processor.process_document(temp_file)
            
            assert result is not None
            assert 'text' in result
            assert 'page_count' in result
            assert 'metadata' in result
            assert 'document_type' in result
            
            assert result['text'] == test_content
            assert result['page_count'] == 1
            assert result['document_type'] == DocumentType.TXT
            assert 'encoding' in result['metadata']
            
        finally:
            os.unlink(temp_file)
    
    def test_process_nonexistent_file(self):
        """Test processing of non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.processor.process_document("nonexistent_file.txt")
    
    def test_process_unsupported_file_type(self):
        """Test processing of unsupported file type."""
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                self.processor.process_document(temp_file, mime_type="application/xyz")
        finally:
            os.unlink(temp_file)
    
    def test_get_document_statistics(self):
        """Test document statistics calculation."""
        test_text = "This is a test document. It has multiple sentences. For testing purposes."
        
        stats = self.processor.get_document_statistics(test_text)
        
        assert stats is not None
        assert 'character_count' in stats
        assert 'word_count' in stats
        assert 'sentence_count' in stats
        assert 'paragraph_count' in stats
        assert 'line_count' in stats
        
        assert stats['character_count'] > 0
        assert stats['word_count'] > 0
        assert stats['sentence_count'] > 0
    
    def test_get_document_statistics_empty_text(self):
        """Test document statistics with empty text."""
        stats = self.processor.get_document_statistics("")
        
        assert stats['character_count'] == 0
        assert stats['word_count'] == 0
        assert stats['sentence_count'] == 0
        assert stats['paragraph_count'] == 0
        assert stats['line_count'] == 0
    
    def test_get_document_statistics_none_text(self):
        """Test document statistics with None text."""
        stats = self.processor.get_document_statistics(None)
        
        assert stats['character_count'] == 0
        assert stats['word_count'] == 0
        assert stats['sentence_count'] == 0
        assert stats['paragraph_count'] == 0
        assert stats['line_count'] == 0
    
    def test_sample_contract_processing(self):
        """Test processing of sample contract document."""
        sample_file = self.test_data_dir / "sample_contract.txt"
        
        if sample_file.exists():
            result = self.processor.process_document(sample_file)
            
            assert result is not None
            assert 'text' in result
            assert len(result['text']) > 0
            assert result['document_type'] == DocumentType.TXT
            
            # Check that key contract terms are present
            text = result['text'].lower()
            assert 'agreement' in text
            assert 'license' in text
            assert 'terms' in text
    
    def test_sample_financial_report_processing(self):
        """Test processing of sample financial report document."""
        sample_file = self.test_data_dir / "sample_financial_report.txt"
        
        if sample_file.exists():
            result = self.processor.process_document(sample_file)
            
            assert result is not None
            assert 'text' in result
            assert len(result['text']) > 0
            assert result['document_type'] == DocumentType.TXT
            
            # Check that key financial terms are present
            text = result['text'].lower()
            assert 'revenue' in text
            assert 'financial' in text
            assert 'quarterly' in text


if __name__ == "__main__":
    pytest.main([__file__])
