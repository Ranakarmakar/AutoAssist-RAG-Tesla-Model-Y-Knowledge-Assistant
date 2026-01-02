"""Tests for document processor."""

import pytest
import os
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.document_processor import DocumentProcessor
from src.exceptions import DocumentProcessingError


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
    
    def test_validate_documents_success(self):
        """Test successful document validation."""
        documents = [
            Document(page_content="This is valid content", metadata={"page": 1}),
            Document(page_content="Another valid page", metadata={"page": 2})
        ]
        
        assert self.processor.validate_documents(documents) is True
    
    def test_validate_documents_empty_list(self):
        """Test validation with empty document list."""
        assert self.processor.validate_documents([]) is False
    
    def test_validate_documents_empty_content(self):
        """Test validation with empty content."""
        documents = [
            Document(page_content="", metadata={"page": 1})
        ]
        
        assert self.processor.validate_documents(documents) is False
    
    def test_validate_documents_whitespace_only(self):
        """Test validation with whitespace-only content."""
        documents = [
            Document(page_content="   \n\t  ", metadata={"page": 1})
        ]
        
        assert self.processor.validate_documents(documents) is False
    
    def test_validate_documents_low_readability(self):
        """Test validation with low readability content."""
        # Create content with many non-printable characters
        bad_content = "Valid text" + "\x00\x01\x02" * 100
        documents = [
            Document(page_content=bad_content, metadata={"page": 1})
        ]
        
        assert self.processor.validate_documents(documents) is False
    
    @patch('src.document_processor.validate_pdf_file')
    @patch('src.document_processor.PyPDFLoader')
    def test_load_pdf_success(self, mock_loader_class, mock_validate):
        """Test successful PDF loading."""
        # Mock validation
        mock_validate.return_value = True
        
        # Mock PyPDFLoader
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        # Mock documents
        mock_docs = [
            Document(page_content="Page 1 content", metadata={"source": "test.pdf"}),
            Document(page_content="Page 2 content", metadata={"source": "test.pdf"})
        ]
        mock_loader.load.return_value = mock_docs
        
        # Mock file operations
        with patch('src.document_processor.get_file_size_mb', return_value=1.5):
            with patch('os.path.basename', return_value="test.pdf"):
                result = self.processor.load_pdf("test.pdf")
        
        assert len(result) == 2
        assert result[0].metadata['source_file'] == "test.pdf"
        assert result[0].metadata['page_number'] == 1
        assert result[1].metadata['page_number'] == 2
    
    @patch('src.document_processor.validate_pdf_file')
    def test_load_pdf_invalid_file(self, mock_validate):
        """Test loading invalid PDF file."""
        mock_validate.return_value = False
        
        with pytest.raises(DocumentProcessingError, match="Invalid PDF file"):
            self.processor.load_pdf("invalid.pdf")
    
    @patch('src.document_processor.validate_pdf_file')
    @patch('src.document_processor.PyPDFLoader')
    def test_load_pdf_no_content(self, mock_loader_class, mock_validate):
        """Test loading PDF with no extractable content."""
        mock_validate.return_value = True
        
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.load.return_value = []
        
        with pytest.raises(DocumentProcessingError, match="No content extracted"):
            self.processor.load_pdf("empty.pdf")
    
    @patch('os.path.exists')
    def test_extract_metadata_file_not_found(self, mock_exists):
        """Test metadata extraction for non-existent file."""
        mock_exists.return_value = False
        
        with pytest.raises(DocumentProcessingError, match="PDF file not found"):
            self.processor.extract_metadata("nonexistent.pdf")
    
    @patch('os.path.exists')
    @patch('os.stat')
    @patch('src.document_processor.get_file_size_mb')
    def test_extract_metadata_success(self, mock_size, mock_stat, mock_exists):
        """Test successful metadata extraction."""
        mock_exists.return_value = True
        
        # Mock file stats
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 1024
        mock_stat_result.st_ctime = 1640995200
        mock_stat_result.st_mtime = 1640995200
        mock_stat.return_value = mock_stat_result
        
        mock_size.return_value = 1.0
        
        with patch('os.path.basename', return_value="test.pdf"):
            metadata = self.processor.extract_metadata("test.pdf")
        
        assert metadata['file_name'] == "test.pdf"
        assert metadata['file_size_bytes'] == 1024
        assert metadata['file_size_mb'] == 1.0
    
    @patch.object(DocumentProcessor, 'load_pdf')
    @patch.object(DocumentProcessor, 'validate_documents')
    def test_process_pdf_file_success(self, mock_validate, mock_load):
        """Test complete PDF processing pipeline."""
        mock_docs = [Document(page_content="Test content", metadata={})]
        mock_load.return_value = mock_docs
        mock_validate.return_value = True
        
        result = self.processor.process_pdf_file("test.pdf")
        
        assert result == mock_docs
        mock_load.assert_called_once_with("test.pdf")
        mock_validate.assert_called_once_with(mock_docs)
    
    @patch.object(DocumentProcessor, 'load_pdf')
    @patch.object(DocumentProcessor, 'validate_documents')
    def test_process_pdf_file_validation_failure(self, mock_validate, mock_load):
        """Test PDF processing with validation failure."""
        mock_docs = [Document(page_content="", metadata={})]
        mock_load.return_value = mock_docs
        mock_validate.return_value = False
        
        with pytest.raises(DocumentProcessingError, match="Document validation failed"):
            self.processor.process_pdf_file("test.pdf")