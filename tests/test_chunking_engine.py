"""Tests for chunking engine."""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.chunking_engine import ChunkingEngine
from src.exceptions import DocumentProcessingError


class TestChunkingEngine:
    """Test cases for ChunkingEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ChunkingEngine(chunk_size=100, chunk_overlap=20)
    
    def test_initialization_default_settings(self):
        """Test chunking engine initialization with default settings."""
        with patch('src.chunking_engine.settings') as mock_settings:
            mock_settings.chunk_size = 500
            mock_settings.chunk_overlap = 50
            
            engine = ChunkingEngine()
            
            assert engine.chunk_size == 500
            assert engine.chunk_overlap == 50
            assert engine.separators == ["\n\n", "\n", ". ", " ", ""]
    
    def test_initialization_custom_settings(self):
        """Test chunking engine initialization with custom settings."""
        custom_separators = ["\n", ". "]
        engine = ChunkingEngine(
            chunk_size=200,
            chunk_overlap=30,
            separators=custom_separators
        )
        
        assert engine.chunk_size == 200
        assert engine.chunk_overlap == 30
        assert engine.separators == custom_separators
    
    def test_split_documents_empty_list(self):
        """Test splitting empty document list."""
        result = self.engine.split_documents([])
        assert result == []
    
    def test_split_documents_success(self):
        """Test successful document splitting."""
        # Create a long document that will be split
        long_content = "This is a test document. " * 20  # 500 characters
        document = Document(
            page_content=long_content,
            metadata={
                'source_file': 'test.pdf',
                'page_number': 1
            }
        )
        
        chunks = self.engine.split_documents([document])
        
        # Should create multiple chunks due to length
        assert len(chunks) > 1
        
        # Check that metadata is enhanced
        for i, chunk in enumerate(chunks):
            assert 'chunk_index' in chunk.metadata
            assert 'chunk_size' in chunk.metadata
            assert 'chunk_word_count' in chunk.metadata
            assert 'chunking_method' in chunk.metadata
            assert chunk.metadata['chunking_method'] == 'RecursiveCharacterTextSplitter'
            assert 'chunk_id' in chunk.metadata
    
    def test_split_documents_preserves_metadata(self):
        """Test that original metadata is preserved during splitting."""
        document = Document(
            page_content="Short content that won't be split.",
            metadata={
                'source_file': 'test.pdf',
                'page_number': 1,
                'custom_field': 'custom_value'
            }
        )
        
        chunks = self.engine.split_documents([document])
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        # Original metadata should be preserved
        assert chunk.metadata['source_file'] == 'test.pdf'
        assert chunk.metadata['page_number'] == 1
        assert chunk.metadata['custom_field'] == 'custom_value'
        
        # Enhanced metadata should be added
        assert 'chunk_index' in chunk.metadata
        assert 'chunk_size' in chunk.metadata
    
    def test_validate_chunk_valid(self):
        """Test chunk validation with valid chunk."""
        chunk = Document(
            page_content="This is a valid chunk with enough content.",
            metadata={}
        )
        
        assert self.engine._validate_chunk(chunk) is True
    
    def test_validate_chunk_empty(self):
        """Test chunk validation with empty chunk."""
        chunk = Document(page_content="", metadata={})
        assert self.engine._validate_chunk(chunk) is False
    
    def test_validate_chunk_whitespace_only(self):
        """Test chunk validation with whitespace-only chunk."""
        chunk = Document(page_content="   \n\t  ", metadata={})
        assert self.engine._validate_chunk(chunk) is False
    
    def test_validate_chunk_too_short(self):
        """Test chunk validation with too short chunk."""
        chunk = Document(page_content="Short", metadata={})
        assert self.engine._validate_chunk(chunk) is False
    
    def test_validate_chunk_no_alphanumeric(self):
        """Test chunk validation with no alphanumeric characters."""
        chunk = Document(page_content="!@#$%^&*()", metadata={})
        assert self.engine._validate_chunk(chunk) is False
    
    def test_create_chunk_metadata(self):
        """Test chunk metadata creation."""
        chunk = Document(
            page_content="This is test content for metadata extraction.",
            metadata={
                'source_file': 'test.pdf',
                'chunk_index': 0
            }
        )
        
        metadata = self.engine.create_chunk_metadata(chunk)
        
        assert 'content_preview' in metadata
        assert 'has_content' in metadata
        assert 'is_valid_chunk' in metadata
        assert metadata['has_content'] is True
        assert metadata['is_valid_chunk'] is True
        assert metadata['source_file'] == 'test.pdf'
    
    def test_get_chunk_statistics_empty(self):
        """Test statistics generation with empty chunk list."""
        stats = self.engine.get_chunk_statistics([])
        
        assert stats['total_chunks'] == 0
        assert stats['total_characters'] == 0
        assert stats['avg_chunk_size'] == 0
    
    def test_get_chunk_statistics_with_chunks(self):
        """Test statistics generation with actual chunks."""
        chunks = [
            Document(page_content="Short chunk", metadata={}),
            Document(page_content="This is a medium length chunk with more content", metadata={}),
            Document(page_content="This is a very long chunk with lots of content that should be considered large", metadata={})
        ]
        
        stats = self.engine.get_chunk_statistics(chunks)
        
        assert stats['total_chunks'] == 3
        assert stats['total_characters'] > 0
        assert stats['total_words'] > 0
        assert stats['avg_chunk_size'] > 0
        assert stats['min_chunk_size'] > 0
        assert stats['max_chunk_size'] > 0
        assert 'chunk_size_distribution' in stats
    
    def test_chunk_single_document(self):
        """Test chunking a single document."""
        document = Document(
            page_content="This is a test document for single document chunking.",
            metadata={'source': 'test.pdf'}
        )
        
        chunks = self.engine.chunk_single_document(document)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Document) for chunk in chunks)
    
    def test_reconfigure(self):
        """Test reconfiguring the chunking engine."""
        original_size = self.engine.chunk_size
        original_overlap = self.engine.chunk_overlap
        
        new_size = 200
        new_overlap = 40
        new_separators = ["\n", ". "]
        
        self.engine.reconfigure(
            chunk_size=new_size,
            chunk_overlap=new_overlap,
            separators=new_separators
        )
        
        assert self.engine.chunk_size == new_size
        assert self.engine.chunk_overlap == new_overlap
        assert self.engine.separators == new_separators
    
    def test_reconfigure_partial(self):
        """Test partial reconfiguration."""
        original_size = self.engine.chunk_size
        original_separators = self.engine.separators
        
        new_overlap = 50
        
        self.engine.reconfigure(chunk_overlap=new_overlap)
        
        # Only overlap should change
        assert self.engine.chunk_size == original_size
        assert self.engine.chunk_overlap == new_overlap
        assert self.engine.separators == original_separators
    
    @patch('src.chunking_engine.RecursiveCharacterTextSplitter')
    def test_split_documents_error_handling(self, mock_splitter_class):
        """Test error handling during document splitting."""
        # Mock the text splitter to raise an exception
        mock_splitter = MagicMock()
        mock_splitter.split_documents.side_effect = Exception("Splitting failed")
        mock_splitter_class.return_value = mock_splitter
        
        # Create new engine to use mocked splitter
        engine = ChunkingEngine()
        
        document = Document(page_content="Test content", metadata={})
        
        with pytest.raises(DocumentProcessingError, match="Failed to chunk documents"):
            engine.split_documents([document])