"""Tests for vector store."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.vector_store import VectorStore
from src.exceptions import VectorStoreError


class TestVectorStore:
    """Test cases for VectorStore class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.store_path = Path(self.temp_dir)
        
        # Mock embeddings
        self.mock_embeddings = MagicMock()
        self.mock_embeddings.model_name = "test-model"
        
        # Create vector store
        self.vector_store = VectorStore(
            embeddings=self.mock_embeddings,
            store_path=str(self.store_path),
            index_name="test_index"
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test vector store initialization."""
        assert self.vector_store.embeddings == self.mock_embeddings
        assert self.vector_store.store_path == self.store_path
        assert self.vector_store.index_name == "test_index"
        assert self.vector_store.vector_store is None
        assert self.vector_store.document_metadata == {}
        assert self.store_path.exists()
    
    @patch('src.vector_store.FAISS')
    def test_add_documents_new_store(self, mock_faiss_class):
        """Test adding documents to a new vector store."""
        # Mock FAISS
        mock_faiss_instance = MagicMock()
        mock_faiss_class.from_texts.return_value = mock_faiss_instance
        
        # Test documents
        documents = [
            Document(page_content="test content 1", metadata={"chunk_id": "chunk_1", "page": 1}),
            Document(page_content="test content 2", metadata={"chunk_id": "chunk_2", "page": 2})
        ]
        
        # Add documents
        doc_ids = self.vector_store.add_documents(documents)
        
        # Verify
        assert len(doc_ids) == 2
        assert "chunk_1" in doc_ids
        assert "chunk_2" in doc_ids
        assert self.vector_store.vector_store == mock_faiss_instance
        assert len(self.vector_store.document_metadata) == 2
        
        # Verify FAISS was called correctly
        mock_faiss_class.from_texts.assert_called_once_with(
            texts=["test content 1", "test content 2"],
            embedding=self.mock_embeddings,
            metadatas=[{"chunk_id": "chunk_1", "page": 1}, {"chunk_id": "chunk_2", "page": 2}]
        )
    
    @patch('src.vector_store.FAISS')
    def test_add_documents_existing_store(self, mock_faiss_class):
        """Test adding documents to an existing vector store."""
        # Setup existing store
        mock_faiss_instance = MagicMock()
        self.vector_store.vector_store = mock_faiss_instance
        
        # Test documents
        documents = [
            Document(page_content="new content", metadata={"chunk_id": "chunk_3", "page": 3})
        ]
        
        # Add documents
        doc_ids = self.vector_store.add_documents(documents)
        
        # Verify
        assert len(doc_ids) == 1
        assert "chunk_3" in doc_ids
        
        # Verify add_texts was called
        mock_faiss_instance.add_texts.assert_called_once_with(
            texts=["new content"],
            metadatas=[{"chunk_id": "chunk_3", "page": 3}]
        )
    
    def test_add_documents_empty_list(self):
        """Test adding empty document list."""
        doc_ids = self.vector_store.add_documents([])
        assert doc_ids == []
    
    def test_similarity_search_no_store(self):
        """Test similarity search without initialized store."""
        with pytest.raises(VectorStoreError, match="Vector store not initialized"):
            self.vector_store.similarity_search("test query")
    
    def test_similarity_search_success(self):
        """Test successful similarity search."""
        # Setup mock vector store
        mock_faiss_instance = MagicMock()
        mock_results = [
            Document(page_content="result 1", metadata={"score": 0.9}),
            Document(page_content="result 2", metadata={"score": 0.8})
        ]
        mock_faiss_instance.similarity_search.return_value = mock_results
        self.vector_store.vector_store = mock_faiss_instance
        
        # Perform search
        results = self.vector_store.similarity_search("test query", k=2)
        
        # Verify
        assert len(results) == 2
        assert results == mock_results
        mock_faiss_instance.similarity_search.assert_called_once_with("test query", k=2)
    
    def test_similarity_search_with_scores_success(self):
        """Test similarity search with scores."""
        # Setup mock vector store
        mock_faiss_instance = MagicMock()
        mock_results = [
            (Document(page_content="result 1"), 0.9),
            (Document(page_content="result 2"), 0.8)
        ]
        mock_faiss_instance.similarity_search_with_score.return_value = mock_results
        self.vector_store.vector_store = mock_faiss_instance
        
        # Perform search
        results = self.vector_store.similarity_search_with_scores("test query", k=2)
        
        # Verify
        assert len(results) == 2
        assert results == mock_results
        mock_faiss_instance.similarity_search_with_score.assert_called_once_with("test query", k=2)
    
    def test_similarity_search_with_threshold(self):
        """Test similarity search with score threshold."""
        # Setup mock vector store
        mock_faiss_instance = MagicMock()
        mock_results = [
            (Document(page_content="result 1"), 0.9),
            (Document(page_content="result 2"), 0.6),
            (Document(page_content="result 3"), 0.4)
        ]
        mock_faiss_instance.similarity_search_with_score.return_value = mock_results
        self.vector_store.vector_store = mock_faiss_instance
        
        # Perform search with threshold
        results = self.vector_store.similarity_search("test query", k=3, score_threshold=0.7)
        
        # Verify only results above threshold are returned
        assert len(results) == 1
        assert results[0].page_content == "result 1"
    
    @patch('src.vector_store.pickle')
    def test_save_index_success(self, mock_pickle):
        """Test successful index saving."""
        # Setup mock vector store
        mock_faiss_instance = MagicMock()
        self.vector_store.vector_store = mock_faiss_instance
        self.vector_store.document_metadata = {"doc1": {"text": "content"}}
        
        # Save index
        saved_path = self.vector_store.save_index()
        
        # Verify
        expected_path = str(self.store_path / "test_index")
        assert saved_path == expected_path
        mock_faiss_instance.save_local.assert_called_once_with(expected_path)
    
    def test_save_index_no_store(self):
        """Test saving index without vector store."""
        with pytest.raises(VectorStoreError, match="No vector store to save"):
            self.vector_store.save_index()
    
    @patch('src.vector_store.FAISS')
    @patch('src.vector_store.pickle')
    def test_load_index_success(self, mock_pickle, mock_faiss_class):
        """Test successful index loading."""
        # Create fake index directory
        index_path = self.store_path / "test_index"
        index_path.mkdir()
        metadata_path = index_path / "metadata.pkl"
        metadata_path.touch()
        
        # Mock FAISS and pickle
        mock_faiss_instance = MagicMock()
        mock_faiss_class.load_local.return_value = mock_faiss_instance
        mock_pickle.load.return_value = {"doc1": {"text": "content"}}
        
        # Load index
        result = self.vector_store.load_index()
        
        # Verify
        assert result is True
        assert self.vector_store.vector_store == mock_faiss_instance
        assert self.vector_store.document_metadata == {"doc1": {"text": "content"}}
    
    def test_load_index_not_found(self):
        """Test loading non-existent index."""
        result = self.vector_store.load_index("nonexistent")
        assert result is False
    
    def test_get_document_count(self):
        """Test getting document count."""
        self.vector_store.document_metadata = {"doc1": {}, "doc2": {}}
        assert self.vector_store.get_document_count() == 2
    
    def test_get_document_by_id_found(self):
        """Test retrieving document by ID."""
        self.vector_store.document_metadata = {
            "doc1": {
                "text": "test content",
                "metadata": {"page": 1}
            }
        }
        
        doc = self.vector_store.get_document_by_id("doc1")
        
        assert doc is not None
        assert doc.page_content == "test content"
        assert doc.metadata == {"page": 1}
    
    def test_get_document_by_id_not_found(self):
        """Test retrieving non-existent document."""
        doc = self.vector_store.get_document_by_id("nonexistent")
        assert doc is None
    
    def test_list_document_ids(self):
        """Test listing document IDs."""
        self.vector_store.document_metadata = {"doc1": {}, "doc2": {}, "doc3": {}}
        ids = self.vector_store.list_document_ids()
        assert set(ids) == {"doc1", "doc2", "doc3"}
    
    def test_get_store_info(self):
        """Test getting store information."""
        self.vector_store.document_metadata = {
            "doc1": {"text_length": 100, "word_count": 20},
            "doc2": {"text_length": 200, "word_count": 40}
        }
        
        info = self.vector_store.get_store_info()
        
        assert info['document_count'] == 2
        assert info['total_characters'] == 300
        assert info['total_words'] == 60
        assert info['avg_document_length'] == 150
        assert info['is_initialized'] is False
        assert info['embedding_model'] == "test-model"
    
    def test_clear_store(self):
        """Test clearing the store."""
        # Setup some data
        self.vector_store.vector_store = MagicMock()
        self.vector_store.document_metadata = {"doc1": {}}
        
        # Clear store
        self.vector_store.clear_store()
        
        # Verify
        assert self.vector_store.vector_store is None
        assert self.vector_store.document_metadata == {}
    
    def test_as_retriever_success(self):
        """Test getting retriever from vector store."""
        # Setup mock vector store
        mock_faiss_instance = MagicMock()
        mock_retriever = MagicMock()
        mock_faiss_instance.as_retriever.return_value = mock_retriever
        self.vector_store.vector_store = mock_faiss_instance
        
        # Get retriever
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Verify
        assert retriever == mock_retriever
        mock_faiss_instance.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
    
    def test_as_retriever_no_store(self):
        """Test getting retriever without initialized store."""
        with pytest.raises(VectorStoreError, match="Vector store not initialized"):
            self.vector_store.as_retriever()
    
    @patch('src.vector_store.FAISS')
    def test_delete_documents_recreate_store(self, mock_faiss_class):
        """Test deleting documents by recreating store."""
        # Setup initial state
        self.vector_store.document_metadata = {
            "doc1": {"text": "content1", "metadata": {"id": "doc1"}},
            "doc2": {"text": "content2", "metadata": {"id": "doc2"}},
            "doc3": {"text": "content3", "metadata": {"id": "doc3"}}
        }
        self.vector_store.vector_store = MagicMock()
        
        # Mock FAISS for recreation
        mock_faiss_instance = MagicMock()
        mock_faiss_class.from_texts.return_value = mock_faiss_instance
        
        # Delete documents
        result = self.vector_store.delete_documents(["doc1", "doc3"])
        
        # Verify
        assert result is True
        assert "doc1" not in self.vector_store.document_metadata
        assert "doc2" in self.vector_store.document_metadata
        assert "doc3" not in self.vector_store.document_metadata
        
        # Verify store was recreated with remaining documents
        mock_faiss_class.from_texts.assert_called_once()
    
    def test_delete_documents_clear_store(self):
        """Test deleting all documents clears store."""
        # Setup initial state
        self.vector_store.document_metadata = {
            "doc1": {"text": "content1", "metadata": {"id": "doc1"}}
        }
        self.vector_store.vector_store = MagicMock()
        
        # Delete all documents
        result = self.vector_store.delete_documents(["doc1"])
        
        # Verify
        assert result is True
        assert len(self.vector_store.document_metadata) == 0
        assert self.vector_store.vector_store is None
    
    def test_delete_documents_empty_list(self):
        """Test deleting empty document list."""
        result = self.vector_store.delete_documents([])
        assert result is True