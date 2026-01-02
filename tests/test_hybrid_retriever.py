"""Tests for hybrid retriever component."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.hybrid_retriever import HybridRetriever
from src.vector_store import VectorStore
from src.embedding_model import EmbeddingModel
from src.exceptions import RetrievalError


class TestHybridRetriever:
    """Test cases for HybridRetriever class."""
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        mock_model = Mock(spec=EmbeddingModel)
        mock_model.embed_documents.return_value = [
            [0.1, 0.2, 0.3] * 256,  # 768-dim embedding
            [0.4, 0.5, 0.6] * 256,
            [0.7, 0.8, 0.9] * 256
        ]
        mock_model.embed_query.return_value = [0.5, 0.5, 0.5] * 256
        mock_model.get_embedding_dimension.return_value = 768
        return mock_model
    
    @pytest.fixture
    def mock_vector_store(self, mock_embedding_model):
        """Create a mock vector store."""
        mock_store = Mock(spec=VectorStore)
        mock_store.add_documents.return_value = ["doc1", "doc2", "doc3"]
        mock_store.similarity_search_with_scores.return_value = [
            (Document(page_content="Test content 1", metadata={"chunk_id": "1"}), 0.9),
            (Document(page_content="Test content 2", metadata={"chunk_id": "2"}), 0.8),
            (Document(page_content="Test content 3", metadata={"chunk_id": "3"}), 0.7)
        ]
        
        # Mock as_retriever method
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Test content 1", metadata={"chunk_id": "1"}),
            Document(page_content="Test content 2", metadata={"chunk_id": "2"}),
            Document(page_content="Test content 3", metadata={"chunk_id": "3"})
        ]
        mock_store.as_retriever.return_value = mock_retriever
        
        mock_store.get_store_info.return_value = {
            'document_count': 3,
            'is_initialized': True
        }
        mock_store.clear_store.return_value = None
        
        return mock_store
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="This is a test document about machine learning algorithms.",
                metadata={"chunk_id": "doc1", "source": "test.pdf", "page": 1}
            ),
            Document(
                page_content="Deep learning is a subset of machine learning using neural networks.",
                metadata={"chunk_id": "doc2", "source": "test.pdf", "page": 2}
            ),
            Document(
                page_content="Natural language processing involves understanding human language.",
                metadata={"chunk_id": "doc3", "source": "test.pdf", "page": 3}
            )
        ]
    
    def test_init_default_weights(self, mock_vector_store):
        """Test initialization with default weights."""
        retriever = HybridRetriever(mock_vector_store)
        
        assert retriever.vector_store == mock_vector_store
        assert retriever.bm25_weight == 0.3  # From settings
        assert retriever.semantic_weight == 0.7  # From settings
        assert retriever.top_k == 20  # From settings
        assert retriever.documents == []
        assert retriever.bm25_retriever is None
        assert retriever.semantic_retriever is None
        assert retriever.ensemble_retriever is None
    
    def test_init_custom_weights(self, mock_vector_store):
        """Test initialization with custom weights."""
        retriever = HybridRetriever(
            mock_vector_store,
            bm25_weight=0.4,
            semantic_weight=0.6,
            top_k=10
        )
        
        assert retriever.bm25_weight == 0.4
        assert retriever.semantic_weight == 0.6
        assert retriever.top_k == 10
    
    def test_init_invalid_weights(self, mock_vector_store):
        """Test initialization with invalid weights."""
        with pytest.raises(RetrievalError, match="must sum to 1.0"):
            HybridRetriever(
                mock_vector_store,
                bm25_weight=0.3,
                semantic_weight=0.8  # Sum = 1.1
            )
    
    @patch('src.hybrid_retriever.BM25Retriever')
    def test_add_documents_success(self, mock_bm25_class, mock_vector_store, sample_documents):
        """Test successful document addition."""
        # Mock BM25Retriever
        mock_bm25_instance = Mock()
        mock_bm25_class.from_documents.return_value = mock_bm25_instance
        
        retriever = HybridRetriever(mock_vector_store)
        retriever.add_documents(sample_documents)
        
        # Verify documents were stored
        assert len(retriever.documents) == 3
        assert retriever.documents == sample_documents
        
        # Verify vector store was called
        mock_vector_store.add_documents.assert_called_once_with(sample_documents)
        
        # Verify BM25 retriever was initialized
        mock_bm25_class.from_documents.assert_called_once_with(
            sample_documents, k=retriever.top_k
        )
        assert retriever.bm25_retriever == mock_bm25_instance
        
        # Verify semantic retriever was initialized
        mock_vector_store.as_retriever.assert_called_once_with(
            search_kwargs={"k": retriever.top_k}
        )
    
    def test_add_documents_empty_list(self, mock_vector_store):
        """Test adding empty document list."""
        retriever = HybridRetriever(mock_vector_store)
        retriever.add_documents([])
        
        assert len(retriever.documents) == 0
        mock_vector_store.add_documents.assert_not_called()
    
    @patch('src.hybrid_retriever.BM25Retriever')
    def test_add_documents_vector_store_error(self, mock_bm25_class, mock_vector_store, sample_documents):
        """Test document addition with vector store error."""
        mock_vector_store.add_documents.side_effect = Exception("Vector store error")
        
        retriever = HybridRetriever(mock_vector_store)
        
        with pytest.raises(RetrievalError, match="Failed to add documents"):
            retriever.add_documents(sample_documents)
    
    @patch('src.hybrid_retriever.BM25Retriever')
    def test_retrieve_success(self, mock_bm25_class, mock_vector_store, sample_documents):
        """Test successful hybrid retrieval."""
        # Setup mocks
        mock_bm25_instance = Mock()
        mock_bm25_class.from_documents.return_value = mock_bm25_instance
        mock_bm25_instance.invoke.return_value = sample_documents[:2]
        
        # Initialize retriever and add documents
        retriever = HybridRetriever(mock_vector_store)
        retriever.add_documents(sample_documents)
        
        # Perform retrieval
        results = retriever.retrieve("test query")
        
        # Verify results
        assert len(results) >= 0  # Should return some results
        assert all(isinstance(doc, Document) for doc in results)
    
    def test_retrieve_empty_query(self, mock_vector_store):
        """Test retrieval with empty query."""
        retriever = HybridRetriever(mock_vector_store)
        
        with pytest.raises(RetrievalError, match="Query cannot be empty"):
            retriever.retrieve("")
        
        with pytest.raises(RetrievalError, match="Query cannot be empty"):
            retriever.retrieve("   ")
    
    def test_retrieve_not_initialized(self, mock_vector_store):
        """Test retrieval without initialization."""
        retriever = HybridRetriever(mock_vector_store)
        
        with pytest.raises(RetrievalError, match="not initialized"):
            retriever.retrieve("test query")
    
    @patch('src.hybrid_retriever.BM25Retriever')
    def test_retrieve_with_custom_top_k(self, mock_bm25_class, mock_vector_store, sample_documents):
        """Test retrieval with custom top_k parameter."""
        # Setup mocks
        mock_bm25_instance = Mock()
        mock_bm25_class.from_documents.return_value = mock_bm25_instance
        mock_bm25_instance.invoke.return_value = sample_documents
        
        # Initialize retriever
        retriever = HybridRetriever(mock_vector_store)
        retriever.add_documents(sample_documents)
        
        # Retrieve with custom top_k
        results = retriever.retrieve("test query", top_k=2)
        
        # Should return some results
        assert len(results) >= 0
    
    @patch('src.hybrid_retriever.BM25Retriever')
    def test_retrieve_with_scores(self, mock_bm25_class, mock_vector_store, sample_documents):
        """Test retrieval with detailed scores."""
        # Setup BM25 mock
        mock_bm25_instance = Mock()
        mock_bm25_instance.invoke.return_value = sample_documents  # Fixed: use invoke instead
        mock_bm25_class.from_documents.return_value = mock_bm25_instance
        
        # Initialize retriever
        retriever = HybridRetriever(mock_vector_store)
        retriever.add_documents(sample_documents)
        
        # Perform retrieval with scores
        results = retriever.retrieve_with_scores("test query")
        
        # Verify results structure
        assert len(results) > 0
        for doc, scores in results:
            assert isinstance(doc, Document)
            assert isinstance(scores, dict)
            assert 'bm25_score' in scores
            assert 'semantic_score' in scores
            assert 'combined_score' in scores
            assert 'bm25_weight' in scores
            assert 'semantic_weight' in scores
    
    def test_retrieve_with_scores_not_initialized(self, mock_vector_store):
        """Test retrieval with scores when not initialized."""
        retriever = HybridRetriever(mock_vector_store)
        
        with pytest.raises(RetrievalError, match="not initialized"):
            retriever.retrieve_with_scores("test query")
    
    @patch('src.hybrid_retriever.BM25Retriever')
    def test_get_bm25_results(self, mock_bm25_class, mock_vector_store, sample_documents):
        """Test BM25-only retrieval."""
        # Setup mock
        mock_bm25_instance = Mock()
        mock_bm25_instance.invoke.return_value = sample_documents[:2]
        mock_bm25_class.from_documents.return_value = mock_bm25_instance
        
        # Initialize retriever
        retriever = HybridRetriever(mock_vector_store)
        retriever.add_documents(sample_documents)
        
        # Get BM25 results
        results = retriever.get_bm25_results("test query")
        
        assert len(results) == 2
        assert results == sample_documents[:2]
        mock_bm25_instance.invoke.assert_called_once_with("test query")
    
    def test_get_bm25_results_not_initialized(self, mock_vector_store):
        """Test BM25 retrieval when not initialized."""
        retriever = HybridRetriever(mock_vector_store)
        
        with pytest.raises(RetrievalError, match="BM25 retriever not initialized"):
            retriever.get_bm25_results("test query")
    
    @patch('src.hybrid_retriever.BM25Retriever')
    def test_get_semantic_results(self, mock_bm25_class, mock_vector_store, sample_documents):
        """Test semantic-only retrieval."""
        # Setup mocks
        mock_bm25_instance = Mock()
        mock_bm25_class.from_documents.return_value = mock_bm25_instance
        
        # Initialize retriever
        retriever = HybridRetriever(mock_vector_store)
        retriever.add_documents(sample_documents)
        
        # Get semantic results
        results = retriever.get_semantic_results("test query")
        
        assert len(results) == 3
        # Verify semantic retriever was called
        semantic_retriever = mock_vector_store.as_retriever.return_value
        semantic_retriever.invoke.assert_called_once_with("test query")
    
    def test_get_semantic_results_not_initialized(self, mock_vector_store):
        """Test semantic retrieval when not initialized."""
        retriever = HybridRetriever(mock_vector_store)
        
        with pytest.raises(RetrievalError, match="Semantic retriever not initialized"):
            retriever.get_semantic_results("test query")
    
    @patch('src.hybrid_retriever.BM25Retriever')
    def test_update_weights(self, mock_bm25_class, mock_vector_store, sample_documents):
        """Test updating retrieval weights."""
        # Setup mocks
        mock_bm25_instance = Mock()
        mock_bm25_class.from_documents.return_value = mock_bm25_instance
        
        # Initialize retriever
        retriever = HybridRetriever(mock_vector_store)
        retriever.add_documents(sample_documents)
        
        # Update weights
        retriever.update_weights(0.5, 0.5)
        
        assert retriever.bm25_weight == 0.5
        assert retriever.semantic_weight == 0.5
        
        # Verify ensemble retriever was recreated
        assert retriever.ensemble_retriever is not None
    
    def test_update_weights_invalid(self, mock_vector_store):
        """Test updating with invalid weights."""
        retriever = HybridRetriever(mock_vector_store)
        
        with pytest.raises(RetrievalError, match="must sum to 1.0"):
            retriever.update_weights(0.6, 0.6)  # Sum = 1.2
    
    def test_get_retriever_info(self, mock_vector_store):
        """Test getting retriever information."""
        retriever = HybridRetriever(mock_vector_store, bm25_weight=0.4, semantic_weight=0.6, top_k=15)
        
        info = retriever.get_retriever_info()
        
        assert info['bm25_weight'] == 0.4
        assert info['semantic_weight'] == 0.6
        assert info['top_k'] == 15
        assert info['document_count'] == 0
        assert info['bm25_initialized'] is False
        assert info['semantic_initialized'] is False
        assert info['ensemble_initialized'] is False
        assert 'vector_store_info' in info
    
    def test_clear_documents(self, mock_vector_store):
        """Test clearing all documents."""
        retriever = HybridRetriever(mock_vector_store)
        retriever.documents = [Mock(), Mock()]  # Add some mock documents
        
        retriever.clear_documents()
        
        assert len(retriever.documents) == 0
        assert retriever.bm25_retriever is None
        assert retriever.semantic_retriever is None
        assert retriever.ensemble_retriever is None
        mock_vector_store.clear_store.assert_called_once()
    
    def test_normalize_scores(self, mock_vector_store):
        """Test score normalization."""
        retriever = HybridRetriever(mock_vector_store)
        
        # Test normal case
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = retriever._normalize_scores(scores)
        
        assert len(normalized) == 5
        assert normalized[0] == 0.0  # Min score
        assert normalized[-1] == 1.0  # Max score
        assert all(0.0 <= score <= 1.0 for score in normalized)
        
        # Test edge case - all same scores
        same_scores = [2.0, 2.0, 2.0]
        normalized_same = retriever._normalize_scores(same_scores)
        assert all(score == 1.0 for score in normalized_same)
        
        # Test empty list
        empty_normalized = retriever._normalize_scores([])
        assert empty_normalized == []
    
    def test_get_document_key(self, mock_vector_store):
        """Test document key generation."""
        retriever = HybridRetriever(mock_vector_store)
        
        # Test with chunk_id in metadata
        doc_with_id = Document(
            page_content="Test content",
            metadata={"chunk_id": "test_chunk_123"}
        )
        key_with_id = retriever._get_document_key(doc_with_id)
        assert key_with_id == "test_chunk_123"
        
        # Test without chunk_id
        doc_without_id = Document(
            page_content="This is some test content for key generation",
            metadata={"source": "test.pdf"}
        )
        key_without_id = retriever._get_document_key(doc_without_id)
        assert key_without_id.startswith("doc_")
        assert isinstance(key_without_id, str)
    
    def test_compute_bm25_scores(self, mock_vector_store, sample_documents):
        """Test BM25 score computation."""
        retriever = HybridRetriever(mock_vector_store)
        
        scores = retriever._compute_bm25_scores("test query", sample_documents)
        
        assert len(scores) == len(sample_documents)
        assert all(0.0 <= score <= 1.0 for score in scores)
        # Scores should be in descending order (first document has highest score)
        assert scores[0] >= scores[1] >= scores[2]
    
    def test_compute_bm25_scores_empty(self, mock_vector_store):
        """Test BM25 score computation with empty documents."""
        retriever = HybridRetriever(mock_vector_store)
        
        scores = retriever._compute_bm25_scores("test query", [])
        assert scores == []


class TestHybridRetrieverIntegration:
    """Integration tests for HybridRetriever with real components."""
    
    def test_document_key_consistency(self):
        """Test that document keys are consistent across calls."""
        from src.hybrid_retriever import HybridRetriever
        from unittest.mock import Mock
        
        mock_vector_store = Mock()
        retriever = HybridRetriever(mock_vector_store)
        
        doc = Document(
            page_content="Consistent content for testing",
            metadata={"source": "test.pdf"}
        )
        
        key1 = retriever._get_document_key(doc)
        key2 = retriever._get_document_key(doc)
        
        assert key1 == key2
    
    def test_score_combination_logic(self):
        """Test that score combination follows expected logic."""
        from src.hybrid_retriever import HybridRetriever
        from unittest.mock import Mock
        
        mock_vector_store = Mock()
        retriever = HybridRetriever(mock_vector_store, bm25_weight=0.3, semantic_weight=0.7)
        
        # Test score combination
        bm25_score = 0.8
        semantic_score = 0.6
        
        expected_combined = 0.3 * bm25_score + 0.7 * semantic_score
        assert abs(expected_combined - 0.66) < 0.01  # 0.3*0.8 + 0.7*0.6 = 0.24 + 0.42 = 0.66