"""Tests for reranker component."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.reranker import Reranker
from src.exceptions import RetrievalError


class TestReranker:
    """Test cases for Reranker class."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="Machine learning is a subset of artificial intelligence.",
                metadata={"chunk_id": "doc1", "source": "ai.pdf", "page": 1}
            ),
            Document(
                page_content="Deep learning uses neural networks with multiple layers.",
                metadata={"chunk_id": "doc2", "source": "ai.pdf", "page": 2}
            ),
            Document(
                page_content="Natural language processing enables computers to understand text.",
                metadata={"chunk_id": "doc3", "source": "nlp.pdf", "page": 1}
            )
        ]
    
    @pytest.fixture
    def sample_documents_with_scores(self):
        """Create sample documents with existing scores."""
        return [
            Document(
                page_content="Machine learning algorithms learn from data.",
                metadata={
                    "chunk_id": "doc1",
                    "retrieval_score": 0.8,
                    "bm25_score": 0.7,
                    "semantic_score": 0.9
                }
            ),
            Document(
                page_content="Deep learning is a machine learning technique.",
                metadata={
                    "chunk_id": "doc2",
                    "retrieval_score": 0.6,
                    "bm25_score": 0.5,
                    "semantic_score": 0.7
                }
            )
        ]
    
    @patch('src.reranker.CrossEncoder')
    def test_initialization_default_settings(self, mock_cross_encoder_class):
        """Test reranker initialization with default settings."""
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker()
        
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"  # From settings
        assert reranker.top_k == 5  # From settings
        assert reranker.score_threshold is None
        assert reranker.cross_encoder == mock_cross_encoder_instance
        
        mock_cross_encoder_class.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    @patch('src.reranker.CrossEncoder')
    def test_initialization_custom_settings(self, mock_cross_encoder_class):
        """Test reranker initialization with custom settings."""
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker(
            model_name="custom-model",
            top_k=10,
            score_threshold=0.5
        )
        
        assert reranker.model_name == "custom-model"
        assert reranker.top_k == 10
        assert reranker.score_threshold == 0.5
        
        mock_cross_encoder_class.assert_called_once_with("custom-model")
    
    @patch('src.reranker.CrossEncoder')
    def test_initialization_failure(self, mock_cross_encoder_class):
        """Test reranker initialization failure."""
        mock_cross_encoder_class.side_effect = Exception("Model loading failed")
        
        with pytest.raises(RetrievalError, match="Failed to initialize reranker"):
            Reranker()
    
    @patch('src.reranker.CrossEncoder')
    def test_rerank_success(self, mock_cross_encoder_class, sample_documents):
        """Test successful document reranking."""
        # Setup mock
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_instance.predict.return_value = [0.9, 0.7, 0.8]
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker(top_k=2)
        
        # Perform reranking
        results = reranker.rerank("machine learning", sample_documents)
        
        # Verify results
        assert len(results) == 2  # top_k=2
        assert all(isinstance(doc, Document) for doc in results)
        
        # Check that documents are reordered by score (highest first)
        # Original order: [0.9, 0.7, 0.8] -> Reordered: [0.9, 0.8] (top 2)
        assert results[0].metadata['chunk_id'] == 'doc1'  # Highest score (0.9)
        assert results[1].metadata['chunk_id'] == 'doc3'  # Second highest (0.8)
        
        # Check reranking metadata
        for doc in results:
            assert 'rerank_score' in doc.metadata
            assert 'reranker_model' in doc.metadata
            assert doc.metadata['reranker_model'] == reranker.model_name
    
    @patch('src.reranker.CrossEncoder')
    def test_rerank_empty_query(self, mock_cross_encoder_class):
        """Test reranking with empty query."""
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker()
        
        with pytest.raises(RetrievalError, match="Query cannot be empty"):
            reranker.rerank("", [])
        
        with pytest.raises(RetrievalError, match="Query cannot be empty"):
            reranker.rerank("   ", [])
    
    @patch('src.reranker.CrossEncoder')
    def test_rerank_empty_documents(self, mock_cross_encoder_class):
        """Test reranking with empty document list."""
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker()
        
        results = reranker.rerank("test query", [])
        assert results == []
    
    @patch('src.reranker.CrossEncoder')
    def test_rerank_preserve_scores(self, mock_cross_encoder_class, sample_documents_with_scores):
        """Test that original scores are preserved in metadata."""
        # Setup mock
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_instance.predict.return_value = [0.9, 0.8]
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker()
        
        # Perform reranking with score preservation
        results = reranker.rerank("test query", sample_documents_with_scores, preserve_scores=True)
        
        # Check that original scores are preserved
        for doc in results:
            assert 'original_retrieval_score' in doc.metadata
            assert 'original_bm25_score' in doc.metadata
            assert 'original_semantic_score' in doc.metadata
            assert 'rerank_score' in doc.metadata
    
    @patch('src.reranker.CrossEncoder')
    def test_rerank_with_scores(self, mock_cross_encoder_class, sample_documents):
        """Test reranking with score return."""
        # Setup mock
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_instance.predict.return_value = np.array([0.9, 0.7, 0.8])
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker(top_k=3)
        
        # Perform reranking with scores
        results = reranker.rerank_with_scores("machine learning", sample_documents)
        
        # Verify results
        assert len(results) == 3
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert 'rerank_score' in doc.metadata
            assert doc.metadata['rerank_score'] == score
        
        # Check ordering (highest score first)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
    
    @patch('src.reranker.CrossEncoder')
    def test_rerank_with_threshold(self, mock_cross_encoder_class, sample_documents):
        """Test reranking with score threshold."""
        # Setup mock
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_instance.predict.return_value = [0.9, 0.3, 0.8]  # One below threshold
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker(score_threshold=0.5)
        
        # Perform reranking
        results = reranker.rerank("test query", sample_documents)
        
        # Should filter out document with score 0.3
        assert len(results) == 2
        for doc in results:
            assert doc.metadata['rerank_score'] >= 0.5
    
    @patch('src.reranker.CrossEncoder')
    def test_score_pairs(self, mock_cross_encoder_class):
        """Test direct scoring of query-document pairs."""
        # Setup mock
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_instance.predict.return_value = [0.9, 0.7, 0.8]
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker()
        
        # Test scoring
        pairs = [
            ["query1", "document1"],
            ["query2", "document2"],
            ["query3", "document3"]
        ]
        
        scores = reranker.score_pairs(pairs)
        
        assert len(scores) == 3
        assert scores == [0.9, 0.7, 0.8]
        mock_cross_encoder_instance.predict.assert_called_once_with(pairs)
    
    @patch('src.reranker.CrossEncoder')
    def test_score_pairs_empty(self, mock_cross_encoder_class):
        """Test scoring empty pairs list."""
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker()
        
        scores = reranker.score_pairs([])
        assert scores == []
    
    @patch('src.reranker.CrossEncoder')
    def test_filter_by_threshold(self, mock_cross_encoder_class):
        """Test filtering documents by threshold."""
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker()
        
        # Create documents with rerank scores
        docs = [
            Document(page_content="doc1", metadata={"rerank_score": 0.9}),
            Document(page_content="doc2", metadata={"rerank_score": 0.3}),
            Document(page_content="doc3", metadata={"rerank_score": 0.7}),
            Document(page_content="doc4", metadata={})  # No score
        ]
        
        filtered = reranker.filter_by_threshold(docs, 0.5)
        
        assert len(filtered) == 2  # Only docs with scores >= 0.5
        assert filtered[0].metadata["rerank_score"] == 0.9
        assert filtered[1].metadata["rerank_score"] == 0.7
    
    @patch('src.reranker.CrossEncoder')
    def test_get_score_statistics(self, mock_cross_encoder_class):
        """Test getting score statistics."""
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker()
        
        # Create documents with rerank scores
        docs = [
            Document(page_content="doc1", metadata={"rerank_score": 0.9}),
            Document(page_content="doc2", metadata={"rerank_score": 0.5}),
            Document(page_content="doc3", metadata={"rerank_score": 0.7}),
            Document(page_content="doc4", metadata={})  # No score
        ]
        
        stats = reranker.get_score_statistics(docs)
        
        assert stats['count'] == 3  # Only docs with scores
        assert stats['min_score'] == 0.5
        assert stats['max_score'] == 0.9
        assert abs(stats['mean_score'] - 0.7) < 0.01  # (0.9 + 0.5 + 0.7) / 3
        assert stats['std_score'] > 0  # Should have some variance
    
    @patch('src.reranker.CrossEncoder')
    def test_get_score_statistics_empty(self, mock_cross_encoder_class):
        """Test getting statistics with no scored documents."""
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker()
        
        docs = [Document(page_content="doc1", metadata={})]
        stats = reranker.get_score_statistics(docs)
        
        assert stats['count'] == 0
        assert stats['min_score'] == 0.0
        assert stats['max_score'] == 0.0
        assert stats['mean_score'] == 0.0
        assert stats['std_score'] == 0.0
    
    @patch('src.reranker.CrossEncoder')
    def test_update_top_k(self, mock_cross_encoder_class):
        """Test updating top-k parameter."""
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker(top_k=5)
        assert reranker.top_k == 5
        
        reranker.update_top_k(10)
        assert reranker.top_k == 10
        
        # Test invalid top_k
        with pytest.raises(ValueError, match="top_k must be positive"):
            reranker.update_top_k(0)
        
        with pytest.raises(ValueError, match="top_k must be positive"):
            reranker.update_top_k(-1)
    
    @patch('src.reranker.CrossEncoder')
    def test_update_score_threshold(self, mock_cross_encoder_class):
        """Test updating score threshold."""
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker(score_threshold=0.5)
        assert reranker.score_threshold == 0.5
        
        reranker.update_score_threshold(0.7)
        assert reranker.score_threshold == 0.7
        
        reranker.update_score_threshold(None)
        assert reranker.score_threshold is None
    
    @patch('src.reranker.CrossEncoder')
    def test_get_model_info(self, mock_cross_encoder_class):
        """Test getting model information."""
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker(
            model_name="test-model",
            top_k=10,
            score_threshold=0.5
        )
        
        info = reranker.get_model_info()
        
        assert info['model_name'] == "test-model"
        assert info['model_type'] == "CrossEncoder"
        assert info['top_k'] == 10
        assert info['score_threshold'] == 0.5
        assert 'capabilities' in info
        assert 'document_reranking' in info['capabilities']
    
    @patch('src.reranker.CrossEncoder')
    def test_validate_documents_valid(self, mock_cross_encoder_class, sample_documents):
        """Test document validation with valid documents."""
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker()
        
        assert reranker.validate_documents(sample_documents) is True
    
    @patch('src.reranker.CrossEncoder')
    def test_validate_documents_invalid(self, mock_cross_encoder_class):
        """Test document validation with invalid documents."""
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker()
        
        # Test empty list
        assert reranker.validate_documents([]) is False
        
        # Test document with empty content
        invalid_docs = [Document(page_content="", metadata={})]
        assert reranker.validate_documents(invalid_docs) is False
        
        # Test document with whitespace-only content
        whitespace_docs = [Document(page_content="   ", metadata={})]
        assert reranker.validate_documents(whitespace_docs) is False
        
        # Test document with invalid metadata (can't create invalid Document, so test validation logic)
        # Create a valid document and then modify its metadata to be invalid
        valid_doc = Document(page_content="test", metadata={})
        valid_doc.metadata = "not_dict"  # Manually set invalid metadata
        invalid_metadata_docs = [valid_doc]
        assert reranker.validate_documents(invalid_metadata_docs) is False
    
    @patch('src.reranker.CrossEncoder')
    def test_batch_rerank(self, mock_cross_encoder_class, sample_documents):
        """Test batch reranking of multiple query-document sets."""
        # Setup mock
        mock_cross_encoder_instance = Mock()
        # Mock will be called twice, once for each batch
        mock_cross_encoder_instance.predict.side_effect = [
            [0.9, 0.7, 0.8],  # First batch scores
            [0.6, 0.8, 0.5]   # Second batch scores
        ]
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker(top_k=2)
        
        # Prepare batches
        batches = [
            ("query1", sample_documents),
            ("query2", sample_documents)
        ]
        
        # Perform batch reranking
        results = reranker.batch_rerank(batches)
        
        # Verify results
        assert len(results) == 2  # Two batches
        for batch_results in results:
            assert len(batch_results) == 2  # top_k=2
            assert all(isinstance(doc, Document) for doc in batch_results)
    
    @patch('src.reranker.CrossEncoder')
    def test_batch_rerank_empty(self, mock_cross_encoder_class):
        """Test batch reranking with empty batches."""
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker()
        
        results = reranker.batch_rerank([])
        assert results == []
    
    @patch('src.reranker.CrossEncoder')
    def test_rerank_prediction_failure(self, mock_cross_encoder_class, sample_documents):
        """Test reranking when cross-encoder prediction fails."""
        # Setup mock to fail
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_instance.predict.side_effect = Exception("Prediction failed")
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker()
        
        with pytest.raises(RetrievalError, match="Failed to rerank documents"):
            reranker.rerank("test query", sample_documents)
    
    @patch('src.reranker.CrossEncoder')
    def test_rerank_with_scores_prediction_failure(self, mock_cross_encoder_class, sample_documents):
        """Test reranking with scores when prediction fails."""
        # Setup mock to fail
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_instance.predict.side_effect = Exception("Prediction failed")
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker()
        
        with pytest.raises(RetrievalError, match="Failed to rerank documents with scores"):
            reranker.rerank_with_scores("test query", sample_documents)
    
    @patch('src.reranker.CrossEncoder')
    def test_score_pairs_prediction_failure(self, mock_cross_encoder_class):
        """Test score pairs when prediction fails."""
        # Setup mock to fail
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_instance.predict.side_effect = Exception("Prediction failed")
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance
        
        reranker = Reranker()
        
        pairs = [["query", "document"]]
        
        with pytest.raises(RetrievalError, match="Failed to score query-document pairs"):
            reranker.score_pairs(pairs)


class TestRerankerIntegration:
    """Integration tests for Reranker with real components."""
    
    def test_score_ordering_consistency(self):
        """Test that score ordering is consistent."""
        from src.reranker import Reranker
        from unittest.mock import Mock, patch
        
        with patch('src.reranker.CrossEncoder') as mock_cross_encoder_class:
            mock_cross_encoder_instance = Mock()
            mock_cross_encoder_instance.predict.return_value = [0.1, 0.9, 0.5, 0.7]
            mock_cross_encoder_class.return_value = mock_cross_encoder_instance
            
            reranker = Reranker(top_k=4)
            
            docs = [
                Document(page_content=f"Document {i}", metadata={"id": i})
                for i in range(4)
            ]
            
            results = reranker.rerank_with_scores("test", docs)
            
            # Check that results are ordered by score (highest first)
            scores = [score for _, score in results]
            assert scores == [0.9, 0.7, 0.5, 0.1]
            
            # Check that documents are reordered accordingly
            doc_ids = [doc.metadata["id"] for doc, _ in results]
            assert doc_ids == [1, 3, 2, 0]  # Corresponding to scores [0.9, 0.7, 0.5, 0.1]