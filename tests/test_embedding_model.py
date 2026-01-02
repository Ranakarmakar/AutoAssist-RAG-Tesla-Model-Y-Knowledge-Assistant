"""Tests for embedding model."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.embedding_model import EmbeddingModel
from src.exceptions import EmbeddingError


class TestEmbeddingModel:
    """Test cases for EmbeddingModel class."""
    
    def setup_method(self):
        """Set up test fixtures with mocked embeddings."""
        with patch('src.embedding_model.HuggingFaceEmbeddings') as mock_embeddings_class:
            # Mock the HuggingFace embeddings
            self.mock_embeddings = MagicMock()
            mock_embeddings_class.return_value = self.mock_embeddings
            
            # Mock embedding responses
            self.mock_embeddings.embed_query.return_value = [0.1] * 768
            self.mock_embeddings.embed_documents.return_value = [[0.1] * 768, [0.2] * 768]
            
            # Create embedding model
            self.model = EmbeddingModel()
    
    def test_initialization_default_model(self):
        """Test embedding model initialization with default model."""
        with patch('src.embedding_model.settings') as mock_settings:
            mock_settings.embedding_model = "test-model"
            
            with patch('src.embedding_model.HuggingFaceEmbeddings') as mock_class:
                mock_embeddings = MagicMock()
                mock_embeddings.embed_query.return_value = [0.1] * 768
                mock_class.return_value = mock_embeddings
                
                model = EmbeddingModel()
                
                assert model.model_name == "test-model"
                assert model.get_embedding_dimension() == 768
    
    def test_initialization_custom_model(self):
        """Test embedding model initialization with custom model."""
        with patch('src.embedding_model.HuggingFaceEmbeddings') as mock_class:
            mock_embeddings = MagicMock()
            mock_embeddings.embed_query.return_value = [0.1] * 512
            mock_class.return_value = mock_embeddings
            
            model = EmbeddingModel(model_name="custom-model")
            
            assert model.model_name == "custom-model"
            assert model.get_embedding_dimension() == 512
    
    def test_embed_query_success(self):
        """Test successful query embedding."""
        query = "test query"
        expected_embedding = [0.1, 0.2, 0.3]
        
        # Reset mock to clear initialization calls
        self.mock_embeddings.embed_query.reset_mock()
        self.mock_embeddings.embed_query.return_value = expected_embedding
        self.model._embedding_dimension = 3
        
        result = self.model.embed_query(query)
        
        assert result == expected_embedding
        self.mock_embeddings.embed_query.assert_called_once_with(query)
    
    def test_embed_query_empty_text(self):
        """Test query embedding with empty text."""
        with pytest.raises(EmbeddingError, match="Query text cannot be empty"):
            self.model.embed_query("")
        
        with pytest.raises(EmbeddingError, match="Query text cannot be empty"):
            self.model.embed_query("   ")
    
    def test_embed_query_dimension_mismatch(self):
        """Test query embedding with dimension mismatch."""
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2]  # Wrong dimension
        self.model._embedding_dimension = 3
        
        with pytest.raises(EmbeddingError, match="dimension mismatch"):
            self.model.embed_query("test")
    
    def test_embed_documents_success(self):
        """Test successful document embedding."""
        texts = ["doc1", "doc2"]
        expected_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        
        self.mock_embeddings.embed_documents.return_value = expected_embeddings
        self.model._embedding_dimension = 2
        
        result = self.model.embed_documents(texts)
        
        assert result == expected_embeddings
        self.mock_embeddings.embed_documents.assert_called_once_with(texts)
    
    def test_embed_documents_empty_list(self):
        """Test document embedding with empty list."""
        result = self.model.embed_documents([])
        assert result == []
    
    def test_embed_documents_count_mismatch(self):
        """Test document embedding with count mismatch."""
        texts = ["doc1", "doc2"]
        self.mock_embeddings.embed_documents.return_value = [[0.1, 0.2]]  # Wrong count
        
        with pytest.raises(EmbeddingError, match="Embedding count mismatch"):
            self.model.embed_documents(texts)
    
    def test_embed_documents_batch_success(self):
        """Test successful batch document embedding."""
        documents = [
            Document(page_content="doc1", metadata={"id": 1}),
            Document(page_content="doc2", metadata={"id": 2})
        ]
        
        expected_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        self.mock_embeddings.embed_documents.return_value = expected_embeddings
        self.model._embedding_dimension = 2
        
        result = self.model.embed_documents_batch(documents)
        
        assert len(result) == 2
        assert result[0].metadata['embedding'] == [0.1, 0.2]
        assert result[1].metadata['embedding'] == [0.3, 0.4]
        assert result[0].metadata['id'] == 1  # Original metadata preserved
        assert result[1].metadata['id'] == 2
    
    def test_embed_documents_batch_empty(self):
        """Test batch embedding with empty list."""
        result = self.model.embed_documents_batch([])
        assert result == []
    
    def test_compute_similarity_success(self):
        """Test successful similarity computation."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]
        
        # These should be orthogonal (similarity = 0)
        similarity = self.model.compute_similarity(embedding1, embedding2)
        assert abs(similarity - 0.0) < 1e-6
        
        # Same vectors should have similarity = 1
        similarity = self.model.compute_similarity(embedding1, embedding1)
        assert abs(similarity - 1.0) < 1e-6
    
    def test_compute_similarity_dimension_mismatch(self):
        """Test similarity computation with dimension mismatch."""
        embedding1 = [1.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]
        
        with pytest.raises(EmbeddingError, match="dimension mismatch"):
            self.model.compute_similarity(embedding1, embedding2)
    
    def test_compute_similarity_zero_vectors(self):
        """Test similarity computation with zero vectors."""
        embedding1 = [0.0, 0.0, 0.0]
        embedding2 = [1.0, 0.0, 0.0]
        
        similarity = self.model.compute_similarity(embedding1, embedding2)
        assert similarity == 0.0
    
    def test_find_most_similar_success(self):
        """Test finding most similar embeddings."""
        query_embedding = [1.0, 0.0]
        document_embeddings = [
            [1.0, 0.0],  # Perfect match
            [0.0, 1.0],  # Orthogonal
            [0.5, 0.5],  # Partial match
        ]
        
        results = self.model.find_most_similar(query_embedding, document_embeddings, top_k=2)
        
        assert len(results) == 2
        assert results[0][0] == 0  # First document (perfect match)
        assert results[0][1] > results[1][1]  # Higher similarity
    
    def test_find_most_similar_empty_documents(self):
        """Test finding similar embeddings with empty document list."""
        query_embedding = [1.0, 0.0]
        results = self.model.find_most_similar(query_embedding, [], top_k=5)
        assert results == []
    
    def test_validate_embedding_valid(self):
        """Test embedding validation with valid embedding."""
        valid_embedding = [0.1, 0.2, 0.3]
        self.model._embedding_dimension = 3
        
        assert self.model.validate_embedding(valid_embedding) is True
    
    def test_validate_embedding_invalid_dimension(self):
        """Test embedding validation with invalid dimension."""
        invalid_embedding = [0.1, 0.2]  # Wrong dimension
        self.model._embedding_dimension = 3
        
        assert self.model.validate_embedding(invalid_embedding) is False
    
    def test_validate_embedding_invalid_values(self):
        """Test embedding validation with invalid values."""
        self.model._embedding_dimension = 3
        
        # Test with NaN
        invalid_embedding = [0.1, float('nan'), 0.3]
        assert self.model.validate_embedding(invalid_embedding) is False
        
        # Test with infinity
        invalid_embedding = [0.1, float('inf'), 0.3]
        assert self.model.validate_embedding(invalid_embedding) is False
        
        # Test with non-numeric values
        invalid_embedding = [0.1, "invalid", 0.3]
        assert self.model.validate_embedding(invalid_embedding) is False
    
    def test_validate_embedding_empty(self):
        """Test embedding validation with empty embedding."""
        assert self.model.validate_embedding([]) is False
        assert self.model.validate_embedding(None) is False
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.model.get_model_info()
        
        assert 'model_name' in info
        assert 'embedding_dimension' in info
        assert 'model_type' in info
        assert info['model_type'] == 'HuggingFaceEmbeddings'
        assert info['normalization'] is True
        assert info['device'] == 'cpu'
    
    @patch('src.embedding_model.HuggingFaceEmbeddings')
    def test_initialization_failure(self, mock_embeddings_class):
        """Test embedding model initialization failure."""
        mock_embeddings_class.side_effect = Exception("Model loading failed")
        
        with pytest.raises(EmbeddingError, match="Failed to initialize embedding model"):
            EmbeddingModel()
    
    def test_embed_query_failure(self):
        """Test query embedding failure."""
        self.mock_embeddings.embed_query.side_effect = Exception("Embedding failed")
        
        with pytest.raises(EmbeddingError, match="Failed to generate query embedding"):
            self.model.embed_query("test")
    
    def test_embed_documents_failure(self):
        """Test document embedding failure."""
        self.mock_embeddings.embed_documents.side_effect = Exception("Embedding failed")
        
        with pytest.raises(EmbeddingError, match="Failed to generate document embeddings"):
            self.model.embed_documents(["test"])