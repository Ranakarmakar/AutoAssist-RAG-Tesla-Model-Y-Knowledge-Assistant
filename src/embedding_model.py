"""Embedding model component for PDF RAG System."""

from typing import List, Dict, Any, Optional
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import settings
from src.logging_config import get_logger
from src.exceptions import EmbeddingError
from src.cache import get_cache_manager

logger = get_logger(__name__)


class EmbeddingModel:
    """Handles document and query embedding using HuggingFace sentence transformers with caching."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        self.model_name = model_name or settings.embedding_model
        self.logger = logger
        self.cache_manager = get_cache_manager() if settings.enable_caching else None
        
        try:
            self.logger.info("Initializing embedding model", model_name=self.model_name)
            
            # Initialize HuggingFace embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
            )
            
            # Get embedding dimension
            self._embedding_dimension = None
            self._initialize_dimension()
            
            self.logger.info(
                "Embedding model initialized successfully",
                model_name=self.model_name,
                embedding_dimension=self._embedding_dimension,
                caching_enabled=self.cache_manager is not None
            )
            
        except Exception as e:
            error_msg = f"Failed to initialize embedding model {self.model_name}: {str(e)}"
            self.logger.error("Embedding model initialization failed", error=str(e))
            raise EmbeddingError(error_msg) from e
    
    def _initialize_dimension(self) -> None:
        """Initialize the embedding dimension by testing with a sample text."""
        try:
            sample_embedding = self.embeddings.embed_query("test")
            self._embedding_dimension = len(sample_embedding)
        except Exception as e:
            self.logger.warning("Could not determine embedding dimension", error=str(e))
            self._embedding_dimension = 768  # Default for all-mpnet-base-v2
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of document texts with caching.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            self.logger.warning("No texts provided for embedding")
            return []
        
        try:
            self.logger.info("Generating document embeddings", text_count=len(texts))
            
            embeddings = []
            cache_misses = []
            cache_miss_indices = []
            
            # Check cache for each text
            for i, text in enumerate(texts):
                if self.cache_manager:
                    cached_embedding = self.cache_manager.get_cached_embedding(text, self.model_name)
                    if cached_embedding is not None:
                        embeddings.append(cached_embedding.tolist())
                        continue
                
                # Cache miss - need to compute embedding
                embeddings.append(None)  # Placeholder
                cache_misses.append(text)
                cache_miss_indices.append(i)
            
            # Compute embeddings for cache misses
            if cache_misses:
                self.logger.debug(
                    "Computing embeddings for cache misses",
                    total_texts=len(texts),
                    cache_misses=len(cache_misses)
                )
                
                computed_embeddings = self.embeddings.embed_documents(cache_misses)
                
                # Validate computed embeddings
                if len(computed_embeddings) != len(cache_misses):
                    raise EmbeddingError(f"Embedding count mismatch: expected {len(cache_misses)}, got {len(computed_embeddings)}")
                
                # Store in cache and update results
                for idx, (text, embedding) in zip(cache_miss_indices, zip(cache_misses, computed_embeddings)):
                    # Validate embedding dimension
                    if len(embedding) != self._embedding_dimension:
                        raise EmbeddingError(
                            f"Embedding dimension mismatch at index {idx}: "
                            f"expected {self._embedding_dimension}, got {len(embedding)}"
                        )
                    
                    embeddings[idx] = embedding
                    
                    if self.cache_manager:
                        self.cache_manager.cache_embedding(
                            text, 
                            self.model_name, 
                            np.array(embedding)
                        )
            
            self.logger.info(
                "Document embeddings generated successfully",
                text_count=len(texts),
                cache_hits=len(texts) - len(cache_misses),
                cache_misses=len(cache_misses),
                embedding_dimension=len(embeddings[0]) if embeddings else 0
            )
            
            return embeddings
            
        except EmbeddingError:
            raise
        except Exception as e:
            error_msg = f"Failed to generate document embeddings: {str(e)}"
            self.logger.error("Document embedding generation failed", error=str(e))
            raise EmbeddingError(error_msg) from e
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query text with caching.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as a list of floats
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not query or not query.strip():
            raise EmbeddingError("Query text cannot be empty")
        
        try:
            self.logger.info("Generating query embedding", query_length=len(query))
            
            # Check cache first
            if self.cache_manager:
                cached_embedding = self.cache_manager.get_cached_embedding(query, self.model_name)
                if cached_embedding is not None:
                    self.logger.debug("Query embedding cache hit", query_length=len(query))
                    return cached_embedding.tolist()
            
            # Generate embedding using HuggingFace
            embedding = self.embeddings.embed_query(query)
            
            # Validate embedding
            if not embedding:
                raise EmbeddingError("Empty embedding generated for query")
            
            if len(embedding) != self._embedding_dimension:
                raise EmbeddingError(
                    f"Query embedding dimension mismatch: "
                    f"expected {self._embedding_dimension}, got {len(embedding)}"
                )
            
            # Cache the result
            if self.cache_manager:
                self.cache_manager.cache_embedding(
                    query, 
                    self.model_name, 
                    np.array(embedding)
                )
            
            self.logger.info(
                "Query embedding generated successfully",
                embedding_dimension=len(embedding)
            )
            
            return embedding
            
        except EmbeddingError:
            raise
        except Exception as e:
            error_msg = f"Failed to generate query embedding: {str(e)}"
            self.logger.error("Query embedding generation failed", error=str(e))
            raise EmbeddingError(error_msg) from e
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension as integer
        """
        return self._embedding_dimension
    
    def embed_documents_batch(self, documents: List[Document]) -> List[Document]:
        """
        Embed a batch of Document objects and add embeddings to their metadata.
        
        Args:
            documents: List of Document objects to embed
            
        Returns:
            List of Document objects with embeddings added to metadata
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not documents:
            self.logger.warning("No documents provided for batch embedding")
            return []
        
        try:
            self.logger.info("Starting batch document embedding", document_count=len(documents))
            
            # Extract texts from documents
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            embeddings = self.embed_documents(texts)
            
            # Add embeddings to document metadata
            embedded_documents = []
            for doc, embedding in zip(documents, embeddings):
                # Create new document with embedding in metadata
                embedded_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        'embedding': embedding,
                        'embedding_model': self.model_name,
                        'embedding_dimension': len(embedding)
                    }
                )
                embedded_documents.append(embedded_doc)
            
            self.logger.info(
                "Batch document embedding completed",
                document_count=len(embedded_documents)
            )
            
            return embedded_documents
            
        except Exception as e:
            error_msg = f"Failed to embed document batch: {str(e)}"
            self.logger.error("Batch document embedding failed", error=str(e))
            raise EmbeddingError(error_msg) from e
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
            
        Raises:
            EmbeddingError: If embeddings are invalid
        """
        try:
            if len(embedding1) != len(embedding2):
                raise EmbeddingError(
                    f"Embedding dimension mismatch: {len(embedding1)} vs {len(embedding2)}"
                )
            
            # Convert to numpy arrays for computation
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            error_msg = f"Failed to compute similarity: {str(e)}"
            self.logger.error("Similarity computation failed", error=str(e))
            raise EmbeddingError(error_msg) from e
    
    def find_most_similar(
        self, 
        query_embedding: List[float], 
        document_embeddings: List[List[float]], 
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find the most similar document embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors
            top_k: Number of top similar documents to return
            
        Returns:
            List of tuples (index, similarity_score) sorted by similarity (highest first)
            
        Raises:
            EmbeddingError: If computation fails
        """
        try:
            if not document_embeddings:
                return []
            
            similarities = []
            for i, doc_embedding in enumerate(document_embeddings):
                similarity = self.compute_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity (highest first) and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            error_msg = f"Failed to find most similar embeddings: {str(e)}"
            self.logger.error("Similarity search failed", error=str(e))
            raise EmbeddingError(error_msg) from e
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate that an embedding has the correct format and dimension.
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if embedding is valid, False otherwise
        """
        try:
            if not embedding:
                return False
            
            if not isinstance(embedding, list):
                return False
            
            if len(embedding) != self._embedding_dimension:
                return False
            
            # Check that all values are numbers
            for value in embedding:
                if not isinstance(value, (int, float)):
                    return False
                if np.isnan(value) or np.isinf(value):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self._embedding_dimension,
            'model_type': 'HuggingFaceEmbeddings',
            'normalization': True,
            'device': 'cpu'
        }