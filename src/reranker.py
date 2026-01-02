"""Reranking component for PDF RAG System using cross-encoder models."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from src.config import settings
from src.logging_config import get_logger
from src.exceptions import RetrievalError

logger = get_logger(__name__)


class Reranker:
    """Reranks retrieval results using cross-encoder models for improved relevance scoring."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ):
        """
        Initialize the reranker.
        
        Args:
            model_name: Cross-encoder model name to use
            top_k: Number of top results to return after reranking
            score_threshold: Minimum relevance score threshold
        """
        self.model_name = model_name or settings.reranker_model
        self.top_k = top_k or settings.rerank_top_k
        self.score_threshold = score_threshold
        self.logger = logger
        
        try:
            self.logger.info(
                "Initializing cross-encoder reranker",
                model_name=self.model_name,
                top_k=self.top_k
            )
            
            # Initialize cross-encoder model
            self.cross_encoder = CrossEncoder(self.model_name)
            
            self.logger.info("Reranker initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize reranker: {str(e)}"
            self.logger.error("Reranker initialization failed", error=str(e))
            raise RetrievalError(error_msg) from e
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document],
        top_k: Optional[int] = None,
        preserve_scores: bool = True
    ) -> List[Document]:
        """
        Rerank documents based on query-document relevance using cross-encoder.
        
        Args:
            query: Query string
            documents: List of documents to rerank
            top_k: Number of top documents to return (overrides default)
            preserve_scores: Whether to preserve original scores in metadata
            
        Returns:
            List of reranked documents
            
        Raises:
            RetrievalError: If reranking fails
        """
        if not query or not query.strip():
            raise RetrievalError("Query cannot be empty")
        
        if not documents:
            self.logger.warning("No documents provided for reranking")
            return []
        
        try:
            k = top_k or self.top_k
            self.logger.info(
                "Starting document reranking",
                query=query,
                document_count=len(documents),
                top_k=k
            )
            
            # Prepare query-document pairs for cross-encoder
            query_doc_pairs = []
            for doc in documents:
                query_doc_pairs.append([query, doc.page_content])
            
            # Get relevance scores from cross-encoder
            relevance_scores = self.cross_encoder.predict(query_doc_pairs)
            
            # Convert to list if numpy array
            if isinstance(relevance_scores, np.ndarray):
                relevance_scores = relevance_scores.tolist()
            
            # Create scored documents with metadata preservation
            scored_documents = []
            for doc, score in zip(documents, relevance_scores):
                # Create new document with reranking score
                new_metadata = doc.metadata.copy()
                
                # Preserve original scores if requested
                if preserve_scores:
                    # Store original retrieval scores if they exist
                    if 'retrieval_score' in new_metadata:
                        new_metadata['original_retrieval_score'] = new_metadata['retrieval_score']
                    if 'bm25_score' in new_metadata:
                        new_metadata['original_bm25_score'] = new_metadata['bm25_score']
                    if 'semantic_score' in new_metadata:
                        new_metadata['original_semantic_score'] = new_metadata['semantic_score']
                
                # Add reranking score
                new_metadata['rerank_score'] = float(score)
                new_metadata['reranker_model'] = self.model_name
                
                reranked_doc = Document(
                    page_content=doc.page_content,
                    metadata=new_metadata
                )
                
                scored_documents.append((reranked_doc, float(score)))
            
            # Sort by relevance score (highest first)
            scored_documents.sort(key=lambda x: x[1], reverse=True)
            
            # Apply score threshold if specified
            if self.score_threshold is not None:
                scored_documents = [
                    (doc, score) for doc, score in scored_documents
                    if score >= self.score_threshold
                ]
            
            # Return top-k documents
            final_documents = [doc for doc, _ in scored_documents[:k]]
            
            self.logger.info(
                "Document reranking completed",
                query=query,
                original_count=len(documents),
                reranked_count=len(final_documents),
                top_score=scored_documents[0][1] if scored_documents else 0.0
            )
            
            return final_documents
            
        except Exception as e:
            error_msg = f"Failed to rerank documents: {str(e)}"
            self.logger.error("Document reranking failed", error=str(e), query=query)
            raise RetrievalError(error_msg) from e
    
    def rerank_with_scores(
        self, 
        query: str, 
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents and return with relevance scores.
        
        Args:
            query: Query string
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of tuples (document, relevance_score)
            
        Raises:
            RetrievalError: If reranking fails
        """
        if not query or not query.strip():
            raise RetrievalError("Query cannot be empty")
        
        if not documents:
            return []
        
        try:
            k = top_k or self.top_k
            self.logger.info(
                "Starting document reranking with scores",
                query=query,
                document_count=len(documents),
                top_k=k
            )
            
            # Prepare query-document pairs
            query_doc_pairs = []
            for doc in documents:
                query_doc_pairs.append([query, doc.page_content])
            
            # Get relevance scores
            relevance_scores = self.cross_encoder.predict(query_doc_pairs)
            
            # Convert to list if numpy array
            if isinstance(relevance_scores, np.ndarray):
                relevance_scores = relevance_scores.tolist()
            
            # Create scored documents
            scored_documents = []
            for doc, score in zip(documents, relevance_scores):
                # Add reranking score to metadata
                new_metadata = doc.metadata.copy()
                new_metadata['rerank_score'] = float(score)
                new_metadata['reranker_model'] = self.model_name
                
                reranked_doc = Document(
                    page_content=doc.page_content,
                    metadata=new_metadata
                )
                
                scored_documents.append((reranked_doc, float(score)))
            
            # Sort by relevance score (highest first)
            scored_documents.sort(key=lambda x: x[1], reverse=True)
            
            # Apply score threshold if specified
            if self.score_threshold is not None:
                scored_documents = [
                    (doc, score) for doc, score in scored_documents
                    if score >= self.score_threshold
                ]
            
            # Return top-k documents with scores
            final_results = scored_documents[:k]
            
            self.logger.info(
                "Document reranking with scores completed",
                query=query,
                original_count=len(documents),
                reranked_count=len(final_results),
                top_score=final_results[0][1] if final_results else 0.0
            )
            
            return final_results
            
        except Exception as e:
            error_msg = f"Failed to rerank documents with scores: {str(e)}"
            self.logger.error("Document reranking with scores failed", error=str(e), query=query)
            raise RetrievalError(error_msg) from e
    
    def score_pairs(self, query_doc_pairs: List[List[str]]) -> List[float]:
        """
        Score query-document pairs directly.
        
        Args:
            query_doc_pairs: List of [query, document] pairs
            
        Returns:
            List of relevance scores
            
        Raises:
            RetrievalError: If scoring fails
        """
        if not query_doc_pairs:
            return []
        
        try:
            self.logger.info("Scoring query-document pairs", pair_count=len(query_doc_pairs))
            
            scores = self.cross_encoder.predict(query_doc_pairs)
            
            # Convert to list if numpy array
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            
            self.logger.info("Query-document pair scoring completed", score_count=len(scores))
            
            return scores
            
        except Exception as e:
            error_msg = f"Failed to score query-document pairs: {str(e)}"
            self.logger.error("Query-document pair scoring failed", error=str(e))
            raise RetrievalError(error_msg) from e
    
    def filter_by_threshold(
        self, 
        documents: List[Document], 
        threshold: float
    ) -> List[Document]:
        """
        Filter documents by reranking score threshold.
        
        Args:
            documents: List of documents with rerank_score in metadata
            threshold: Minimum score threshold
            
        Returns:
            List of documents above threshold
        """
        filtered_docs = []
        
        for doc in documents:
            rerank_score = doc.metadata.get('rerank_score')
            if rerank_score is not None and rerank_score >= threshold:
                filtered_docs.append(doc)
        
        self.logger.info(
            "Documents filtered by threshold",
            original_count=len(documents),
            filtered_count=len(filtered_docs),
            threshold=threshold
        )
        
        return filtered_docs
    
    def get_score_statistics(self, documents: List[Document]) -> Dict[str, float]:
        """
        Get statistics about reranking scores in a document list.
        
        Args:
            documents: List of documents with rerank_score in metadata
            
        Returns:
            Dictionary with score statistics
        """
        scores = []
        for doc in documents:
            rerank_score = doc.metadata.get('rerank_score')
            if rerank_score is not None:
                scores.append(rerank_score)
        
        if not scores:
            return {
                'count': 0,
                'min_score': 0.0,
                'max_score': 0.0,
                'mean_score': 0.0,
                'std_score': 0.0
            }
        
        scores_array = np.array(scores)
        
        return {
            'count': len(scores),
            'min_score': float(np.min(scores_array)),
            'max_score': float(np.max(scores_array)),
            'mean_score': float(np.mean(scores_array)),
            'std_score': float(np.std(scores_array))
        }
    
    def update_top_k(self, new_top_k: int) -> None:
        """
        Update the top-k parameter for reranking.
        
        Args:
            new_top_k: New top-k value
        """
        if new_top_k <= 0:
            raise ValueError("top_k must be positive")
        
        old_top_k = self.top_k
        self.top_k = new_top_k
        
        self.logger.info(
            "Updated top-k parameter",
            old_top_k=old_top_k,
            new_top_k=new_top_k
        )
    
    def update_score_threshold(self, new_threshold: Optional[float]) -> None:
        """
        Update the score threshold for filtering.
        
        Args:
            new_threshold: New threshold value (None to disable)
        """
        old_threshold = self.score_threshold
        self.score_threshold = new_threshold
        
        self.logger.info(
            "Updated score threshold",
            old_threshold=old_threshold,
            new_threshold=new_threshold
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the reranker configuration.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'model_type': 'CrossEncoder',
            'top_k': self.top_k,
            'score_threshold': self.score_threshold,
            'capabilities': [
                'document_reranking',
                'relevance_scoring',
                'score_filtering',
                'metadata_preservation'
            ]
        }
    
    def validate_documents(self, documents: List[Document]) -> bool:
        """
        Validate that documents are suitable for reranking.
        
        Args:
            documents: List of documents to validate
            
        Returns:
            True if documents are valid for reranking
        """
        if not documents:
            return False
        
        for doc in documents:
            if not doc.page_content or not doc.page_content.strip():
                return False
            
            if not isinstance(doc.metadata, dict):
                return False
        
        return True
    
    def batch_rerank(
        self,
        query_document_batches: List[Tuple[str, List[Document]]],
        top_k: Optional[int] = None
    ) -> List[List[Document]]:
        """
        Rerank multiple query-document batches efficiently.
        
        Args:
            query_document_batches: List of (query, documents) tuples
            top_k: Number of top documents to return per batch
            
        Returns:
            List of reranked document lists
            
        Raises:
            RetrievalError: If batch reranking fails
        """
        if not query_document_batches:
            return []
        
        try:
            k = top_k or self.top_k
            self.logger.info(
                "Starting batch reranking",
                batch_count=len(query_document_batches),
                top_k=k
            )
            
            results = []
            for query, documents in query_document_batches:
                reranked_docs = self.rerank(query, documents, top_k=k)
                results.append(reranked_docs)
            
            self.logger.info("Batch reranking completed", batch_count=len(results))
            
            return results
            
        except Exception as e:
            error_msg = f"Failed to perform batch reranking: {str(e)}"
            self.logger.error("Batch reranking failed", error=str(e))
            raise RetrievalError(error_msg) from e