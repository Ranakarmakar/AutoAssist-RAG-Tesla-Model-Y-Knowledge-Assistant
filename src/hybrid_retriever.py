"""Hybrid retrieval system for PDF RAG System using BM25 and semantic search."""

from typing import List, Dict, Any, Optional, Tuple
import math
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import settings
from src.logging_config import get_logger
from src.exceptions import RetrievalError
from src.vector_store import VectorStore

logger = get_logger(__name__)


class SimpleEnsembleRetriever:
    """Simple ensemble retriever that combines results from multiple retrievers."""
    
    def __init__(self, retrievers: List[BaseRetriever], weights: List[float]):
        """
        Initialize ensemble retriever.
        
        Args:
            retrievers: List of retrievers to ensemble
            weights: List of weights for each retriever
        """
        if len(retrievers) != len(weights):
            raise ValueError("Number of retrievers must match number of weights")
        
        if abs(sum(weights) - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        
        self.retrievers = retrievers
        self.weights = weights
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents using ensemble of retrievers.
        
        Args:
            query: Query string
            
        Returns:
            List of documents from ensemble retrieval
        """
        all_docs = []
        doc_scores = {}
        
        # Get results from each retriever
        for retriever, weight in zip(self.retrievers, self.weights):
            # Use invoke method for LangChain retrievers
            if hasattr(retriever, 'invoke'):
                docs = retriever.invoke(query)
            elif hasattr(retriever, 'get_relevant_documents'):
                docs = retriever.get_relevant_documents(query)
            else:
                # Fallback to _get_relevant_documents
                docs = retriever._get_relevant_documents(query)
            
            # Assign scores based on rank and weight
            for rank, doc in enumerate(docs):
                doc_key = self._get_doc_key(doc)
                
                # Use reciprocal rank fusion scoring
                score = weight / (rank + 1)
                
                if doc_key in doc_scores:
                    doc_scores[doc_key]['score'] += score
                else:
                    doc_scores[doc_key] = {'doc': doc, 'score': score}
        
        # Sort by combined score and return documents
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [item['doc'] for item in sorted_docs]
    
    def _get_doc_key(self, doc: Document) -> str:
        """Generate a unique key for a document."""
        if 'chunk_id' in doc.metadata:
            return doc.metadata['chunk_id']
        return f"doc_{hash(doc.page_content[:100])}"


logger = get_logger(__name__)


class HybridRetriever:
    """Combines BM25 keyword search with semantic vector search for hybrid retrieval."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        top_k: Optional[int] = None
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_store: Initialized vector store for semantic search
            bm25_weight: Weight for BM25 retrieval (default from settings)
            semantic_weight: Weight for semantic retrieval (default from settings)
            top_k: Number of documents to retrieve (default from settings)
        """
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight or settings.bm25_weight
        self.semantic_weight = semantic_weight or settings.semantic_weight
        self.top_k = top_k or settings.retrieval_top_k
        self.logger = logger
        
        # Validate weights
        total_weight = self.bm25_weight + self.semantic_weight
        if abs(total_weight - 1.0) > 0.01:
            raise RetrievalError(
                f"BM25 and semantic weights must sum to 1.0, got {total_weight}"
            )
        
        # Initialize retrievers
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.semantic_retriever: Optional[BaseRetriever] = None
        self.ensemble_retriever: Optional[SimpleEnsembleRetriever] = None
        
        # Document corpus for BM25
        self.documents: List[Document] = []
        
        self.logger.info(
            "HybridRetriever initialized",
            bm25_weight=self.bm25_weight,
            semantic_weight=self.semantic_weight,
            top_k=self.top_k
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to both BM25 and semantic retrievers.
        
        Args:
            documents: List of documents to add
            
        Raises:
            RetrievalError: If adding documents fails
        """
        if not documents:
            self.logger.warning("No documents provided to add")
            return
        
        try:
            self.logger.info("Adding documents to hybrid retriever", document_count=len(documents))
            
            # Store documents for BM25
            self.documents.extend(documents)
            
            # Add to vector store (semantic search)
            self.vector_store.add_documents(documents)
            
            # Reinitialize retrievers with new documents
            self._initialize_retrievers()
            
            self.logger.info(
                "Documents added successfully",
                total_documents=len(self.documents)
            )
            
        except Exception as e:
            error_msg = f"Failed to add documents to hybrid retriever: {str(e)}"
            self.logger.error("Document addition failed", error=str(e))
            raise RetrievalError(error_msg) from e
    
    def _initialize_retrievers(self) -> None:
        """Initialize BM25 and semantic retrievers."""
        try:
            self.logger.info("Initializing retrievers")
            
            # Initialize BM25 retriever
            if self.documents:
                self.bm25_retriever = BM25Retriever.from_documents(
                    self.documents,
                    k=self.top_k
                )
                self.logger.info("BM25 retriever initialized", document_count=len(self.documents))
            
            # Initialize semantic retriever
            self.semantic_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.top_k}
            )
            self.logger.info("Semantic retriever initialized")
            
            # Initialize ensemble retriever
            if self.bm25_retriever and self.semantic_retriever:
                self.ensemble_retriever = SimpleEnsembleRetriever(
                    retrievers=[self.bm25_retriever, self.semantic_retriever],
                    weights=[self.bm25_weight, self.semantic_weight]
                )
                self.logger.info("Ensemble retriever initialized")
            
        except Exception as e:
            error_msg = f"Failed to initialize retrievers: {str(e)}"
            self.logger.error("Retriever initialization failed", error=str(e))
            raise RetrievalError(error_msg) from e
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Perform hybrid retrieval combining BM25 and semantic search.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve (overrides default)
            
        Returns:
            List of retrieved documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        if not query or not query.strip():
            raise RetrievalError("Query cannot be empty")
        
        if not self.ensemble_retriever:
            raise RetrievalError("Hybrid retriever not initialized. Add documents first.")
        
        try:
            k = top_k or self.top_k
            self.logger.info("Performing hybrid retrieval", query=query, top_k=k)
            
            # Use ensemble retriever
            results = self.ensemble_retriever.get_relevant_documents(query)
            
            # Limit results to requested number
            if len(results) > k:
                results = results[:k]
            
            self.logger.info(
                "Hybrid retrieval completed",
                query=query,
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Hybrid retrieval failed: {str(e)}"
            self.logger.error("Hybrid retrieval failed", error=str(e), query=query)
            raise RetrievalError(error_msg) from e
    
    def retrieve_with_scores(
        self, 
        query: str, 
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, Dict[str, float]]]:
        """
        Perform hybrid retrieval with detailed scoring information.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of tuples (document, scores_dict)
            
        Raises:
            RetrievalError: If retrieval fails
        """
        if not query or not query.strip():
            raise RetrievalError("Query cannot be empty")
        
        if not self.bm25_retriever or not self.semantic_retriever:
            raise RetrievalError("Retrievers not initialized. Add documents first.")
        
        try:
            k = top_k or self.top_k
            self.logger.info("Performing hybrid retrieval with scores", query=query, top_k=k)
            
            # Get results from both retrievers
            bm25_results = self.bm25_retriever.invoke(query)
            semantic_results = self.vector_store.similarity_search_with_scores(query, k=k)
            
            # Combine and score results
            combined_results = self._combine_results_with_scores(
                query, bm25_results, semantic_results, k
            )
            
            self.logger.info(
                "Hybrid retrieval with scores completed",
                query=query,
                results_count=len(combined_results)
            )
            
            return combined_results
            
        except Exception as e:
            error_msg = f"Hybrid retrieval with scores failed: {str(e)}"
            self.logger.error("Hybrid retrieval with scores failed", error=str(e), query=query)
            raise RetrievalError(error_msg) from e
    
    def _combine_results_with_scores(
        self,
        query: str,
        bm25_results: List[Document],
        semantic_results: List[Tuple[Document, float]],
        top_k: int
    ) -> List[Tuple[Document, Dict[str, float]]]:
        """
        Combine BM25 and semantic results with normalized scoring.
        
        Args:
            query: Original query
            bm25_results: Results from BM25 retriever
            semantic_results: Results from semantic retriever with scores
            top_k: Number of top results to return
            
        Returns:
            Combined results with scores
        """
        # Create document index for efficient lookup
        doc_scores = {}
        
        # Process BM25 results (assign rank-based scores)
        bm25_scores = self._compute_bm25_scores(query, bm25_results)
        for i, (doc, score) in enumerate(zip(bm25_results, bm25_scores)):
            doc_key = self._get_document_key(doc)
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {
                    'document': doc,
                    'bm25_score': 0.0,
                    'semantic_score': 0.0,
                    'combined_score': 0.0
                }
            doc_scores[doc_key]['bm25_score'] = score
        
        # Process semantic results
        semantic_scores = [score for _, score in semantic_results]
        normalized_semantic_scores = self._normalize_scores(semantic_scores)
        
        for (doc, _), norm_score in zip(semantic_results, normalized_semantic_scores):
            doc_key = self._get_document_key(doc)
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {
                    'document': doc,
                    'bm25_score': 0.0,
                    'semantic_score': 0.0,
                    'combined_score': 0.0
                }
            doc_scores[doc_key]['semantic_score'] = norm_score
        
        # Compute combined scores
        for doc_key, scores in doc_scores.items():
            scores['combined_score'] = (
                self.bm25_weight * scores['bm25_score'] +
                self.semantic_weight * scores['semantic_score']
            )
        
        # Sort by combined score and return top_k
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:top_k]
        
        # Format results
        final_results = []
        for result in sorted_results:
            doc = result['document']
            scores = {
                'bm25_score': result['bm25_score'],
                'semantic_score': result['semantic_score'],
                'combined_score': result['combined_score'],
                'bm25_weight': self.bm25_weight,
                'semantic_weight': self.semantic_weight
            }
            final_results.append((doc, scores))
        
        return final_results
    
    def _compute_bm25_scores(self, query: str, documents: List[Document]) -> List[float]:
        """
        Compute BM25 scores for documents (simplified implementation).
        
        Args:
            query: Query string
            documents: List of documents
            
        Returns:
            List of BM25 scores
        """
        if not documents:
            return []
        
        # Simple rank-based scoring (since BM25Retriever doesn't expose scores)
        # Higher rank = higher score
        scores = []
        for i in range(len(documents)):
            # Exponential decay based on rank
            score = math.exp(-i * 0.1)  # Score decreases with rank
            scores.append(score)
        
        # Normalize scores
        return self._normalize_scores(scores)
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            scores: List of scores to normalize
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _get_document_key(self, doc: Document) -> str:
        """
        Generate a unique key for a document.
        
        Args:
            doc: Document to generate key for
            
        Returns:
            Unique document key
        """
        # Use chunk_id if available, otherwise use hash of content
        if 'chunk_id' in doc.metadata:
            return doc.metadata['chunk_id']
        
        # Use first 100 characters of content as key
        content_key = doc.page_content[:100].replace('\n', ' ').strip()
        return f"doc_{hash(content_key)}"
    
    def get_bm25_results(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Get results from BM25 retriever only.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            BM25 retrieval results
            
        Raises:
            RetrievalError: If BM25 retrieval fails
        """
        if not self.bm25_retriever:
            raise RetrievalError("BM25 retriever not initialized")
        
        try:
            k = top_k or self.top_k
            self.logger.info("Performing BM25 retrieval", query=query, top_k=k)
            
            # Use invoke method for BM25Retriever
            results = self.bm25_retriever.invoke(query)
            if len(results) > k:
                results = results[:k]
            
            self.logger.info("BM25 retrieval completed", results_count=len(results))
            return results
            
        except Exception as e:
            error_msg = f"BM25 retrieval failed: {str(e)}"
            self.logger.error("BM25 retrieval failed", error=str(e))
            raise RetrievalError(error_msg) from e
    
    def get_semantic_results(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Get results from semantic retriever only.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            Semantic retrieval results
            
        Raises:
            RetrievalError: If semantic retrieval fails
        """
        if not self.semantic_retriever:
            raise RetrievalError("Semantic retriever not initialized")
        
        try:
            k = top_k or self.top_k
            self.logger.info("Performing semantic retrieval", query=query, top_k=k)
            
            # Use invoke method for semantic retriever
            results = self.semantic_retriever.invoke(query)
            if len(results) > k:
                results = results[:k]
            
            self.logger.info("Semantic retrieval completed", results_count=len(results))
            return results
            
        except Exception as e:
            error_msg = f"Semantic retrieval failed: {str(e)}"
            self.logger.error("Semantic retrieval failed", error=str(e))
            raise RetrievalError(error_msg) from e
    
    def update_weights(self, bm25_weight: float, semantic_weight: float) -> None:
        """
        Update retrieval weights and reinitialize ensemble retriever.
        
        Args:
            bm25_weight: New BM25 weight
            semantic_weight: New semantic weight
            
        Raises:
            RetrievalError: If weight update fails
        """
        # Validate weights
        total_weight = bm25_weight + semantic_weight
        if abs(total_weight - 1.0) > 0.01:
            raise RetrievalError(
                f"BM25 and semantic weights must sum to 1.0, got {total_weight}"
            )
        
        try:
            self.logger.info(
                "Updating retrieval weights",
                old_bm25_weight=self.bm25_weight,
                old_semantic_weight=self.semantic_weight,
                new_bm25_weight=bm25_weight,
                new_semantic_weight=semantic_weight
            )
            
            self.bm25_weight = bm25_weight
            self.semantic_weight = semantic_weight
            
            # Reinitialize ensemble retriever with new weights
            if self.bm25_retriever and self.semantic_retriever:
                self.ensemble_retriever = SimpleEnsembleRetriever(
                    retrievers=[self.bm25_retriever, self.semantic_retriever],
                    weights=[self.bm25_weight, self.semantic_weight]
                )
            
            self.logger.info("Retrieval weights updated successfully")
            
        except Exception as e:
            error_msg = f"Failed to update retrieval weights: {str(e)}"
            self.logger.error("Weight update failed", error=str(e))
            raise RetrievalError(error_msg) from e
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """
        Get information about the hybrid retriever configuration.
        
        Returns:
            Dictionary containing retriever information
        """
        return {
            'bm25_weight': self.bm25_weight,
            'semantic_weight': self.semantic_weight,
            'top_k': self.top_k,
            'document_count': len(self.documents),
            'bm25_initialized': self.bm25_retriever is not None,
            'semantic_initialized': self.semantic_retriever is not None,
            'ensemble_initialized': self.ensemble_retriever is not None,
            'vector_store_info': self.vector_store.get_store_info()
        }
    
    def clear_documents(self) -> None:
        """Clear all documents from the hybrid retriever."""
        self.documents = []
        self.bm25_retriever = None
        self.semantic_retriever = None
        self.ensemble_retriever = None
        self.vector_store.clear_store()
        self.logger.info("Hybrid retriever cleared")