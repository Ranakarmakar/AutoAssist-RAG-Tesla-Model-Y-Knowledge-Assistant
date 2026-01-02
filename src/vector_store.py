"""Vector storage component for PDF RAG System using FAISS."""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import settings
from src.logging_config import get_logger
from src.exceptions import VectorStoreError

logger = get_logger(__name__)


class VectorStore:
    """Handles vector storage and retrieval using FAISS with LangChain integration."""
    
    def __init__(
        self, 
        embeddings: Embeddings,
        store_path: Optional[str] = None,
        index_name: str = "pdf_rag_index"
    ):
        """
        Initialize the vector store.
        
        Args:
            embeddings: Embedding model instance
            store_path: Path to store the vector index
            index_name: Name of the index
        """
        self.embeddings = embeddings
        self.store_path = Path(store_path or settings.vector_store_path)
        self.index_name = index_name
        self.logger = logger
        
        # Ensure store directory exists
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS vector store
        self.vector_store: Optional[FAISS] = None
        self.document_metadata: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(
            "VectorStore initialized",
            store_path=str(self.store_path),
            index_name=self.index_name
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
            
        Raises:
            VectorStoreError: If adding documents fails
        """
        if not documents:
            self.logger.warning("No documents provided to add")
            return []
        
        try:
            self.logger.info("Adding documents to vector store", document_count=len(documents))
            
            # Extract texts and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            if self.vector_store is None:
                # Create new vector store
                self.logger.info("Creating new FAISS vector store")
                self.vector_store = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
            else:
                # Add to existing vector store
                self.logger.info("Adding to existing FAISS vector store")
                self.vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas
                )
            
            # Generate document IDs
            doc_ids = []
            for i, doc in enumerate(documents):
                doc_id = doc.metadata.get('chunk_id', f"doc_{len(self.document_metadata) + i}")
                doc_ids.append(doc_id)
                
                # Store metadata separately for easy access
                self.document_metadata[doc_id] = {
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'text_length': len(doc.page_content),
                    'word_count': len(doc.page_content.split())
                }
            
            self.logger.info(
                "Documents added successfully",
                document_count=len(documents),
                total_documents=len(self.document_metadata)
            )
            
            return doc_ids
            
        except Exception as e:
            error_msg = f"Failed to add documents to vector store: {str(e)}"
            self.logger.error("Document addition failed", error=str(e))
            raise VectorStoreError(error_msg) from e
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Perform similarity search for a query.
        
        Args:
            query: Query string
            k: Number of documents to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of most similar documents
            
        Raises:
            VectorStoreError: If search fails
        """
        if self.vector_store is None:
            raise VectorStoreError("Vector store not initialized. Add documents first.")
        
        try:
            self.logger.info(
                "Performing similarity search",
                query_length=len(query),
                k=k,
                score_threshold=score_threshold
            )
            
            if score_threshold is not None:
                # Use similarity search with score threshold
                results = self.vector_store.similarity_search_with_score(query, k=k)
                filtered_results = [
                    doc for doc, score in results 
                    if score >= score_threshold
                ]
                
                self.logger.info(
                    "Similarity search completed with threshold",
                    total_results=len(results),
                    filtered_results=len(filtered_results),
                    threshold=score_threshold
                )
                
                return filtered_results
            else:
                # Regular similarity search
                results = self.vector_store.similarity_search(query, k=k)
                
                self.logger.info(
                    "Similarity search completed",
                    results_count=len(results)
                )
                
                return results
                
        except Exception as e:
            error_msg = f"Similarity search failed: {str(e)}"
            self.logger.error("Similarity search failed", error=str(e))
            raise VectorStoreError(error_msg) from e
    
    def similarity_search_with_scores(
        self, 
        query: str, 
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return documents with scores.
        
        Args:
            query: Query string
            k: Number of documents to return
            
        Returns:
            List of tuples (document, similarity_score)
            
        Raises:
            VectorStoreError: If search fails
        """
        if self.vector_store is None:
            raise VectorStoreError("Vector store not initialized. Add documents first.")
        
        try:
            self.logger.info(
                "Performing similarity search with scores",
                query_length=len(query),
                k=k
            )
            
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            self.logger.info(
                "Similarity search with scores completed",
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Similarity search with scores failed: {str(e)}"
            self.logger.error("Similarity search with scores failed", error=str(e))
            raise VectorStoreError(error_msg) from e
    
    def save_index(self, index_name: Optional[str] = None) -> str:
        """
        Save the vector store index to disk.
        
        Args:
            index_name: Name for the saved index
            
        Returns:
            Path to the saved index
            
        Raises:
            VectorStoreError: If saving fails
        """
        if self.vector_store is None:
            raise VectorStoreError("No vector store to save")
        
        try:
            save_name = index_name or self.index_name
            index_path = self.store_path / save_name
            
            self.logger.info("Saving vector store index", index_path=str(index_path))
            
            # Save FAISS index (this creates the directory)
            self.vector_store.save_local(str(index_path))
            
            # Ensure directory exists for metadata
            index_path.mkdir(parents=True, exist_ok=True)
            
            # Save metadata separately
            metadata_path = index_path / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.document_metadata, f)
            
            self.logger.info(
                "Vector store index saved successfully",
                index_path=str(index_path),
                document_count=len(self.document_metadata)
            )
            
            return str(index_path)
            
        except Exception as e:
            error_msg = f"Failed to save vector store index: {str(e)}"
            self.logger.error("Index saving failed", error=str(e))
            raise VectorStoreError(error_msg) from e
    
    def load_index(self, index_name: Optional[str] = None) -> bool:
        """
        Load a vector store index from disk.
        
        Args:
            index_name: Name of the index to load
            
        Returns:
            True if loaded successfully, False otherwise
            
        Raises:
            VectorStoreError: If loading fails
        """
        try:
            load_name = index_name or self.index_name
            index_path = self.store_path / load_name
            
            if not index_path.exists():
                self.logger.warning(
                    "Vector store index not found",
                    index_path=str(index_path)
                )
                return False
            
            self.logger.info("Loading vector store index", index_path=str(index_path))
            
            # Load FAISS index
            self.vector_store = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load metadata
            metadata_path = index_path / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.document_metadata = pickle.load(f)
            else:
                self.logger.warning("Metadata file not found, starting with empty metadata")
                self.document_metadata = {}
            
            self.logger.info(
                "Vector store index loaded successfully",
                document_count=len(self.document_metadata)
            )
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to load vector store index: {str(e)}"
            self.logger.error("Index loading failed", error=str(e))
            raise VectorStoreError(error_msg) from e
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        Delete documents from the vector store.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            True if deletion was successful
            
        Note:
            FAISS doesn't support direct deletion, so this recreates the index
        """
        if not doc_ids:
            return True
        
        try:
            self.logger.info("Deleting documents", doc_ids=doc_ids)
            
            # Remove from metadata
            for doc_id in doc_ids:
                if doc_id in self.document_metadata:
                    del self.document_metadata[doc_id]
            
            # If we have remaining documents, recreate the vector store
            if self.document_metadata:
                remaining_docs = []
                for doc_id, doc_data in self.document_metadata.items():
                    doc = Document(
                        page_content=doc_data['text'],
                        metadata=doc_data['metadata']
                    )
                    remaining_docs.append(doc)
                
                # Recreate vector store with remaining documents
                self.vector_store = None
                self.add_documents(remaining_docs)
            else:
                # No documents left, clear vector store
                self.vector_store = None
            
            self.logger.info(
                "Documents deleted successfully",
                deleted_count=len(doc_ids),
                remaining_count=len(self.document_metadata)
            )
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to delete documents: {str(e)}"
            self.logger.error("Document deletion failed", error=str(e))
            raise VectorStoreError(error_msg) from e
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            Number of documents
        """
        return len(self.document_metadata)
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        if doc_id not in self.document_metadata:
            return None
        
        doc_data = self.document_metadata[doc_id]
        return Document(
            page_content=doc_data['text'],
            metadata=doc_data['metadata']
        )
    
    def list_document_ids(self) -> List[str]:
        """
        Get list of all document IDs in the vector store.
        
        Returns:
            List of document IDs
        """
        return list(self.document_metadata.keys())
    
    def get_store_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store.
        
        Returns:
            Dictionary containing store information
        """
        total_chars = sum(
            doc_data['text_length'] 
            for doc_data in self.document_metadata.values()
        )
        total_words = sum(
            doc_data['word_count'] 
            for doc_data in self.document_metadata.values()
        )
        
        return {
            'store_path': str(self.store_path),
            'index_name': self.index_name,
            'document_count': len(self.document_metadata),
            'total_characters': total_chars,
            'total_words': total_words,
            'avg_document_length': total_chars / len(self.document_metadata) if self.document_metadata else 0,
            'is_initialized': self.vector_store is not None,
            'embedding_model': getattr(self.embeddings, 'model_name', 'unknown')
        }
    
    def clear_store(self) -> None:
        """Clear all documents from the vector store."""
        self.vector_store = None
        self.document_metadata = {}
        self.logger.info("Vector store cleared")
    
    def as_retriever(self, **kwargs) -> Any:
        """
        Get the vector store as a LangChain retriever.
        
        Args:
            **kwargs: Additional arguments for the retriever
            
        Returns:
            LangChain retriever instance
            
        Raises:
            VectorStoreError: If vector store is not initialized
        """
        if self.vector_store is None:
            raise VectorStoreError("Vector store not initialized. Add documents first.")
        
        return self.vector_store.as_retriever(**kwargs)