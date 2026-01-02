"""LangGraph workflow orchestration for PDF RAG System."""

from typing import Dict, List, Any, Optional, TypedDict
import asyncio
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from src.config import settings
from src.logging_config import get_logger
from src.exceptions import WorkflowError
from src.document_processor import DocumentProcessor
from src.chunking_engine import ChunkingEngine
from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore
from src.query_rewriter import QueryRewriter
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.answer_generator import AnswerGenerator
from src.cache import get_cache_manager
from src.metrics import get_metrics_collector, time_operation, increment_counter, record_error
import hashlib
import time

logger = get_logger(__name__)


class RAGWorkflowState(TypedDict):
    """State object for RAG workflow using LangGraph."""
    
    # Input
    query: str
    pdf_path: Optional[str]
    
    # Processing stages
    documents: List[Document]
    chunks: List[Document]
    enhanced_query: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    
    # Output
    answer: str
    citations: List[Dict[str, Any]]
    confidence: str
    
    # Metadata and control
    error: Optional[str]
    stage: str
    metadata: Dict[str, Any]
    
    # Messages for debugging
    messages: List[str]


class RAGWorkflow:
    """LangGraph-based workflow orchestrator for the RAG pipeline."""
    
    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        enable_caching: bool = True
    ):
        """
        Initialize the RAG workflow.
        
        Args:
            vector_store_path: Path for vector store persistence
            enable_caching: Whether to enable component caching
        """
        self.vector_store_path = vector_store_path or settings.vector_store_path
        self.enable_caching = enable_caching
        self.logger = logger
        self.cache_manager = get_cache_manager() if settings.enable_caching else None
        self.metrics_collector = get_metrics_collector()
        
        # Initialize components
        self._initialize_components()
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        self.logger.info(
            "RAG workflow initialized",
            vector_store_path=self.vector_store_path,
            enable_caching=enable_caching,
            caching_enabled=self.cache_manager is not None
        )
    
    def _initialize_components(self) -> None:
        """Initialize all RAG pipeline components."""
        try:
            self.logger.info("Initializing RAG workflow components")
            
            # Core processing components
            self.doc_processor = DocumentProcessor()
            self.chunker = ChunkingEngine()
            self.embedding_model = EmbeddingModel()
            self.vector_store = VectorStore(
                self.embedding_model.embeddings,
                store_path=self.vector_store_path
            )
            
            # Query processing components
            self.query_rewriter = QueryRewriter()
            self.hybrid_retriever = HybridRetriever(self.vector_store)
            self.reranker = Reranker()
            self.answer_generator = AnswerGenerator()
            
            self.logger.info("All RAG workflow components initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize RAG workflow components: {str(e)}"
            self.logger.error("Component initialization failed", error=str(e))
            raise WorkflowError(error_msg) from e
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create workflow graph
        workflow = StateGraph(RAGWorkflowState)
        
        # Add workflow nodes
        workflow.add_node("process_documents", self._process_documents_node)
        workflow.add_node("chunk_documents", self._chunk_documents_node)
        workflow.add_node("store_embeddings", self._store_embeddings_node)
        workflow.add_node("enhance_query", self._enhance_query_node)
        workflow.add_node("retrieve_documents", self._retrieve_documents_node)
        workflow.add_node("rerank_documents", self._rerank_documents_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define workflow edges and conditions
        workflow.set_entry_point("process_documents")
        
        # Document processing flow
        workflow.add_conditional_edges(
            "process_documents",
            self._should_continue_after_processing,
            {
                "continue": "chunk_documents",
                "error": "handle_error",
                "skip_processing": "enhance_query"  # If no new documents to process
            }
        )
        
        workflow.add_conditional_edges(
            "chunk_documents",
            self._should_continue_after_chunking,
            {
                "continue": "store_embeddings",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "store_embeddings",
            self._should_continue_after_storage,
            {
                "continue": "enhance_query",
                "error": "handle_error"
            }
        )
        
        # Query processing flow
        workflow.add_conditional_edges(
            "enhance_query",
            self._should_continue_after_enhancement,
            {
                "continue": "retrieve_documents",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "retrieve_documents",
            self._should_continue_after_retrieval,
            {
                "continue": "rerank_documents",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "rerank_documents",
            self._should_continue_after_reranking,
            {
                "continue": "generate_answer",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    # Workflow nodes
    
    def _process_documents_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Process PDF documents if provided."""
        with time_operation("workflow_process_documents"):
            try:
                self.logger.info("Processing documents", stage="process_documents")
                
                state["stage"] = "process_documents"
                state["messages"].append("Starting document processing")
                
                if not state.get("pdf_path"):
                    # No new documents to process, skip to query enhancement
                    state["documents"] = []
                    state["messages"].append("No PDF provided, skipping document processing")
                    return state
                
                # Process PDF document (limit to first 5 pages for testing)
                pdf_path = state["pdf_path"]
                if not Path(pdf_path).exists():
                    raise WorkflowError(f"PDF file not found: {pdf_path}")
                
                documents = self.doc_processor.process_pdf_file(pdf_path)
                
                # Limit to first 5 pages for faster testing
                if len(documents) > 5:
                    documents = documents[:5]
                    state["messages"].append(f"Limited processing to first 5 pages for performance")
                
                state["documents"] = documents
                state["metadata"]["document_count"] = len(documents)
                state["messages"].append(f"Processed {len(documents)} pages from PDF")
                
                # Record metrics
                increment_counter("workflow_documents_processed", len(documents))
                
                self.logger.info(
                    "Document processing completed",
                    document_count=len(documents),
                    pdf_path=pdf_path
                )
                
                return state
                
            except Exception as e:
                error_msg = f"Document processing failed: {str(e)}"
                self.logger.error("Document processing node failed", error=str(e))
                record_error("WorkflowError", "process_documents")
                state["error"] = error_msg
                state["messages"].append(f"Error: {error_msg}")
                return state
    
    def _chunk_documents_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Chunk processed documents."""
        with time_operation("workflow_chunk_documents"):
            try:
                self.logger.info("Chunking documents", stage="chunk_documents")
                
                state["stage"] = "chunk_documents"
                state["messages"].append("Starting document chunking")
                
                documents = state["documents"]
                if not documents:
                    state["chunks"] = []
                    state["messages"].append("No documents to chunk")
                    return state
                
                # Chunk documents
                chunks = self.chunker.split_documents(documents)
                
                state["chunks"] = chunks
                state["metadata"]["chunk_count"] = len(chunks)
                state["messages"].append(f"Created {len(chunks)} chunks from documents")
                
                # Record metrics
                increment_counter("workflow_chunks_created", len(chunks))
                
                self.logger.info(
                    "Document chunking completed",
                    chunk_count=len(chunks),
                    original_documents=len(documents)
                )
                
                return state
                
            except Exception as e:
                error_msg = f"Document chunking failed: {str(e)}"
                self.logger.error("Document chunking node failed", error=str(e))
                record_error("WorkflowError", "chunk_documents")
                state["error"] = error_msg
                state["messages"].append(f"Error: {error_msg}")
                return state
    
    def _store_embeddings_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Store document embeddings in vector store."""
        with time_operation("workflow_store_embeddings"):
            try:
                self.logger.info("Storing embeddings", stage="store_embeddings")
                
                state["stage"] = "store_embeddings"
                state["messages"].append("Starting embedding storage")
                
                chunks = state["chunks"]
                if not chunks:
                    state["messages"].append("No chunks to store")
                    return state
                
                # Add chunks to vector store and hybrid retriever
                self.vector_store.add_documents(chunks)
                self.hybrid_retriever.add_documents(chunks)
                
                stored_count = self.vector_store.get_document_count()
                state["metadata"]["stored_documents"] = stored_count
                state["messages"].append(f"Stored {len(chunks)} chunks in vector store")
                
                # Record metrics
                increment_counter("workflow_embeddings_stored", len(chunks))
                
                self.logger.info(
                    "Embedding storage completed",
                    chunks_added=len(chunks),
                    total_stored=stored_count
                )
                
                return state
                
            except Exception as e:
                error_msg = f"Embedding storage failed: {str(e)}"
                self.logger.error("Embedding storage node failed", error=str(e))
                record_error("WorkflowError", "store_embeddings")
                state["error"] = error_msg
                state["messages"].append(f"Error: {error_msg}")
                return state
    
    def _enhance_query_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Enhance the user query."""
        with time_operation("workflow_enhance_query"):
            try:
                self.logger.info("Enhancing query", stage="enhance_query")
                
                state["stage"] = "enhance_query"
                state["messages"].append("Starting query enhancement")
                
                query = state["query"]
                if not query or not query.strip():
                    raise WorkflowError("Query cannot be empty")
                
                # Enhance query
                enhanced_query = self.query_rewriter.enhance_query(query)
                
                state["enhanced_query"] = enhanced_query
                state["metadata"]["original_query"] = query
                state["messages"].append("Query enhanced successfully")
                
                # Record metrics
                increment_counter("workflow_queries_enhanced")
                
                self.logger.info(
                    "Query enhancement completed",
                    original_query=query,
                    enhanced_length=len(enhanced_query)
                )
                
                return state
                
            except Exception as e:
                error_msg = f"Query enhancement failed: {str(e)}"
                self.logger.error("Query enhancement node failed", error=str(e))
                record_error("WorkflowError", "enhance_query")
                state["error"] = error_msg
                state["messages"].append(f"Error: {error_msg}")
                return state
    
    def _retrieve_documents_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Retrieve relevant documents using hybrid search."""
        with time_operation("workflow_retrieve_documents"):
            try:
                self.logger.info("Retrieving documents", stage="retrieve_documents")
                
                state["stage"] = "retrieve_documents"
                state["messages"].append("Starting document retrieval")
                
                enhanced_query = state["enhanced_query"]
                
                # Check if vector store has any documents
                doc_count = self.vector_store.get_document_count()
                if doc_count == 0:
                    # No documents available for retrieval
                    state["retrieved_docs"] = []
                    state["metadata"]["retrieved_count"] = 0
                    state["messages"].append("No documents available for retrieval")
                    
                    self.logger.info(
                        "Document retrieval skipped - no documents available",
                        query=enhanced_query
                    )
                    
                    return state
                
                # Retrieve documents using hybrid search
                retrieved_docs = self.hybrid_retriever.retrieve(
                    enhanced_query,
                    top_k=settings.retrieval_top_k
                )
                
                state["retrieved_docs"] = retrieved_docs
                state["metadata"]["retrieved_count"] = len(retrieved_docs)
                state["messages"].append(f"Retrieved {len(retrieved_docs)} relevant documents")
                
                # Record metrics
                increment_counter("workflow_documents_retrieved", len(retrieved_docs))
                
                self.logger.info(
                    "Document retrieval completed",
                    query=enhanced_query,
                    retrieved_count=len(retrieved_docs)
                )
                
                return state
                
            except Exception as e:
                error_msg = f"Document retrieval failed: {str(e)}"
                self.logger.error("Document retrieval node failed", error=str(e))
                record_error("WorkflowError", "retrieve_documents")
                state["error"] = error_msg
                state["messages"].append(f"Error: {error_msg}")
                return state
    
    def _rerank_documents_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Rerank retrieved documents for relevance."""
        with time_operation("workflow_rerank_documents"):
            try:
                self.logger.info("Reranking documents", stage="rerank_documents")
                
                state["stage"] = "rerank_documents"
                state["messages"].append("Starting document reranking")
                
                enhanced_query = state["enhanced_query"]
                retrieved_docs = state["retrieved_docs"]
                
                if not retrieved_docs:
                    state["reranked_docs"] = []
                    state["messages"].append("No documents to rerank")
                    return state
                
                # Rerank documents
                reranked_docs = self.reranker.rerank(enhanced_query, retrieved_docs)
                
                state["reranked_docs"] = reranked_docs
                state["metadata"]["reranked_count"] = len(reranked_docs)
                state["messages"].append(f"Reranked to top {len(reranked_docs)} documents")
                
                # Record metrics
                increment_counter("workflow_documents_reranked", len(reranked_docs))
                
                self.logger.info(
                    "Document reranking completed",
                    original_count=len(retrieved_docs),
                    reranked_count=len(reranked_docs)
                )
                
                return state
                
            except Exception as e:
                error_msg = f"Document reranking failed: {str(e)}"
                self.logger.error("Document reranking node failed", error=str(e))
                record_error("WorkflowError", "rerank_documents")
                state["error"] = error_msg
                state["messages"].append(f"Error: {error_msg}")
                return state
    
    def _generate_answer_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Generate final answer with citations."""
        with time_operation("workflow_generate_answer"):
            try:
                self.logger.info("Generating answer", stage="generate_answer")
                
                state["stage"] = "generate_answer"
                state["messages"].append("Starting answer generation")
                
                original_query = state["metadata"]["original_query"]
                reranked_docs = state["reranked_docs"]
                
                # Generate answer
                result = self.answer_generator.generate_answer(original_query, reranked_docs)
                
                state["answer"] = result["answer"]
                state["citations"] = result["citations"]
                state["confidence"] = result["confidence"]
                state["metadata"]["answer_metadata"] = {
                    "sufficient_information": result["sufficient_information"],
                    "source_count": result["source_count"],
                    "model_used": result["model_used"]
                }
                state["messages"].append("Answer generated successfully")
                
                # Record metrics
                increment_counter("workflow_answers_generated")
                
                self.logger.info(
                    "Answer generation completed",
                    answer_length=len(result["answer"]),
                    citation_count=len(result["citations"]),
                    confidence=result["confidence"]
                )
                
                return state
                
            except Exception as e:
                error_msg = f"Answer generation failed: {str(e)}"
                self.logger.error("Answer generation node failed", error=str(e))
                record_error("WorkflowError", "generate_answer")
                state["error"] = error_msg
                state["messages"].append(f"Error: {error_msg}")
                return state
    
    def _handle_error_node(self, state: RAGWorkflowState) -> RAGWorkflowState:
        """Handle workflow errors."""
        self.logger.error(
            "Workflow error occurred",
            stage=state.get("stage", "unknown"),
            error=state.get("error", "Unknown error")
        )
        
        state["answer"] = f"An error occurred during processing: {state.get('error', 'Unknown error')}"
        state["citations"] = []
        state["confidence"] = "low"
        state["messages"].append("Workflow completed with error")
        
        return state
    
    # Conditional edge functions
    
    def _should_continue_after_processing(self, state: RAGWorkflowState) -> str:
        """Determine next step after document processing."""
        if state.get("error"):
            return "error"
        elif not state.get("documents"):
            return "skip_processing"  # No documents to process, go to query enhancement
        else:
            return "continue"
    
    def _should_continue_after_chunking(self, state: RAGWorkflowState) -> str:
        """Determine next step after document chunking."""
        return "error" if state.get("error") else "continue"
    
    def _should_continue_after_storage(self, state: RAGWorkflowState) -> str:
        """Determine next step after embedding storage."""
        return "error" if state.get("error") else "continue"
    
    def _should_continue_after_enhancement(self, state: RAGWorkflowState) -> str:
        """Determine next step after query enhancement."""
        return "error" if state.get("error") else "continue"
    
    def _should_continue_after_retrieval(self, state: RAGWorkflowState) -> str:
        """Determine next step after document retrieval."""
        return "error" if state.get("error") else "continue"
    
    def _should_continue_after_reranking(self, state: RAGWorkflowState) -> str:
        """Determine next step after document reranking."""
        return "error" if state.get("error") else "continue"
    
    def _generate_context_hash(self, state: RAGWorkflowState) -> str:
        """Generate hash for current context to use as cache key."""
        # Create hash based on documents in vector store and query processing state
        context_data = {
            "document_count": self.vector_store.get_document_count(),
            "reranked_docs_count": len(state.get("reranked_docs", [])),
            "enhanced_query": state.get("enhanced_query", ""),
            "model_config": {
                "embedding_model": self.embedding_model.model_name,
                "reranker_model": self.reranker.model_name,
                "llm_model": self.answer_generator.llm.model_name if hasattr(self.answer_generator.llm, 'model_name') else "unknown"
            }
        }
        
        # Create hash from context data
        context_str = str(sorted(context_data.items()))
        return hashlib.sha256(context_str.encode()).hexdigest()[:16]
    
    def _generate_context_hash_for_query(self) -> str:
        """Generate hash for current query context (without state)."""
        context_data = {
            "document_count": self.vector_store.get_document_count(),
            "model_config": {
                "embedding_model": self.embedding_model.model_name,
                "reranker_model": self.reranker.model_name,
                "llm_model": self.answer_generator.llm.model_name if hasattr(self.answer_generator.llm, 'model_name') else "unknown"
            }
        }
        
        # Create hash from context data
        context_str = str(sorted(context_data.items()))
        return hashlib.sha256(context_str.encode()).hexdigest()[:16]
    
    def _check_query_cache(self, query: str, context_hash: str) -> Optional[Dict[str, Any]]:
        """Check if query result is cached."""
        if not self.cache_manager:
            return None
        
        return self.cache_manager.get_cached_query_result(query, context_hash)
    
    def _cache_query_result(self, query: str, context_hash: str, result: Dict[str, Any]) -> None:
        """Cache query result."""
        if not self.cache_manager:
            return
        
        # Create cacheable result (exclude non-serializable items)
        cacheable_result = {
            "answer": result["answer"],
            "citations": result["citations"],
            "confidence": result["confidence"],
            "metadata": result.get("metadata", {}),
            "cached_at": time.time()
        }
        
        self.cache_manager.cache_query_result(query, context_hash, cacheable_result)
    
    def process_query(
        self,
        query: str,
        pdf_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query: User's question
            pdf_path: Optional path to PDF document to process
            
        Returns:
            Dictionary containing answer, citations, and metadata
            
        Raises:
            WorkflowError: If workflow execution fails
        """
        with time_operation("workflow_process_query_total"):
            try:
                self.logger.info(
                    "Starting RAG workflow execution",
                    query=query,
                    pdf_path=pdf_path
                )
                
                # Record request metrics
                increment_counter("workflow_requests_total")
                if pdf_path:
                    increment_counter("workflow_requests_with_document")
                else:
                    increment_counter("workflow_requests_query_only")
                
                # Check cache for query results (only if no new PDF to process)
                if not pdf_path and self.cache_manager:
                    context_hash = self._generate_context_hash_for_query()
                    cached_result = self._check_query_cache(query, context_hash)
                    if cached_result:
                        increment_counter("workflow_cache_hits")
                        self.logger.info(
                            "Query cache hit - returning cached result",
                            query=query,
                            cached_at=cached_result.get("cached_at")
                        )
                        return cached_result
                    else:
                        increment_counter("workflow_cache_misses")
                
                # Initialize workflow state
                initial_state: RAGWorkflowState = {
                    "query": query,
                    "pdf_path": pdf_path,
                    "documents": [],
                    "chunks": [],
                    "enhanced_query": "",
                    "retrieved_docs": [],
                    "reranked_docs": [],
                    "answer": "",
                    "citations": [],
                    "confidence": "low",
                    "error": None,
                    "stage": "initialized",
                    "metadata": {},
                    "messages": []
                }
                
                # Execute workflow
                final_state = self.workflow.invoke(initial_state)
                
                # Extract results
                result = {
                    "answer": final_state["answer"],
                    "citations": final_state["citations"],
                    "confidence": final_state["confidence"],
                    "metadata": final_state["metadata"],
                    "messages": final_state["messages"],
                    "error": final_state.get("error")
                }
                
                # Record success/error metrics
                if final_state.get("error"):
                    increment_counter("workflow_requests_failed")
                else:
                    increment_counter("workflow_requests_successful")
                
                # Cache successful query results (only if no new PDF was processed)
                if not pdf_path and not final_state.get("error") and self.cache_manager:
                    context_hash = self._generate_context_hash_for_query()
                    self._cache_query_result(query, context_hash, result)
                    self.logger.debug("Query result cached", query=query)
                
                self.logger.info(
                    "RAG workflow execution completed",
                    success=not final_state.get("error"),
                    answer_length=len(final_state["answer"]),
                    citation_count=len(final_state["citations"])
                )
                
                return result
                
            except Exception as e:
                error_msg = f"RAG workflow execution failed: {str(e)}"
                self.logger.error("Workflow execution failed", error=str(e))
                increment_counter("workflow_requests_failed")
                record_error("WorkflowError", "process_query")
                raise WorkflowError(error_msg) from e
    
    async def process_query_async(
        self,
        query: str,
        pdf_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously process a query through the RAG pipeline.
        
        Args:
            query: User's question
            pdf_path: Optional path to PDF document to process
            
        Returns:
            Dictionary containing answer, citations, and metadata
        """
        # Run synchronous workflow in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process_query,
            query,
            pdf_path
        )
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """
        Get information about the workflow configuration.
        
        Returns:
            Dictionary containing workflow information
        """
        return {
            "vector_store_path": self.vector_store_path,
            "enable_caching": self.enable_caching,
            "components": {
                "document_processor": self.doc_processor.__class__.__name__,
                "chunker": self.chunker.__class__.__name__,
                "embedding_model": self.embedding_model.__class__.__name__,
                "vector_store": self.vector_store.__class__.__name__,
                "query_rewriter": self.query_rewriter.__class__.__name__,
                "hybrid_retriever": self.hybrid_retriever.__class__.__name__,
                "reranker": self.reranker.__class__.__name__,
                "answer_generator": self.answer_generator.__class__.__name__
            },
            "workflow_nodes": [
                "process_documents",
                "chunk_documents", 
                "store_embeddings",
                "enhance_query",
                "retrieve_documents",
                "rerank_documents",
                "generate_answer",
                "handle_error"
            ]
        }
    
    def clear_vector_store(self) -> None:
        """Clear the vector store and hybrid retriever."""
        try:
            self.vector_store.clear_store()
            # Reinitialize hybrid retriever to clear BM25 index
            self.hybrid_retriever = HybridRetriever(self.vector_store)
            self.logger.info("Vector store and retriever cleared")
        except Exception as e:
            self.logger.error("Failed to clear vector store", error=str(e))
            raise WorkflowError(f"Failed to clear vector store: {str(e)}") from e
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        try:
            if self.cache_manager:
                self.cache_manager.clear_all_caches()
                self.logger.info("All caches cleared")
            else:
                self.logger.info("No cache manager available - caching disabled")
        except Exception as e:
            self.logger.error("Failed to clear caches", error=str(e))
            raise WorkflowError(f"Failed to clear caches: {str(e)}") from e
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if self.cache_manager:
                return self.cache_manager.get_cache_stats()
            else:
                return {"caching_enabled": False, "message": "Caching is disabled"}
        except Exception as e:
            self.logger.error("Failed to get cache stats", error=str(e))
            return {"error": str(e)}