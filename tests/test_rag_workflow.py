"""Tests for RAG workflow orchestration."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from langchain_core.documents import Document

from src.rag_workflow import RAGWorkflow, RAGWorkflowState
from src.exceptions import WorkflowError


class TestRAGWorkflow:
    """Test cases for RAGWorkflow class."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="Machine learning is a method of data analysis.",
                metadata={"source": "test.pdf", "page": 1}
            ),
            Document(
                page_content="Deep learning uses neural networks.",
                metadata={"source": "test.pdf", "page": 2}
            )
        ]
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            Document(
                page_content="Machine learning is a method of data analysis.",
                metadata={"source": "test.pdf", "page": 1, "chunk_id": "chunk_1"}
            ),
            Document(
                page_content="Deep learning uses neural networks.",
                metadata={"source": "test.pdf", "page": 2, "chunk_id": "chunk_2"}
            )
        ]
    
    @patch('src.rag_workflow.DocumentProcessor')
    @patch('src.rag_workflow.ChunkingEngine')
    @patch('src.rag_workflow.EmbeddingModel')
    @patch('src.rag_workflow.VectorStore')
    @patch('src.rag_workflow.QueryRewriter')
    @patch('src.rag_workflow.HybridRetriever')
    @patch('src.rag_workflow.Reranker')
    @patch('src.rag_workflow.AnswerGenerator')
    def test_initialization_success(
        self,
        mock_answer_gen,
        mock_reranker,
        mock_hybrid_retriever,
        mock_query_rewriter,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_doc_processor
    ):
        """Test successful workflow initialization."""
        # Setup mocks
        mock_embedding_model.return_value.embeddings = Mock()
        
        workflow = RAGWorkflow()
        
        # Verify components were initialized
        mock_doc_processor.assert_called_once()
        mock_chunker.assert_called_once()
        mock_embedding_model.assert_called_once()
        mock_vector_store.assert_called_once()
        mock_query_rewriter.assert_called_once()
        mock_hybrid_retriever.assert_called_once()
        mock_reranker.assert_called_once()
        mock_answer_gen.assert_called_once()
        
        # Verify workflow is built
        assert workflow.workflow is not None
    
    @patch('src.rag_workflow.DocumentProcessor')
    def test_initialization_failure(self, mock_doc_processor):
        """Test workflow initialization failure."""
        mock_doc_processor.side_effect = Exception("Initialization failed")
        
        with pytest.raises(WorkflowError, match="Failed to initialize RAG workflow components"):
            RAGWorkflow()
    
    @patch('src.rag_workflow.DocumentProcessor')
    @patch('src.rag_workflow.ChunkingEngine')
    @patch('src.rag_workflow.EmbeddingModel')
    @patch('src.rag_workflow.VectorStore')
    @patch('src.rag_workflow.QueryRewriter')
    @patch('src.rag_workflow.HybridRetriever')
    @patch('src.rag_workflow.Reranker')
    @patch('src.rag_workflow.AnswerGenerator')
    def test_process_documents_node_success(
        self,
        mock_answer_gen,
        mock_reranker,
        mock_hybrid_retriever,
        mock_query_rewriter,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_doc_processor,
        sample_documents
    ):
        """Test successful document processing node."""
        # Setup mocks
        mock_embedding_model.return_value.embeddings = Mock()
        mock_doc_processor.return_value.process_pdf_file.return_value = sample_documents
        
        workflow = RAGWorkflow()
        
        # Test state
        state: RAGWorkflowState = {
            "query": "test query",
            "pdf_path": "test.pdf",
            "documents": [],
            "chunks": [],
            "enhanced_query": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "citations": [],
            "confidence": "low",
            "error": None,
            "stage": "",
            "metadata": {},
            "messages": []
        }
        
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            result = workflow._process_documents_node(state)
        
        # Verify results
        assert result["stage"] == "process_documents"
        assert len(result["documents"]) == 2
        assert result["metadata"]["document_count"] == 2
        assert result["error"] is None
        assert "Processed 2 pages from PDF" in result["messages"]
    
    @patch('src.rag_workflow.DocumentProcessor')
    @patch('src.rag_workflow.ChunkingEngine')
    @patch('src.rag_workflow.EmbeddingModel')
    @patch('src.rag_workflow.VectorStore')
    @patch('src.rag_workflow.QueryRewriter')
    @patch('src.rag_workflow.HybridRetriever')
    @patch('src.rag_workflow.Reranker')
    @patch('src.rag_workflow.AnswerGenerator')
    def test_process_documents_node_no_pdf(
        self,
        mock_answer_gen,
        mock_reranker,
        mock_hybrid_retriever,
        mock_query_rewriter,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_doc_processor
    ):
        """Test document processing node with no PDF."""
        # Setup mocks
        mock_embedding_model.return_value.embeddings = Mock()
        
        workflow = RAGWorkflow()
        
        # Test state without PDF
        state: RAGWorkflowState = {
            "query": "test query",
            "pdf_path": None,
            "documents": [],
            "chunks": [],
            "enhanced_query": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "citations": [],
            "confidence": "low",
            "error": None,
            "stage": "",
            "metadata": {},
            "messages": []
        }
        
        result = workflow._process_documents_node(state)
        
        # Verify results
        assert result["stage"] == "process_documents"
        assert len(result["documents"]) == 0
        assert result["error"] is None
        assert "No PDF provided, skipping document processing" in result["messages"]
    
    @patch('src.rag_workflow.DocumentProcessor')
    @patch('src.rag_workflow.ChunkingEngine')
    @patch('src.rag_workflow.EmbeddingModel')
    @patch('src.rag_workflow.VectorStore')
    @patch('src.rag_workflow.QueryRewriter')
    @patch('src.rag_workflow.HybridRetriever')
    @patch('src.rag_workflow.Reranker')
    @patch('src.rag_workflow.AnswerGenerator')
    def test_chunk_documents_node_success(
        self,
        mock_answer_gen,
        mock_reranker,
        mock_hybrid_retriever,
        mock_query_rewriter,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_doc_processor,
        sample_documents,
        sample_chunks
    ):
        """Test successful document chunking node."""
        # Setup mocks
        mock_embedding_model.return_value.embeddings = Mock()
        mock_chunker.return_value.split_documents.return_value = sample_chunks
        
        workflow = RAGWorkflow()
        
        # Test state
        state: RAGWorkflowState = {
            "query": "test query",
            "pdf_path": None,
            "documents": sample_documents,
            "chunks": [],
            "enhanced_query": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "citations": [],
            "confidence": "low",
            "error": None,
            "stage": "",
            "metadata": {},
            "messages": []
        }
        
        result = workflow._chunk_documents_node(state)
        
        # Verify results
        assert result["stage"] == "chunk_documents"
        assert len(result["chunks"]) == 2
        assert result["metadata"]["chunk_count"] == 2
        assert result["error"] is None
        assert "Created 2 chunks from documents" in result["messages"]
    
    @patch('src.rag_workflow.DocumentProcessor')
    @patch('src.rag_workflow.ChunkingEngine')
    @patch('src.rag_workflow.EmbeddingModel')
    @patch('src.rag_workflow.VectorStore')
    @patch('src.rag_workflow.QueryRewriter')
    @patch('src.rag_workflow.HybridRetriever')
    @patch('src.rag_workflow.Reranker')
    @patch('src.rag_workflow.AnswerGenerator')
    def test_enhance_query_node_success(
        self,
        mock_answer_gen,
        mock_reranker,
        mock_hybrid_retriever,
        mock_query_rewriter,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_doc_processor
    ):
        """Test successful query enhancement node."""
        # Setup mocks
        mock_embedding_model.return_value.embeddings = Mock()
        mock_query_rewriter.return_value.enhance_query.return_value = "enhanced test query"
        
        workflow = RAGWorkflow()
        
        # Test state
        state: RAGWorkflowState = {
            "query": "test query",
            "pdf_path": None,
            "documents": [],
            "chunks": [],
            "enhanced_query": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "citations": [],
            "confidence": "low",
            "error": None,
            "stage": "",
            "metadata": {},
            "messages": []
        }
        
        result = workflow._enhance_query_node(state)
        
        # Verify results
        assert result["stage"] == "enhance_query"
        assert result["enhanced_query"] == "enhanced test query"
        assert result["metadata"]["original_query"] == "test query"
        assert result["error"] is None
        assert "Query enhanced successfully" in result["messages"]
    
    @patch('src.rag_workflow.DocumentProcessor')
    @patch('src.rag_workflow.ChunkingEngine')
    @patch('src.rag_workflow.EmbeddingModel')
    @patch('src.rag_workflow.VectorStore')
    @patch('src.rag_workflow.QueryRewriter')
    @patch('src.rag_workflow.HybridRetriever')
    @patch('src.rag_workflow.Reranker')
    @patch('src.rag_workflow.AnswerGenerator')
    def test_retrieve_documents_node_success(
        self,
        mock_answer_gen,
        mock_reranker,
        mock_hybrid_retriever,
        mock_query_rewriter,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_doc_processor,
        sample_chunks
    ):
        """Test successful document retrieval node."""
        # Setup mocks
        mock_embedding_model.return_value.embeddings = Mock()
        mock_hybrid_retriever.return_value.retrieve.return_value = sample_chunks
        
        workflow = RAGWorkflow()
        
        # Test state
        state: RAGWorkflowState = {
            "query": "test query",
            "pdf_path": None,
            "documents": [],
            "chunks": [],
            "enhanced_query": "enhanced test query",
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "citations": [],
            "confidence": "low",
            "error": None,
            "stage": "",
            "metadata": {},
            "messages": []
        }
        
        result = workflow._retrieve_documents_node(state)
        
        # Verify results
        assert result["stage"] == "retrieve_documents"
        assert len(result["retrieved_docs"]) == 2
        assert result["metadata"]["retrieved_count"] == 2
        assert result["error"] is None
        assert "Retrieved 2 relevant documents" in result["messages"]
    
    @patch('src.rag_workflow.DocumentProcessor')
    @patch('src.rag_workflow.ChunkingEngine')
    @patch('src.rag_workflow.EmbeddingModel')
    @patch('src.rag_workflow.VectorStore')
    @patch('src.rag_workflow.QueryRewriter')
    @patch('src.rag_workflow.HybridRetriever')
    @patch('src.rag_workflow.Reranker')
    @patch('src.rag_workflow.AnswerGenerator')
    def test_generate_answer_node_success(
        self,
        mock_answer_gen,
        mock_reranker,
        mock_hybrid_retriever,
        mock_query_rewriter,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_doc_processor,
        sample_chunks
    ):
        """Test successful answer generation node."""
        # Setup mocks
        mock_embedding_model.return_value.embeddings = Mock()
        mock_answer_gen.return_value.generate_answer.return_value = {
            "answer": "Test answer",
            "citations": [{"source": "test.pdf", "page": 1}],
            "confidence": "high",
            "sufficient_information": True,
            "source_count": 2,
            "model_used": "test-model"
        }
        
        workflow = RAGWorkflow()
        
        # Test state
        state: RAGWorkflowState = {
            "query": "test query",
            "pdf_path": None,
            "documents": [],
            "chunks": [],
            "enhanced_query": "enhanced test query",
            "retrieved_docs": [],
            "reranked_docs": sample_chunks,
            "answer": "",
            "citations": [],
            "confidence": "low",
            "error": None,
            "stage": "",
            "metadata": {"original_query": "test query"},
            "messages": []
        }
        
        result = workflow._generate_answer_node(state)
        
        # Verify results
        assert result["stage"] == "generate_answer"
        assert result["answer"] == "Test answer"
        assert len(result["citations"]) == 1
        assert result["confidence"] == "high"
        assert result["error"] is None
        assert "Answer generated successfully" in result["messages"]
    
    @patch('src.rag_workflow.DocumentProcessor')
    @patch('src.rag_workflow.ChunkingEngine')
    @patch('src.rag_workflow.EmbeddingModel')
    @patch('src.rag_workflow.VectorStore')
    @patch('src.rag_workflow.QueryRewriter')
    @patch('src.rag_workflow.HybridRetriever')
    @patch('src.rag_workflow.Reranker')
    @patch('src.rag_workflow.AnswerGenerator')
    def test_handle_error_node(
        self,
        mock_answer_gen,
        mock_reranker,
        mock_hybrid_retriever,
        mock_query_rewriter,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_doc_processor
    ):
        """Test error handling node."""
        # Setup mocks
        mock_embedding_model.return_value.embeddings = Mock()
        
        workflow = RAGWorkflow()
        
        # Test state with error
        state: RAGWorkflowState = {
            "query": "test query",
            "pdf_path": None,
            "documents": [],
            "chunks": [],
            "enhanced_query": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "citations": [],
            "confidence": "low",
            "error": "Test error occurred",
            "stage": "test_stage",
            "metadata": {},
            "messages": []
        }
        
        result = workflow._handle_error_node(state)
        
        # Verify results
        assert "An error occurred during processing: Test error occurred" in result["answer"]
        assert result["citations"] == []
        assert result["confidence"] == "low"
        assert "Workflow completed with error" in result["messages"]
    
    @patch('src.rag_workflow.DocumentProcessor')
    @patch('src.rag_workflow.ChunkingEngine')
    @patch('src.rag_workflow.EmbeddingModel')
    @patch('src.rag_workflow.VectorStore')
    @patch('src.rag_workflow.QueryRewriter')
    @patch('src.rag_workflow.HybridRetriever')
    @patch('src.rag_workflow.Reranker')
    @patch('src.rag_workflow.AnswerGenerator')
    def test_conditional_edges(
        self,
        mock_answer_gen,
        mock_reranker,
        mock_hybrid_retriever,
        mock_query_rewriter,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_doc_processor
    ):
        """Test conditional edge functions."""
        # Setup mocks
        mock_embedding_model.return_value.embeddings = Mock()
        
        workflow = RAGWorkflow()
        
        # Test error condition
        error_state: RAGWorkflowState = {
            "query": "test",
            "pdf_path": None,
            "documents": [],
            "chunks": [],
            "enhanced_query": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "citations": [],
            "confidence": "low",
            "error": "Test error",
            "stage": "",
            "metadata": {},
            "messages": []
        }
        
        assert workflow._should_continue_after_processing(error_state) == "error"
        assert workflow._should_continue_after_chunking(error_state) == "error"
        assert workflow._should_continue_after_enhancement(error_state) == "error"
        
        # Test success condition
        success_state: RAGWorkflowState = {
            "query": "test",
            "pdf_path": None,
            "documents": [Mock()],
            "chunks": [],
            "enhanced_query": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "citations": [],
            "confidence": "low",
            "error": None,
            "stage": "",
            "metadata": {},
            "messages": []
        }
        
        assert workflow._should_continue_after_processing(success_state) == "continue"
        assert workflow._should_continue_after_chunking(success_state) == "continue"
        
        # Test skip processing condition
        skip_state: RAGWorkflowState = {
            "query": "test",
            "pdf_path": None,
            "documents": [],
            "chunks": [],
            "enhanced_query": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "citations": [],
            "confidence": "low",
            "error": None,
            "stage": "",
            "metadata": {},
            "messages": []
        }
        
        assert workflow._should_continue_after_processing(skip_state) == "skip_processing"
    
    @patch('src.rag_workflow.DocumentProcessor')
    @patch('src.rag_workflow.ChunkingEngine')
    @patch('src.rag_workflow.EmbeddingModel')
    @patch('src.rag_workflow.VectorStore')
    @patch('src.rag_workflow.QueryRewriter')
    @patch('src.rag_workflow.HybridRetriever')
    @patch('src.rag_workflow.Reranker')
    @patch('src.rag_workflow.AnswerGenerator')
    def test_get_workflow_info(
        self,
        mock_answer_gen,
        mock_reranker,
        mock_hybrid_retriever,
        mock_query_rewriter,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_doc_processor
    ):
        """Test getting workflow information."""
        # Setup mocks
        mock_embedding_model.return_value.embeddings = Mock()
        
        workflow = RAGWorkflow(vector_store_path="test_path", enable_caching=False)
        
        info = workflow.get_workflow_info()
        
        # Verify information
        assert info["vector_store_path"] == "test_path"
        assert info["enable_caching"] is False
        assert "components" in info
        assert "workflow_nodes" in info
        assert len(info["workflow_nodes"]) == 8
        assert "process_documents" in info["workflow_nodes"]
        assert "generate_answer" in info["workflow_nodes"]
    
    @patch('src.rag_workflow.DocumentProcessor')
    @patch('src.rag_workflow.ChunkingEngine')
    @patch('src.rag_workflow.EmbeddingModel')
    @patch('src.rag_workflow.VectorStore')
    @patch('src.rag_workflow.QueryRewriter')
    @patch('src.rag_workflow.HybridRetriever')
    @patch('src.rag_workflow.Reranker')
    @patch('src.rag_workflow.AnswerGenerator')
    def test_clear_vector_store(
        self,
        mock_answer_gen,
        mock_reranker,
        mock_hybrid_retriever,
        mock_query_rewriter,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_doc_processor
    ):
        """Test clearing vector store."""
        # Setup mocks
        mock_embedding_model.return_value.embeddings = Mock()
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        workflow = RAGWorkflow()
        
        workflow.clear_vector_store()
        
        # Verify vector store was cleared
        mock_vector_store_instance.clear_store.assert_called_once()
        
        # Verify hybrid retriever was reinitialized
        assert mock_hybrid_retriever.call_count == 2  # Once during init, once during clear
    
    @patch('src.rag_workflow.DocumentProcessor')
    @patch('src.rag_workflow.ChunkingEngine')
    @patch('src.rag_workflow.EmbeddingModel')
    @patch('src.rag_workflow.VectorStore')
    @patch('src.rag_workflow.QueryRewriter')
    @patch('src.rag_workflow.HybridRetriever')
    @patch('src.rag_workflow.Reranker')
    @patch('src.rag_workflow.AnswerGenerator')
    def test_process_query_empty_query(
        self,
        mock_answer_gen,
        mock_reranker,
        mock_hybrid_retriever,
        mock_query_rewriter,
        mock_vector_store,
        mock_embedding_model,
        mock_chunker,
        mock_doc_processor
    ):
        """Test processing empty query."""
        # Setup mocks
        mock_embedding_model.return_value.embeddings = Mock()
        
        workflow = RAGWorkflow()
        
        # Mock workflow execution to simulate empty query error
        workflow.workflow = Mock()
        workflow.workflow.invoke.return_value = {
            "answer": "An error occurred during processing: Query cannot be empty",
            "citations": [],
            "confidence": "low",
            "metadata": {},
            "messages": ["Error: Query cannot be empty"],
            "error": "Query cannot be empty"
        }
        
        result = workflow.process_query("")
        
        # Verify error handling
        assert "Query cannot be empty" in result["answer"]
        assert result["confidence"] == "low"
        assert result["error"] == "Query cannot be empty"


class TestRAGWorkflowIntegration:
    """Integration tests for RAGWorkflow."""
    
    def test_workflow_state_structure(self):
        """Test that workflow state structure is correct."""
        from src.rag_workflow import RAGWorkflowState
        
        # Test state can be created with all required fields
        state: RAGWorkflowState = {
            "query": "test",
            "pdf_path": None,
            "documents": [],
            "chunks": [],
            "enhanced_query": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "citations": [],
            "confidence": "low",
            "error": None,
            "stage": "",
            "metadata": {},
            "messages": []
        }
        
        # Verify all required fields are present
        required_fields = [
            "query", "pdf_path", "documents", "chunks", "enhanced_query",
            "retrieved_docs", "reranked_docs", "answer", "citations",
            "confidence", "error", "stage", "metadata", "messages"
        ]
        
        for field in required_fields:
            assert field in state
    
    def test_workflow_error_propagation(self):
        """Test that errors propagate correctly through workflow."""
        from src.rag_workflow import RAGWorkflow
        
        # Test that WorkflowError is raised for component initialization failure
        with patch('src.rag_workflow.DocumentProcessor', side_effect=Exception("Test error")):
            with pytest.raises(WorkflowError, match="Failed to initialize RAG workflow components"):
                RAGWorkflow()