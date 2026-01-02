"""Unit tests for FastAPI REST API."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
import io

from src.api import app, api_state, get_workflow
from src.rag_workflow import RAGWorkflow
from src.exceptions import WorkflowError


class TestAPIEndpoints:
    """Test suite for API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        # Reset API state for each test
        api_state.workflow = None
        api_state.active_requests = 0
        api_state.request_count = 0
    
    def test_root_endpoint(self):
        """Test root endpoint returns basic information."""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "PDF RAG System API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "/docs" in data["docs"]
        assert "/health" in data["health"]
    
    @patch('src.api.get_workflow')
    @patch('psutil.Process')
    def test_health_check_success(self, mock_process, mock_get_workflow):
        """Test health check endpoint with successful response."""
        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.vector_store.get_document_count.return_value = 42
        mock_get_workflow.return_value = mock_workflow
        
        # Mock psutil
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_process.return_value = mock_process_instance
        
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["total_chunks"] == 42
        assert data["memory_usage_mb"] == 100.0
        assert "uptime_seconds" in data
        assert "timestamp" in data
    
    @patch('src.api.get_workflow')
    def test_health_check_failure(self, mock_get_workflow):
        """Test health check endpoint with failure."""
        mock_get_workflow.side_effect = Exception("Workflow initialization failed")
        
        response = self.client.get("/health")
        
        assert response.status_code == 500
        assert "Health check failed" in response.json()["detail"]
    
    @patch('src.api.get_workflow')
    def test_upload_document_success(self, mock_get_workflow):
        """Test successful document upload."""
        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.process_query.return_value = {
            "error": None,
            "metadata": {
                "document_count": 3,
                "chunk_count": 15
            }
        }
        mock_get_workflow.return_value = mock_workflow
        
        # Create test PDF content
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<\n/Size 1\n/Root 1 0 R\n>>\nstartxref\n9\n%%EOF"
        
        with patch('src.api.ensure_directory_exists'), \
             patch('tempfile.NamedTemporaryFile') as mock_temp, \
             patch('os.unlink'):
            
            # Mock temporary file
            mock_temp_file = Mock()
            mock_temp_file.name = "/tmp/test.pdf"
            mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
            mock_temp_file.__exit__ = Mock(return_value=None)
            mock_temp.return_value = mock_temp_file
            
            response = self.client.post(
                "/upload",
                files={"file": ("test.pdf", pdf_content, "application/pdf")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.pdf"
        assert data["file_size"] == len(pdf_content)
        assert data["pages_processed"] == 3
        assert data["chunks_created"] == 15
        assert data["status"] == "completed"
        assert "processing_time_ms" in data
        assert "document_id" in data
        assert "timestamp" in data
    
    def test_upload_document_invalid_file_type(self):
        """Test document upload with invalid file type."""
        response = self.client.post(
            "/upload",
            files={"file": ("test.txt", b"Not a PDF", "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Only PDF files are supported" in response.json()["detail"]
    
    def test_upload_document_no_filename(self):
        """Test document upload with no filename."""
        response = self.client.post(
            "/upload",
            files={"file": ("", b"content", "application/pdf")}
        )
        
        assert response.status_code == 400
        assert "Only PDF files are supported" in response.json()["detail"]
    
    @patch('src.api.get_workflow')
    def test_upload_document_processing_error(self, mock_get_workflow):
        """Test document upload with processing error."""
        # Mock workflow with error
        mock_workflow = Mock()
        mock_workflow.process_query.return_value = {
            "error": "Processing failed"
        }
        mock_get_workflow.return_value = mock_workflow
        
        pdf_content = b"%PDF-1.4\ntest content"
        
        with patch('src.api.ensure_directory_exists'), \
             patch('tempfile.NamedTemporaryFile') as mock_temp, \
             patch('os.unlink'):
            
            mock_temp_file = Mock()
            mock_temp_file.name = "/tmp/test.pdf"
            mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
            mock_temp_file.__exit__ = Mock(return_value=None)
            mock_temp.return_value = mock_temp_file
            
            response = self.client.post(
                "/upload",
                files={"file": ("test.pdf", pdf_content, "application/pdf")}
            )
        
        assert response.status_code == 422
        assert "Document processing failed" in response.json()["detail"]
    
    @patch('src.api.get_workflow')
    def test_process_query_success(self, mock_get_workflow):
        """Test successful query processing."""
        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.process_query_async = AsyncMock(return_value={
            "error": None,
            "answer": "This is the answer",
            "citations": [{"source": "doc1", "page": 1}],
            "confidence": "high",
            "metadata": {
                "answer_metadata": {
                    "sufficient_information": True,
                    "source_count": 1
                }
            }
        })
        mock_get_workflow.return_value = mock_workflow
        
        response = self.client.post(
            "/query",
            json={
                "query": "What is the answer?",
                "include_citations": True,
                "check_sufficiency": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is the answer?"
        assert data["answer"] == "This is the answer"
        assert len(data["citations"]) == 1
        assert data["confidence"] == "high"
        assert data["sufficient_information"] is True
        assert data["source_count"] == 1
        assert "processing_time_ms" in data
        assert "request_id" in data
        assert "timestamp" in data
    
    def test_process_query_empty_query(self):
        """Test query processing with empty query."""
        response = self.client.post(
            "/query",
            json={
                "query": "",
                "include_citations": True
            }
        )
        
        assert response.status_code == 422
        # Pydantic validation error for empty query
    
    def test_process_query_whitespace_only(self):
        """Test query processing with whitespace-only query."""
        response = self.client.post(
            "/query",
            json={
                "query": "   ",
                "include_citations": True
            }
        )
        
        assert response.status_code == 422
        # Pydantic validation error for whitespace-only query
    
    @patch('src.api.get_workflow')
    def test_process_query_workflow_error(self, mock_get_workflow):
        """Test query processing with workflow error."""
        # Mock workflow with error
        mock_workflow = Mock()
        mock_workflow.process_query_async = AsyncMock(return_value={
            "error": "Query processing failed"
        })
        mock_get_workflow.return_value = mock_workflow
        
        response = self.client.post(
            "/query",
            json={
                "query": "What is the answer?",
                "include_citations": True
            }
        )
        
        assert response.status_code == 422
        assert "Query processing failed" in response.json()["detail"]
    
    @patch('src.api.get_workflow')
    def test_query_with_document_success(self, mock_get_workflow):
        """Test successful query with document upload."""
        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.process_query_async = AsyncMock(return_value={
            "error": None,
            "answer": "Answer from document",
            "citations": [{"source": "uploaded_doc", "page": 1}],
            "confidence": "medium",
            "metadata": {
                "answer_metadata": {
                    "sufficient_information": True,
                    "source_count": 1
                }
            }
        })
        mock_get_workflow.return_value = mock_workflow
        
        pdf_content = b"%PDF-1.4\ntest content"
        
        with patch('src.api.ensure_directory_exists'), \
             patch('tempfile.NamedTemporaryFile') as mock_temp, \
             patch('os.unlink'):
            
            mock_temp_file = Mock()
            mock_temp_file.name = "/tmp/test.pdf"
            mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
            mock_temp_file.__exit__ = Mock(return_value=None)
            mock_temp.return_value = mock_temp_file
            
            response = self.client.post(
                "/query-with-document",
                data={"query": "What does this document say?", "include_citations": "true"},
                files={"file": ("test.pdf", pdf_content, "application/pdf")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What does this document say?"
        assert data["answer"] == "Answer from document"
        assert len(data["citations"]) == 1
        assert data["confidence"] == "medium"
    
    def test_query_with_document_empty_query(self):
        """Test query with document with empty query."""
        pdf_content = b"%PDF-1.4\ntest content"
        
        response = self.client.post(
            "/query-with-document",
            data={"query": "", "include_citations": "true"},
            files={"file": ("test.pdf", pdf_content, "application/pdf")}
        )
        
        assert response.status_code == 400
        assert "Query cannot be empty" in response.json()["detail"]
    
    def test_query_with_document_invalid_file(self):
        """Test query with document with invalid file type."""
        response = self.client.post(
            "/query-with-document",
            data={"query": "What does this say?", "include_citations": "true"},
            files={"file": ("test.txt", b"Not a PDF", "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Only PDF files are supported" in response.json()["detail"]
    
    @patch('src.api.get_workflow')
    def test_clear_documents_success(self, mock_get_workflow):
        """Test successful document clearing."""
        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.clear_vector_store.return_value = None
        mock_get_workflow.return_value = mock_workflow
        
        response = self.client.delete("/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "All documents cleared" in data["message"]
        assert "request_id" in data
        assert "timestamp" in data
    
    @patch('src.api.get_workflow')
    def test_clear_documents_error(self, mock_get_workflow):
        """Test document clearing with error."""
        # Mock workflow with error
        mock_workflow = Mock()
        mock_workflow.clear_vector_store.side_effect = Exception("Clear failed")
        mock_get_workflow.return_value = mock_workflow
        
        response = self.client.delete("/documents")
        
        assert response.status_code == 500
        assert "Failed to clear documents" in response.json()["detail"]
    
    @patch('src.api.get_workflow')
    @patch('psutil.Process')
    def test_get_metrics_success(self, mock_process, mock_get_workflow):
        """Test successful metrics retrieval."""
        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.get_workflow_info.return_value = {
            "vector_store_path": "/path/to/store",
            "enable_caching": True,
            "components": {"component1": "Component1"}
        }
        mock_workflow.vector_store.get_document_count.return_value = 100
        mock_get_workflow.return_value = mock_workflow
        
        # Mock psutil
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 200 * 1024 * 1024  # 200MB
        mock_process_instance.memory_percent.return_value = 15.5
        mock_process_instance.cpu_percent.return_value = 25.0
        mock_process.return_value = mock_process_instance
        
        # Set some API state
        api_state.active_requests = 2
        api_state.request_count = 50
        
        response = self.client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check system metrics
        assert data["system"]["memory_usage_mb"] == 200.0
        assert data["system"]["memory_percent"] == 15.5
        assert data["system"]["cpu_percent"] == 25.0
        assert data["system"]["active_requests"] == 2
        assert data["system"]["total_requests"] == 50
        
        # Check workflow metrics
        assert data["workflow"]["total_chunks"] == 100
        assert data["workflow"]["vector_store_path"] == "/path/to/store"
        assert data["workflow"]["enable_caching"] is True
        
        assert "timestamp" in data
    
    @patch('src.api.get_workflow')
    def test_get_metrics_error(self, mock_get_workflow):
        """Test metrics retrieval with error."""
        mock_get_workflow.side_effect = Exception("Metrics failed")
        
        response = self.client.get("/metrics")
        
        assert response.status_code == 500
        assert "Failed to get metrics" in response.json()["detail"]


class TestAPIValidation:
    """Test suite for API input validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    def test_query_request_validation(self):
        """Test QueryRequest model validation."""
        from src.api import QueryRequest
        
        # Valid request
        valid_request = QueryRequest(
            query="What is the answer?",
            include_citations=True,
            check_sufficiency=False
        )
        assert valid_request.query == "What is the answer?"
        assert valid_request.include_citations is True
        assert valid_request.check_sufficiency is False
        
        # Test query stripping
        stripped_request = QueryRequest(query="  spaced query  ")
        assert stripped_request.query == "spaced query"
        
        # Test validation errors
        with pytest.raises(ValueError, match="Query cannot be empty"):
            QueryRequest(query="")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            QueryRequest(query="   ")
    
    def test_query_request_length_validation(self):
        """Test query length validation."""
        from src.api import QueryRequest
        
        # Test maximum length
        long_query = "x" * 1001  # Exceeds 1000 char limit
        with pytest.raises(ValueError):
            QueryRequest(query=long_query)
        
        # Test valid length
        valid_query = "x" * 1000  # Exactly at limit
        request = QueryRequest(query=valid_query)
        assert len(request.query) == 1000


class TestAPIState:
    """Test suite for API state management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset API state
        api_state.workflow = None
        api_state.active_requests = 0
        api_state.request_count = 0
    
    def test_api_state_initialization(self):
        """Test API state initialization."""
        from src.api import APIState
        
        state = APIState()
        assert state.workflow is None
        assert state.active_requests == 0
        assert state.request_count == 0
        assert state.get_uptime() >= 0
    
    def test_api_state_uptime(self):
        """Test API state uptime calculation."""
        import time
        from src.api import APIState
        
        state = APIState()
        initial_uptime = state.get_uptime()
        
        # Wait a small amount
        time.sleep(0.01)
        
        later_uptime = state.get_uptime()
        assert later_uptime > initial_uptime


class TestAPIUtilities:
    """Test suite for API utility functions."""
    
    def test_generate_request_id(self):
        """Test request ID generation."""
        from src.api import generate_request_id
        
        id1 = generate_request_id()
        id2 = generate_request_id()
        
        assert id1 != id2
        assert len(id1) > 0
        assert len(id2) > 0
        # Should be valid UUID format
        import uuid
        uuid.UUID(id1)  # Should not raise exception
        uuid.UUID(id2)  # Should not raise exception
    
    def test_get_current_timestamp(self):
        """Test timestamp generation."""
        from src.api import get_current_timestamp
        
        timestamp = get_current_timestamp()
        assert len(timestamp) > 0
        
        # Should be valid ISO format
        from datetime import datetime
        parsed = datetime.fromisoformat(timestamp)
        assert parsed is not None


class TestAPIErrorHandling:
    """Test suite for API error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    def test_http_exception_handler(self):
        """Test HTTP exception handling."""
        # Test 404 endpoint
        response = self.client.get("/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "error_type" in data
        assert "request_id" in data
        assert "timestamp" in data
        assert data["error_type"] == "HTTPException"
    
    @patch('src.api.get_workflow')
    def test_general_exception_handler(self, mock_get_workflow):
        """Test general exception handling."""
        # Mock workflow to raise unexpected exception
        mock_get_workflow.side_effect = RuntimeError("Unexpected error")
        
        response = self.client.get("/health")
        
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Internal server error"
        assert data["error_type"] == "RuntimeError"
        assert "request_id" in data
        assert "timestamp" in data
        assert "details" in data


class TestAPIStartupShutdown:
    """Test suite for API startup and shutdown events."""
    
    @patch('src.api.validate_required_settings')
    @patch('src.api.ensure_directory_exists')
    def test_startup_event_success(self, mock_ensure_dir, mock_validate):
        """Test successful startup event."""
        from src.api import startup_event
        import asyncio
        
        # Mock successful validation
        mock_validate.return_value = None
        mock_ensure_dir.return_value = None
        
        # Should not raise exception
        asyncio.run(startup_event())
        
        # Verify calls
        mock_validate.assert_called_once()
        assert mock_ensure_dir.call_count == 2  # Two directories
    
    @patch('src.api.validate_required_settings')
    def test_startup_event_failure(self, mock_validate):
        """Test startup event with failure."""
        from src.api import startup_event
        from src.exceptions import ConfigurationError
        import asyncio
        
        # Mock validation failure
        mock_validate.side_effect = ValueError("Config error")
        
        with pytest.raises(ConfigurationError, match="Startup failed"):
            asyncio.run(startup_event())
    
    def test_shutdown_event(self):
        """Test shutdown event."""
        from src.api import shutdown_event
        import asyncio
        
        # Should not raise exception
        asyncio.run(shutdown_event())


if __name__ == "__main__":
    pytest.main([__file__])