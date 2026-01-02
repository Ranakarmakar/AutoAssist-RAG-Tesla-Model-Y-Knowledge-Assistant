"""FastAPI REST API for PDF RAG System."""

import os
import tempfile
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

from src.config import settings
from src.logging_config import get_logger
from src.exceptions import WorkflowError, ConfigurationError
from src.rag_workflow import RAGWorkflow
from src.utils import ensure_directory_exists
from src.metrics import get_metrics_collector, time_operation, increment_counter, record_error

logger = get_logger(__name__)

# Pydantic models for API requests and responses

class QueryRequest(BaseModel):
    """Request model for query processing."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="User's question")
    include_citations: bool = Field(True, description="Whether to include source citations")
    check_sufficiency: bool = Field(True, description="Whether to check information sufficiency")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()


class QueryResponse(BaseModel):
    """Response model for query processing."""
    
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Generated answer")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Source citations")
    confidence: str = Field(..., description="Answer confidence level")
    sufficient_information: bool = Field(..., description="Whether information was sufficient")
    source_count: int = Field(..., description="Number of source documents used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: str = Field(..., description="Response timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    
    filename: str = Field(..., description="Uploaded filename")
    file_size: int = Field(..., description="File size in bytes")
    pages_processed: int = Field(..., description="Number of pages processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    document_id: str = Field(..., description="Unique document identifier")
    timestamp: str = Field(..., description="Upload timestamp")
    status: str = Field(..., description="Processing status")


class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    
    status: str = Field(..., description="System status")
    version: str = Field(..., description="System version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    total_documents: int = Field(..., description="Total documents in system")
    total_chunks: int = Field(..., description="Total chunks in vector store")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    active_requests: int = Field(..., description="Number of active requests")
    timestamp: str = Field(..., description="Status timestamp")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    request_id: str = Field(..., description="Request identifier")
    timestamp: str = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# Global state management
class APIState:
    """Global API state management."""
    
    def __init__(self):
        self.workflow: Optional[RAGWorkflow] = None
        self.start_time = datetime.now()
        self.active_requests = 0
        self.request_count = 0
        self.metrics_collector = get_metrics_collector()
        
    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()


# Global API state
api_state = APIState()


# FastAPI app initialization
app = FastAPI(
    title="PDF RAG System API",
    description="""
    ## PDF Retrieval-Augmented Generation (RAG) System API

    A comprehensive REST API for processing PDF documents and answering questions using advanced AI techniques.

    ### Features
    - **Document Processing**: Upload and process PDF documents with intelligent chunking
    - **Hybrid Retrieval**: Combines keyword (BM25) and semantic (vector) search
    - **Query Enhancement**: AI-powered query rewriting for better results
    - **Answer Generation**: Contextual answers with source citations
    - **Performance Monitoring**: Comprehensive metrics and system health monitoring
    - **Caching**: Intelligent caching for improved performance

    ### Workflow
    1. **Upload Document**: Process PDF into searchable chunks
    2. **Query Processing**: Enhanced query understanding and retrieval
    3. **Answer Generation**: AI-generated answers with citations
    4. **Monitoring**: Track performance and system health

    ### Models Used
    - **LLM**: Groq llama-3.3-70b-versatile for query enhancement and answer generation
    - **Embeddings**: sentence-transformers/all-mpnet-base-v2 for semantic search
    - **Reranking**: cross-encoder/ms-marco-MiniLM-L-6-v2 for relevance scoring

    ### Authentication
    Currently no authentication required. Configure API keys via environment variables.

    ### Rate Limits
    No rate limits currently enforced. Monitor system resources via `/metrics` endpoints.

    ### Support
    For issues and questions, check the health endpoint and system metrics.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "PDF RAG System",
        "url": "https://github.com/your-repo/pdf-rag-system",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://your-production-domain.com",
            "description": "Production server"
        }
    ],
    tags_metadata=[
        {
            "name": "system",
            "description": "System information and health monitoring"
        },
        {
            "name": "documents",
            "description": "Document upload and management operations"
        },
        {
            "name": "queries",
            "description": "Query processing and answer generation"
        },
        {
            "name": "metrics",
            "description": "Performance metrics and system monitoring"
        },
        {
            "name": "management",
            "description": "System management and maintenance operations"
        }
    ]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency functions

async def get_workflow() -> RAGWorkflow:
    """Get or initialize the RAG workflow."""
    if api_state.workflow is None:
        try:
            logger.info("Initializing RAG workflow for API")
            api_state.workflow = RAGWorkflow(
                vector_store_path=settings.vector_store_path,
                enable_caching=True
            )
            logger.info("RAG workflow initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize RAG workflow", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize RAG system: {str(e)}"
            )
    
    return api_state.workflow


def generate_request_id() -> str:
    """Generate unique request ID."""
    return str(uuid.uuid4())


def get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


async def track_request():
    """Track active request count."""
    api_state.active_requests += 1
    api_state.request_count += 1
    increment_counter("api_requests_total")
    try:
        yield
    finally:
        api_state.active_requests -= 1


# API endpoints

@app.get("/", response_model=Dict[str, str], tags=["system"])
async def root():
    """
    ## Root Endpoint
    
    Get basic system information and available endpoints.
    
    **Returns:**
    - Service name and version
    - Available endpoint URLs
    - System status
    
    **Example Response:**
    ```json
    {
        "service": "PDF RAG System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }
    ```
    """
    return {
        "service": "PDF RAG System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=SystemStatusResponse, tags=["system"])
async def health_check(workflow: RAGWorkflow = Depends(get_workflow)):
    """
    ## Health Check Endpoint
    
    Get comprehensive system health and status information.
    
    **Returns:**
    - System health status (healthy/degraded/critical)
    - Uptime and version information
    - Resource usage (memory, CPU)
    - Document and chunk counts
    - Active request monitoring
    
    **Health Status Levels:**
    - `healthy`: All systems operating normally
    - `degraded`: Some performance issues detected
    - `critical`: Serious issues requiring attention
    
    **Example Response:**
    ```json
    {
        "status": "healthy",
        "version": "1.0.0",
        "uptime_seconds": 3600.5,
        "total_documents": 0,
        "total_chunks": 1250,
        "memory_usage_mb": 256.7,
        "active_requests": 2,
        "timestamp": "2024-01-15T10:30:00"
    }
    ```
    
    **Use Cases:**
    - Monitor system health in production
    - Check resource usage and capacity
    - Verify system is ready to process requests
    - Load balancer health checks
    """
    try:
        # Get system metrics
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Get document counts
        total_chunks = workflow.vector_store.get_document_count()
        
        return SystemStatusResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=api_state.get_uptime(),
            total_documents=0,  # Could track this separately
            total_chunks=total_chunks,
            memory_usage_mb=memory_mb,
            active_requests=api_state.active_requests,
            timestamp=get_current_timestamp()
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/upload", response_model=DocumentUploadResponse, tags=["documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload and process"),
    workflow: RAGWorkflow = Depends(get_workflow),
    _: None = Depends(track_request)
):
    """
    ## Upload PDF Document
    
    Upload and process a PDF document for later querying.
    
    **Process:**
    1. Validates PDF file format and size
    2. Extracts text content from PDF pages
    3. Splits content into searchable chunks
    4. Generates embeddings for semantic search
    5. Stores in vector database for retrieval
    
    **File Requirements:**
    - Format: PDF only
    - Size limit: 50MB maximum
    - Content: Text-based PDFs (not scanned images)
    
    **Processing Details:**
    - Limited to first 5 pages for performance
    - Configurable chunk size (default: 1000 characters)
    - Chunk overlap for context preservation (default: 200 characters)
    
    **Example Response:**
    ```json
    {
        "filename": "manual.pdf",
        "file_size": 2048576,
        "pages_processed": 5,
        "chunks_created": 45,
        "processing_time_ms": 15420.5,
        "document_id": "uuid-string",
        "timestamp": "2024-01-15T10:30:00",
        "status": "completed"
    }
    ```
    
    **Error Responses:**
    - `400`: Invalid file format or empty file
    - `413`: File size exceeds 50MB limit
    - `422`: PDF processing failed (corrupted file, no text content)
    - `500`: Internal server error
    
    **Use Cases:**
    - Upload company documents for Q&A
    - Process manuals and guides
    - Index research papers and reports
    - Build knowledge base from PDF content
    """
    with time_operation("api_upload_document"):
        request_id = generate_request_id()
        start_time = datetime.now()
        
        logger.info("Document upload started", request_id=request_id, filename=file.filename)
        increment_counter("api_upload_requests")
        
        # Validate file
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            increment_counter("api_upload_validation_errors")
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        if file.size and file.size > 50 * 1024 * 1024:  # 50MB limit
            increment_counter("api_upload_size_errors")
            raise HTTPException(
                status_code=413,
                detail="File size exceeds 50MB limit"
            )
        
        temp_file_path = None
        
        try:
            # Save uploaded file to temporary location
            ensure_directory_exists(settings.upload_path)
            
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.pdf',
                dir=settings.upload_path
            ) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Process document through workflow (query-less processing)
            # We'll use a dummy query to trigger document processing
            result = workflow.process_query(
                query="Document processing",
                pdf_path=temp_file_path
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if result.get("error"):
                raise WorkflowError(f"Document processing failed: {result['error']}")
            
            # Extract processing results
            pages_processed = result["metadata"].get("document_count", 0)
            chunks_created = result["metadata"].get("chunk_count", 0)
            
            response = DocumentUploadResponse(
                filename=file.filename,
                file_size=len(content),
                pages_processed=pages_processed,
                chunks_created=chunks_created,
                processing_time_ms=processing_time,
                document_id=request_id,
                timestamp=get_current_timestamp(),
                status="completed"
            )
            
            # Record success metrics
            increment_counter("api_upload_successful")
            
            logger.info(
                "Document upload completed",
                request_id=request_id,
                filename=file.filename,
                pages_processed=pages_processed,
                chunks_created=chunks_created,
                processing_time_ms=processing_time
            )
            
            return response
            
        except WorkflowError as e:
            logger.error("Document processing failed", request_id=request_id, error=str(e))
            increment_counter("api_upload_processing_errors")
            record_error("WorkflowError", "upload_document")
            raise HTTPException(
                status_code=422,
                detail=f"Document processing failed: {str(e)}"
            )
        except Exception as e:
            logger.error("Document upload failed", request_id=request_id, error=str(e))
            increment_counter("api_upload_errors")
            record_error("Exception", "upload_document")
            raise HTTPException(
                status_code=500,
                detail=f"Document upload failed: {str(e)}"
            )
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning("Failed to clean up temporary file", error=str(e))


@app.post("/query", response_model=QueryResponse, tags=["queries"])
async def process_query(
    request: QueryRequest,
    workflow: RAGWorkflow = Depends(get_workflow),
    _: None = Depends(track_request)
):
    """
    ## Process Query
    
    Process a question against previously uploaded documents.
    
    **Process:**
    1. Enhances query using AI for better understanding
    2. Performs hybrid search (keyword + semantic)
    3. Reranks results by relevance
    4. Generates contextual answer with citations
    5. Assesses information sufficiency and confidence
    
    **Query Enhancement:**
    - Expands abbreviations and technical terms
    - Adds context and clarifying information
    - Generates alternative phrasings for better retrieval
    
    **Retrieval Strategy:**
    - BM25 keyword search (30% weight)
    - Semantic vector search (70% weight)
    - Cross-encoder reranking for final relevance
    
    **Request Body:**
    ```json
    {
        "query": "What are the safety features of the vehicle?",
        "include_citations": true,
        "check_sufficiency": true
    }
    ```
    
    **Example Response:**
    ```json
    {
        "query": "What are the safety features?",
        "answer": "The vehicle includes several safety features: airbags, ABS braking system, and electronic stability control...",
        "citations": [
            {
                "source": "manual.pdf",
                "page": 15,
                "text": "Safety features include...",
                "relevance_score": 0.95
            }
        ],
        "confidence": "high",
        "sufficient_information": true,
        "source_count": 3,
        "processing_time_ms": 2340.5,
        "request_id": "uuid-string",
        "timestamp": "2024-01-15T10:30:00"
    }
    ```
    
    **Confidence Levels:**
    - `high`: Strong evidence from multiple sources
    - `medium`: Some evidence but limited sources
    - `low`: Insufficient or unclear information
    
    **Error Responses:**
    - `400`: Invalid query (empty or too long)
    - `422`: Query processing failed
    - `500`: Internal server error
    
    **Use Cases:**
    - Ask questions about uploaded documents
    - Get specific information with source citations
    - Research and fact-checking
    - Document analysis and summarization
    
    **Prerequisites:**
    - At least one document must be uploaded via `/upload`
    - Documents must be successfully processed and indexed
    """
    with time_operation("api_process_query"):
        request_id = generate_request_id()
        start_time = datetime.now()
        
        logger.info(
            "Query processing started",
            request_id=request_id,
            query=request.query
        )
        increment_counter("api_query_requests")
        
        try:
            # Process query through workflow
            result = await workflow.process_query_async(
                query=request.query,
                pdf_path=None  # Use existing documents in vector store
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if result.get("error"):
                raise WorkflowError(f"Query processing failed: {result['error']}")
            
            response = QueryResponse(
                query=request.query,
                answer=result["answer"],
                citations=result["citations"],
                confidence=result["confidence"],
                sufficient_information=result["metadata"].get("answer_metadata", {}).get("sufficient_information", False),
                source_count=result["metadata"].get("answer_metadata", {}).get("source_count", 0),
                processing_time_ms=processing_time,
                request_id=request_id,
                timestamp=get_current_timestamp(),
                metadata=result["metadata"]
            )
            
            # Record success metrics
            increment_counter("api_query_successful")
            
            logger.info(
                "Query processing completed",
                request_id=request_id,
                query=request.query,
                answer_length=len(result["answer"]),
                confidence=result["confidence"],
                processing_time_ms=processing_time
            )
            
            return response
            
        except WorkflowError as e:
            logger.error("Query processing failed", request_id=request_id, error=str(e))
            increment_counter("api_query_processing_errors")
            record_error("WorkflowError", "process_query")
            raise HTTPException(
                status_code=422,
                detail=f"Query processing failed: {str(e)}"
            )
        except Exception as e:
            logger.error("Query processing error", request_id=request_id, error=str(e))
            increment_counter("api_query_errors")
            record_error("Exception", "process_query")
            raise HTTPException(
                status_code=500,
                detail=f"Query processing error: {str(e)}"
            )


@app.post("/query-with-document", response_model=QueryResponse)
async def query_with_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    query: str = Form(..., description="User's question"),
    include_citations: bool = Form(True, description="Whether to include citations"),
    workflow: RAGWorkflow = Depends(get_workflow),
    _: None = Depends(track_request)
):
    """Process a query with a new document upload."""
    request_id = generate_request_id()
    start_time = datetime.now()
    
    logger.info(
        "Query with document processing started",
        request_id=request_id,
        query=query,
        filename=file.filename
    )
    
    # Validate inputs
    if not query or not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    temp_file_path = None
    
    try:
        # Save uploaded file to temporary location
        ensure_directory_exists(settings.upload_path)
        
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.pdf',
            dir=settings.upload_path
        ) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process query with document through workflow
        result = await workflow.process_query_async(
            query=query.strip(),
            pdf_path=temp_file_path
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        if result.get("error"):
            raise WorkflowError(f"Query with document processing failed: {result['error']}")
        
        response = QueryResponse(
            query=query.strip(),
            answer=result["answer"],
            citations=result["citations"],
            confidence=result["confidence"],
            sufficient_information=result["metadata"].get("answer_metadata", {}).get("sufficient_information", False),
            source_count=result["metadata"].get("answer_metadata", {}).get("source_count", 0),
            processing_time_ms=processing_time,
            request_id=request_id,
            timestamp=get_current_timestamp(),
            metadata=result["metadata"]
        )
        
        logger.info(
            "Query with document processing completed",
            request_id=request_id,
            query=query,
            filename=file.filename,
            answer_length=len(result["answer"]),
            confidence=result["confidence"],
            processing_time_ms=processing_time
        )
        
        return response
        
    except WorkflowError as e:
        logger.error("Query with document processing failed", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=422,
            detail=f"Query with document processing failed: {str(e)}"
        )
    except Exception as e:
        logger.error("Query with document error", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Query with document error: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning("Failed to clean up temporary file", error=str(e))


@app.delete("/cache", response_model=Dict[str, str], tags=["management"])
async def clear_cache(
    workflow: RAGWorkflow = Depends(get_workflow),
    _: None = Depends(track_request)
):
    """
    ## Clear All Caches
    
    Clear all system caches including embeddings and query results.
    
    **What Gets Cleared:**
    - Embedding cache (document and query embeddings)
    - Query result cache (previously computed answers)
    - File-based persistent cache
    - In-memory cache entries
    
    **Impact:**
    - Next queries will be slower until cache rebuilds
    - Useful for testing or after configuration changes
    - Frees up memory and disk space
    
    **Example Response:**
    ```json
    {
        "status": "success",
        "message": "All caches cleared successfully",
        "request_id": "uuid-string",
        "timestamp": "2024-01-15T10:30:00"
    }
    ```
    
    **Use Cases:**
    - Clear cache after document updates
    - Free up memory when running low
    - Reset system state for testing
    - Force fresh computation of results
    """
    request_id = generate_request_id()
    
    logger.info("Cache clearing started", request_id=request_id)
    
    try:
        workflow.clear_caches()
        
        logger.info("Cache clearing completed", request_id=request_id)
        
        return {
            "status": "success",
            "message": "All caches cleared successfully",
            "request_id": request_id,
            "timestamp": get_current_timestamp()
        }
        
    except Exception as e:
        logger.error("Cache clearing failed", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@app.delete("/documents", response_model=Dict[str, str], tags=["management"])
async def clear_documents(
    workflow: RAGWorkflow = Depends(get_workflow),
    _: None = Depends(track_request)
):
    """
    ## Clear All Documents
    
    Remove all documents from the vector store and search indices.
    
    **What Gets Cleared:**
    - All document chunks from vector store
    - FAISS vector index
    - BM25 keyword search index
    - Document metadata and embeddings
    
    **Impact:**
    - All previously uploaded documents are removed
    - Queries will return no results until new documents are uploaded
    - Frees up significant memory and disk space
    - Cannot be undone - documents must be re-uploaded
    
    **Example Response:**
    ```json
    {
        "status": "success",
        "message": "All documents cleared from vector store",
        "request_id": "uuid-string",
        "timestamp": "2024-01-15T10:30:00"
    }
    ```
    
    **Use Cases:**
    - Start fresh with new document set
    - Free up storage space
    - Reset system for different use case
    - Clean up after testing
    
    **Warning:**
    This operation cannot be undone. All documents must be re-uploaded.
    """
    request_id = generate_request_id()
    
    logger.info("Document clearing started", request_id=request_id)
    
    try:
        workflow.clear_vector_store()
        
        logger.info("Document clearing completed", request_id=request_id)
        
        return {
            "status": "success",
            "message": "All documents cleared from vector store",
            "request_id": request_id,
            "timestamp": get_current_timestamp()
        }
        
    except Exception as e:
        logger.error("Document clearing failed", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear documents: {str(e)}"
        )


@app.get("/metrics", response_model=Dict[str, Any], tags=["metrics"])
async def get_metrics(workflow: RAGWorkflow = Depends(get_workflow)):
    """
    ## System Metrics Overview
    
    Get comprehensive system metrics and performance statistics.
    
    **Includes:**
    - System resource usage (CPU, memory, uptime)
    - Workflow statistics (documents, chunks, components)
    - Cache performance metrics
    - Complete performance metrics from metrics collector
    
    **Example Response:**
    ```json
    {
        "system": {
            "uptime_seconds": 3600.5,
            "memory_usage_mb": 256.7,
            "memory_percent": 15.2,
            "cpu_percent": 12.5,
            "active_requests": 2,
            "total_requests": 150
        },
        "workflow": {
            "total_chunks": 1250,
            "vector_store_path": "./data/vector_store",
            "enable_caching": true,
            "components": ["DocumentProcessor", "ChunkingEngine", ...]
        },
        "cache": {
            "memory_cache": {
                "total_entries": 45,
                "estimated_size_bytes": 125000,
                "max_size": 1000
            },
            "file_cache_enabled": true
        },
        "performance_metrics": {
            "timing_metrics": {...},
            "counter_metrics": {...},
            "error_metrics": {...}
        }
    }
    ```
    
    **Use Cases:**
    - Monitor system health and performance
    - Track resource usage and capacity planning
    - Analyze cache effectiveness
    - Debug performance issues
    """
    try:
        import psutil
        process = psutil.Process()
        
        # System metrics
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        # Workflow metrics
        workflow_info = workflow.get_workflow_info()
        total_chunks = workflow.vector_store.get_document_count()
        
        # Cache metrics
        cache_stats = workflow.get_cache_stats()
        
        # Enhanced metrics from metrics collector
        metrics_collector = get_metrics_collector()
        all_metrics = metrics_collector.get_all_metrics()
        
        return {
            "system": {
                "uptime_seconds": api_state.get_uptime(),
                "memory_usage_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "cpu_percent": cpu_percent,
                "active_requests": api_state.active_requests,
                "total_requests": api_state.request_count
            },
            "workflow": {
                "total_chunks": total_chunks,
                "vector_store_path": workflow_info["vector_store_path"],
                "enable_caching": workflow_info["enable_caching"],
                "components": list(workflow_info["components"].keys())
            },
            "cache": cache_stats,
            "performance_metrics": all_metrics,
            "timestamp": get_current_timestamp()
        }
        
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )


@app.get("/metrics/performance", response_model=Dict[str, Any])
async def get_performance_metrics():
    """Get detailed performance metrics."""
    try:
        metrics_collector = get_metrics_collector()
        return {
            "performance_summary": metrics_collector.get_performance_summary(),
            "timing_metrics": metrics_collector.get_timing_metrics(),
            "counter_metrics": metrics_collector.get_counter_metrics(),
            "error_metrics": metrics_collector.get_error_metrics(),
            "timestamp": get_current_timestamp()
        }
    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


@app.get("/metrics/system", response_model=Dict[str, Any])
async def get_system_metrics():
    """Get detailed system health metrics."""
    try:
        metrics_collector = get_metrics_collector()
        system_health = metrics_collector.get_system_health()
        
        # Get recent system metrics
        recent_cpu = metrics_collector.get_recent_gauge_values("system_cpu_percent", 300)  # Last 5 minutes
        recent_memory = metrics_collector.get_recent_gauge_values("system_memory_percent", 300)
        
        return {
            "system_health": system_health,
            "recent_cpu_usage": recent_cpu,
            "recent_memory_usage": recent_memory,
            "process_metrics": {
                "memory_rss": metrics_collector.get_recent_gauge_values("process_memory_rss_mb", 300),
                "cpu_percent": metrics_collector.get_recent_gauge_values("process_cpu_percent", 300)
            },
            "timestamp": get_current_timestamp()
        }
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system metrics: {str(e)}"
        )


@app.get("/metrics/workflow", response_model=Dict[str, Any])
async def get_workflow_metrics():
    """Get detailed workflow performance metrics."""
    try:
        metrics_collector = get_metrics_collector()
        
        # Get workflow-specific timing metrics
        timing_metrics = metrics_collector.get_timing_metrics()
        workflow_timings = {k: v for k, v in timing_metrics.items() if k.startswith("workflow_")}
        
        # Get workflow-specific counter metrics
        counter_metrics = metrics_collector.get_counter_metrics()
        workflow_counters = {k: v for k, v in counter_metrics.items() if k.startswith("workflow_")}
        
        return {
            "workflow_timing_metrics": workflow_timings,
            "workflow_counter_metrics": workflow_counters,
            "workflow_error_metrics": {k: v for k, v in metrics_collector.get_error_metrics().items() if "workflow" in k},
            "timestamp": get_current_timestamp()
        }
    except Exception as e:
        logger.error("Failed to get workflow metrics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflow metrics: {str(e)}"
        )


@app.get("/metrics/api", response_model=Dict[str, Any])
async def get_api_metrics():
    """Get detailed API performance metrics."""
    try:
        metrics_collector = get_metrics_collector()
        
        # Get API-specific timing metrics
        timing_metrics = metrics_collector.get_timing_metrics()
        api_timings = {k: v for k, v in timing_metrics.items() if k.startswith("api_")}
        
        # Get API-specific counter metrics
        counter_metrics = metrics_collector.get_counter_metrics()
        api_counters = {k: v for k, v in counter_metrics.items() if k.startswith("api_")}
        
        return {
            "api_timing_metrics": api_timings,
            "api_counter_metrics": api_counters,
            "api_error_metrics": {k: v for k, v in metrics_collector.get_error_metrics().items() if "api" in k},
            "active_requests": api_state.active_requests,
            "total_requests": api_state.request_count,
            "timestamp": get_current_timestamp()
        }
    except Exception as e:
        logger.error("Failed to get API metrics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get API metrics: {str(e)}"
        )


@app.delete("/metrics", response_model=Dict[str, str])
async def reset_metrics():
    """Reset all performance metrics."""
    try:
        metrics_collector = get_metrics_collector()
        metrics_collector.reset_metrics()
        
        logger.info("Performance metrics reset")
        
        return {
            "status": "success",
            "message": "All performance metrics reset successfully",
            "timestamp": get_current_timestamp()
        }
    except Exception as e:
        logger.error("Failed to reset metrics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset metrics: {str(e)}"
        )


# Exception handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_type="HTTPException",
            request_id=generate_request_id(),
            timestamp=get_current_timestamp()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc), exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_type=type(exc).__name__,
            request_id=generate_request_id(),
            timestamp=get_current_timestamp(),
            details={"message": str(exc)}
        ).dict()
    )


# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info("PDF RAG System API starting up")
    
    try:
        # Validate configuration
        from src.config import validate_required_settings
        validate_required_settings(settings)
        
        # Ensure required directories exist
        ensure_directory_exists(settings.vector_store_path)
        ensure_directory_exists(settings.upload_path)
        
        logger.info("PDF RAG System API startup completed")
        
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        raise ConfigurationError(f"Startup failed: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("PDF RAG System API shutting down")
    
    # Clean up resources if needed
    if api_state.workflow:
        try:
            # Any cleanup needed for workflow
            pass
        except Exception as e:
            logger.warning("Cleanup warning during shutdown", error=str(e))
    
    logger.info("PDF RAG System API shutdown completed")


# Development server function

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info"
):
    """Run the FastAPI server."""
    logger.info(
        "Starting PDF RAG System API server",
        host=host,
        port=port,
        reload=reload
    )
    
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    run_server(reload=True)