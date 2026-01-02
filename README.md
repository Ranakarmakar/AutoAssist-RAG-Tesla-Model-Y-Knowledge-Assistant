# PDF RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for PDF documents using LangChain, LangGraph, and Groq LLM.

## Features

- **Document Processing**: PDF parsing and text extraction using LangChain's PyPDFLoader
- **Intelligent Chunking**: Recursive text splitting with configurable chunk sizes and overlap
- **Hybrid Retrieval**: Combines BM25 keyword search with semantic vector search using FAISS
- **Query Enhancement**: Groq LLM-powered query rewriting for better retrieval
- **Reranking**: Cross-encoder models for relevance-based result reordering
- **Answer Generation**: Contextual answer generation with citations using Groq LLM
- **Workflow Orchestration**: LangGraph-based workflow management with error handling
- **REST API**: FastAPI-based web API with comprehensive endpoints
- **Comprehensive Testing**: 259+ tests covering all components and integration scenarios

## Architecture

The system uses a modular architecture with the following components:

1. **Document Processor**: Handles PDF loading and validation
2. **Chunking Engine**: Splits documents into manageable chunks
3. **Embedding Model**: Generates vector embeddings using sentence-transformers
4. **Vector Store**: FAISS-based vector storage with persistence
5. **Query Rewriter**: Enhances queries using Groq LLM
6. **Hybrid Retriever**: Combines keyword and semantic search
7. **Reranker**: Reorders results by relevance using cross-encoder models
8. **Answer Generator**: Generates answers with citations using Groq LLM
9. **RAG Workflow**: LangGraph orchestration of the complete pipeline
10. **REST API**: FastAPI web interface for all functionality

## Quick Start

### Prerequisites

- Python 3.8+
- Groq API key (get from [Groq Console](https://console.groq.com/))

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-rag-system
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

### Running the API Server

1. Start the API server:
```bash
python run_api.py
```

2. The API will be available at:
   - Main API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### API Usage Examples

#### Upload and Query a Document
```bash
# Upload a PDF document
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Query the uploaded document
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main topics in this document?", "include_citations": true}'
```

#### Query with Document Upload
```bash
curl -X POST "http://localhost:8000/query-with-document" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "query=What are the safety features?" \
  -F "include_citations=true" \
  -F "file=@document.pdf"
```

#### Python Client Example
```python
import requests

# Query with document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/query-with-document",
        files={"file": ("document.pdf", f, "application/pdf")},
        data={
            "query": "What are the main topics?",
            "include_citations": "true"
        }
    )

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Citations: {len(result['citations'])}")
```

### Using the Python Client

Run the example client:
```bash
python api_client_example.py
```

This will demonstrate:
- Health checking
- Document upload
- Querying with and without documents
- System metrics
- Document clearing

## API Endpoints

### Core Endpoints

- `GET /` - Root endpoint with basic information
- `GET /health` - Health check with system status
- `GET /metrics` - System metrics and statistics

### Document Management

- `POST /upload` - Upload and process a PDF document
- `DELETE /documents` - Clear all documents from the system

### Query Processing

- `POST /query` - Query using existing documents
- `POST /query-with-document` - Query with simultaneous document upload

## Configuration

Key configuration options in `src/config.py`:

```python
# LLM Configuration
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.1

# Retrieval Configuration
RETRIEVAL_TOP_K = 20
RERANKER_TOP_K = 5

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector Store
VECTOR_STORE_PATH = "./data/vector_store"
```

## Testing

Run the complete test suite:
```bash
# All tests
pytest

# Specific test categories
pytest tests/                    # Unit tests
pytest test_*_integration.py     # Integration tests
pytest test_complete_rag_pipeline.py  # End-to-end tests
```

Current test status: **259 tests passing**

## Development

### Project Structure

```
pdf-rag-system/
├── src/                     # Source code
│   ├── api.py              # FastAPI REST API
│   ├── rag_workflow.py     # LangGraph workflow orchestration
│   ├── document_processor.py
│   ├── chunking_engine.py
│   ├── embedding_model.py
│   ├── vector_store.py
│   ├── query_rewriter.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   ├── answer_generator.py
│   ├── config.py
│   ├── logging_config.py
│   ├── exceptions.py
│   └── utils.py
├── tests/                   # Unit tests
├── test_*_integration.py    # Integration tests
├── data/                    # Data directory
├── run_api.py              # API server launcher
├── api_client_example.py   # Example API client
└── requirements.txt
```

### Adding New Features

1. Implement the feature in the appropriate module
2. Add comprehensive unit tests in `tests/`
3. Add integration tests as `test_*_integration.py`
4. Update the workflow in `rag_workflow.py` if needed
5. Add API endpoints in `api.py` if required
6. Update documentation

## Performance

The system is optimized for performance with:

- **Efficient Chunking**: Configurable chunk sizes and overlap
- **Hybrid Retrieval**: Combines fast keyword search with semantic search
- **Reranking**: Only reranks top candidates for efficiency
- **Caching**: Component-level caching for repeated operations
- **Async Support**: Asynchronous query processing in the API

Typical performance (Tesla Owner's Manual, 313 pages):
- Document processing: ~15-20 seconds
- Query processing: ~2-5 seconds
- Memory usage: ~200-500 MB

## Troubleshooting

### Common Issues

1. **Groq API Key**: Ensure your Groq API key is set in `.env`
2. **Memory Issues**: Reduce chunk size or limit document pages for large PDFs
3. **Slow Performance**: Check internet connection for Groq API calls
4. **Import Errors**: Ensure virtual environment is activated and dependencies installed

### Logging

The system uses structured logging. Check logs for detailed error information:
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python run_api.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.