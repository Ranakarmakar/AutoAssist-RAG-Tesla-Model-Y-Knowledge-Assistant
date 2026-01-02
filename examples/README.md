# PDF RAG System - Examples and Demos

This directory contains comprehensive examples and demonstrations of the PDF RAG System capabilities.

## 📁 Directory Structure

```
examples/
├── README.md                    # This file
├── USAGE_GUIDE.md              # Comprehensive usage guide
├── simple_demo.py              # Simple command-line demo
├── advanced_api_client.py      # Advanced Python client with error handling
├── batch_processing_demo.py    # Batch processing multiple documents/queries
├── performance_test.py         # Performance testing and benchmarking
├── pdf_rag_demo.ipynb         # Interactive Jupyter notebook demo
├── sample_queries.txt         # Sample queries for testing
└── configurations/            # Configuration examples
    ├── development.env        # Development environment config
    ├── production.env         # Production environment config
    └── docker-compose.yml     # Docker deployment example
```

## 🚀 Quick Start

### 1. Start the API Server

```bash
# From the project root directory
python run_api.py
```

The server will start at `http://localhost:8000`. You can verify it's running by visiting `http://localhost:8000/docs` for the interactive API documentation.

### 2. Simple Demo

The easiest way to get started:

```bash
# Upload a document and ask a question
python examples/simple_demo.py data/Owners_Manual.pdf "What are the safety features?"

# Interactive mode for multiple questions
python examples/simple_demo.py data/Owners_Manual.pdf --interactive

# Just interactive mode (using existing documents)
python examples/simple_demo.py --interactive
```

### 3. Jupyter Notebook Demo

For an interactive experience with visualizations:

```bash
# Install notebook dependencies
pip install jupyter matplotlib pandas

# Start Jupyter
jupyter notebook examples/pdf_rag_demo.ipynb
```

## 📚 Example Scripts

### Simple Demo (`simple_demo.py`)

**Purpose:** Basic command-line interface for document upload and querying.

**Features:**
- Document upload with progress feedback
- Single question or interactive mode
- Health checking and error handling
- Simple, user-friendly interface

**Usage Examples:**
```bash
# Basic usage
python examples/simple_demo.py manual.pdf "How do I maintain this?"

# Interactive mode
python examples/simple_demo.py manual.pdf --interactive

# Use different API server
python examples/simple_demo.py manual.pdf "Question?" --url http://remote-server:8000
```

### Advanced API Client (`advanced_api_client.py`)

**Purpose:** Production-ready Python client with comprehensive features.

**Features:**
- Retry logic with exponential backoff
- Batch document processing
- Performance monitoring and benchmarking
- System health monitoring
- Comprehensive error handling

**Usage Examples:**
```python
from examples.advanced_api_client import PDFRAGClient

client = PDFRAGClient()

# Upload and query
upload_result = client.upload_document("document.pdf")
query_result = client.query_documents("What is this about?")

# Batch operations
results = client.batch_upload(["doc1.pdf", "doc2.pdf", "doc3.pdf"])

# Performance monitoring
metrics = client.get_metrics()
benchmark = client.benchmark_queries(["Query 1", "Query 2"], iterations=3)
```

### Batch Processing (`batch_processing_demo.py`)

**Purpose:** Process multiple documents and queries efficiently.

**Features:**
- Concurrent document upload
- Batch query processing
- Comprehensive reporting (JSON/CSV)
- Performance analysis
- Progress monitoring

**Usage Examples:**
```bash
# Process directory of PDFs with queries from file
python examples/batch_processing_demo.py \
  --docs-dir ./pdfs \
  --queries-file examples/sample_queries.txt \
  --output results.json \
  --csv results.csv

# Process specific files with inline queries
python examples/batch_processing_demo.py \
  --docs manual1.pdf manual2.pdf \
  --queries "What is warranty?" "How to maintain?" \
  --workers 5
```

### Performance Testing (`performance_test.py`)

**Purpose:** Comprehensive performance testing and benchmarking.

**Features:**
- Concurrent user simulation
- Load testing with ramp-up
- Cache performance evaluation
- Response time analysis
- System resource monitoring
- Performance visualization

**Usage Examples:**
```bash
# Basic concurrent test
python examples/performance_test.py --concurrent 5 --queries 20

# Load test with gradual ramp-up
python examples/performance_test.py \
  --load-test \
  --users 10 \
  --ramp-up 30 \
  --duration 60 \
  --plot

# Cache performance test
python examples/performance_test.py --cache-test --iterations 10

# Full test suite
python examples/performance_test.py --full-test --output perf_results.json
```

### Jupyter Notebook (`pdf_rag_demo.ipynb`)

**Purpose:** Interactive demonstration with visualizations and explanations.

**Features:**
- Step-by-step walkthrough
- Interactive query interface
- Performance visualization
- System monitoring charts
- Educational content

**Sections:**
1. System health check
2. Document upload and processing
3. Query processing with citations
4. Performance analysis
5. Interactive query interface
6. System management
7. Summary and next steps

## 📋 Sample Queries (`sample_queries.txt`)

Pre-written queries for testing different aspects of document understanding:

- **General Information:** "What is this document about?"
- **Safety & Compliance:** "What are the safety features?"
- **Technical Specs:** "What are the technical specifications?"
- **Operation & Maintenance:** "How do I maintain this system?"
- **Warranty & Support:** "What warranty information is provided?"
- **Installation & Setup:** "How do I install this system?"
- **Features & Capabilities:** "What are the main features?"

## ⚙️ Configuration Examples

### Development Environment (`configurations/development.env`)

Optimized for development with debug logging and relaxed timeouts:

```env
# Development configuration
LOG_LEVEL=DEBUG
CHUNK_SIZE=800
CHUNK_OVERLAP=100
RETRIEVAL_TOP_K=15
ENABLE_CACHING=true
CACHE_MEMORY_SIZE=500
```

### Production Environment (`configurations/production.env`)

Optimized for production with performance tuning:

```env
# Production configuration
LOG_LEVEL=INFO
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=25
ENABLE_CACHING=true
CACHE_MEMORY_SIZE=5000
EMBEDDING_CACHE_TTL=2592000  # 30 days
```

### Docker Deployment (`configurations/docker-compose.yml`)

Complete Docker setup with Redis caching and monitoring:

```yaml
version: '3.8'
services:
  pdf-rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

## 🔧 Usage Patterns

### Basic Workflow

1. **Start the API server**
2. **Upload documents** using any of the demo scripts
3. **Ask questions** about the uploaded content
4. **Monitor performance** using metrics endpoints
5. **Manage system** (clear cache, documents) as needed

### Development Workflow

1. Use `simple_demo.py` for quick testing
2. Use `advanced_api_client.py` for integration development
3. Use `performance_test.py` to validate performance
4. Use Jupyter notebook for interactive exploration

### Production Workflow

1. Use `batch_processing_demo.py` for bulk operations
2. Monitor with `performance_test.py` load testing
3. Use advanced client patterns for robust applications
4. Deploy with Docker configuration examples

## 📊 Performance Expectations

Based on testing with the Tesla Owner's Manual (313 pages):

- **Document Upload:** ~15-30 seconds for large PDFs
- **Query Processing:** ~2-5 seconds for complex queries
- **Cache Hit:** ~200-500ms for repeated queries
- **Concurrent Users:** Supports 5-10 concurrent users effectively
- **Memory Usage:** ~200-500MB depending on document size

## 🛠️ Troubleshooting

### Common Issues

1. **API Server Not Running**
   ```bash
   # Check if server is running
   curl http://localhost:8000/health
   
   # Start server if needed
   python run_api.py
   ```

2. **Document Upload Fails**
   - Check file size (max 50MB)
   - Verify PDF format
   - Check available disk space

3. **Queries Return No Results**
   - Verify documents are uploaded
   - Check system health
   - Clear cache and retry

4. **Performance Issues**
   - Monitor system metrics
   - Check memory usage
   - Consider clearing cache
   - Reduce concurrent users

### Getting Help

- **API Documentation:** `http://localhost:8000/docs`
- **System Health:** `http://localhost:8000/health`
- **Metrics:** `http://localhost:8000/metrics`
- **Usage Guide:** `examples/USAGE_GUIDE.md`

## 🚀 Next Steps

1. **Try the Examples:** Start with `simple_demo.py` and work your way up
2. **Read the Usage Guide:** Comprehensive documentation in `USAGE_GUIDE.md`
3. **Explore the API:** Interactive docs at `/docs` when server is running
4. **Customize Configuration:** Modify settings for your use case
5. **Build Your Application:** Use the advanced client as a starting point

## 📝 Contributing

When adding new examples:

1. Follow the existing code style and patterns
2. Include comprehensive error handling
3. Add usage examples and documentation
4. Test with various document types and queries
5. Update this README with new examples

---

**Happy RAG-ing! 🎉**