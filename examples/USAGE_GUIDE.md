# PDF RAG System - Comprehensive Usage Guide

This guide provides detailed examples and best practices for using the PDF RAG System in various scenarios.

## Table of Contents

1. [Quick Start](#quick-start)
2. [API Usage Examples](#api-usage-examples)
3. [Python Client Examples](#python-client-examples)
4. [Configuration Guide](#configuration-guide)
5. [Performance Optimization](#performance-optimization)
6. [Monitoring and Metrics](#monitoring-and-metrics)
7. [Error Handling](#error-handling)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Basic Setup

```bash
# Clone and setup
git clone <repository-url>
cd pdf-rag-system
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your Groq API key

# Start the server
python run_api.py
```

### 2. First API Call

```bash
# Check system health
curl http://localhost:8000/health

# Upload a document
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"

# Ask a question
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "include_citations": true}'
```

## API Usage Examples

### Document Upload

#### Basic Upload
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@manual.pdf"
```

#### Upload with Error Handling
```bash
response=$(curl -s -w "%{http_code}" -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@manual.pdf")

http_code="${response: -3}"
body="${response%???}"

if [ "$http_code" -eq 200 ]; then
    echo "Upload successful: $body"
else
    echo "Upload failed with code $http_code: $body"
fi
```

### Query Processing

#### Simple Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the safety features?",
    "include_citations": true,
    "check_sufficiency": true
  }'
```

#### Query with Document Upload
```bash
curl -X POST "http://localhost:8000/query-with-document" \
  -H "Content-Type: multipart/form-data" \
  -F "query=What are the main topics in this document?" \
  -F "include_citations=true" \
  -F "file=@document.pdf"
```

#### Batch Queries
```bash
# Create a script for multiple queries
queries=("What is the warranty?" "How do I maintain this?" "What are safety features?")

for query in "${queries[@]}"; do
    echo "Processing: $query"
    curl -X POST "http://localhost:8000/query" \
      -H "Content-Type: application/json" \
      -d "{\"query\": \"$query\", \"include_citations\": true}" \
      | jq '.answer' | head -c 100
    echo -e "\n---"
done
```

### System Management

#### Get System Metrics
```bash
# Basic metrics
curl http://localhost:8000/metrics | jq '.'

# Performance metrics
curl http://localhost:8000/metrics/performance | jq '.performance_summary'

# System health
curl http://localhost:8000/metrics/system | jq '.system_health'
```

#### Cache Management
```bash
# Clear cache
curl -X DELETE http://localhost:8000/cache

# Clear all documents
curl -X DELETE http://localhost:8000/documents
```

## Python Client Examples

### Basic Client Usage

```python
import requests
import json

class PDFRAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def upload_document(self, file_path):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{self.base_url}/upload", files=files)
            response.raise_for_status()
            return response.json()
    
    def query(self, question, include_citations=True):
        data = {
            "query": question,
            "include_citations": include_citations,
            "check_sufficiency": True
        }
        response = requests.post(f"{self.base_url}/query", json=data)
        response.raise_for_status()
        return response.json()

# Usage
client = PDFRAGClient()

# Upload document
upload_result = client.upload_document("manual.pdf")
print(f"Uploaded: {upload_result['chunks_created']} chunks created")

# Ask questions
result = client.query("What are the main features?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Citations: {len(result['citations'])}")
```

### Advanced Client with Error Handling

```python
import requests
import time
from typing import Optional, Dict, Any

class AdvancedPDFRAGClient:
    def __init__(self, base_url="http://localhost:8000", timeout=30, max_retries=3):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
    
    def _make_request(self, method, endpoint, **kwargs):
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Request failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health."""
        return self._make_request("GET", "/health")
    
    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """Upload a PDF document."""
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'application/pdf')}
            return self._make_request("POST", "/upload", files=files)
    
    def query(self, question: str, include_citations: bool = True) -> Dict[str, Any]:
        """Query the documents."""
        data = {
            "query": question,
            "include_citations": include_citations,
            "check_sufficiency": True
        }
        return self._make_request("POST", "/query", json=data)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return self._make_request("GET", "/metrics")
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear system cache."""
        return self._make_request("DELETE", "/cache")

# Usage with error handling
try:
    client = AdvancedPDFRAGClient()
    
    # Check health first
    health = client.health_check()
    print(f"System status: {health['status']}")
    
    # Upload and query
    upload_result = client.upload_document("document.pdf")
    query_result = client.query("What is this about?")
    
    print(f"Answer: {query_result['answer'][:100]}...")
    
except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Async Client Example

```python
import asyncio
import aiohttp
from typing import Dict, Any

class AsyncPDFRAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    async def upload_document(self, session: aiohttp.ClientSession, file_path: str) -> Dict[str, Any]:
        """Upload document asynchronously."""
        with open(file_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=file_path, content_type='application/pdf')
            
            async with session.post(f"{self.base_url}/upload", data=data) as response:
                response.raise_for_status()
                return await response.json()
    
    async def query(self, session: aiohttp.ClientSession, question: str) -> Dict[str, Any]:
        """Query documents asynchronously."""
        data = {
            "query": question,
            "include_citations": True,
            "check_sufficiency": True
        }
        
        async with session.post(f"{self.base_url}/query", json=data) as response:
            response.raise_for_status()
            return await response.json()
    
    async def batch_queries(self, questions: list) -> list:
        """Process multiple queries concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = [self.query(session, q) for q in questions]
            return await asyncio.gather(*tasks)

# Usage
async def main():
    client = AsyncPDFRAGClient()
    
    # Process multiple queries concurrently
    questions = [
        "What are the main features?",
        "How does it work?",
        "What are the requirements?"
    ]
    
    results = await client.batch_queries(questions)
    
    for i, result in enumerate(results):
        print(f"Q{i+1}: {questions[i]}")
        print(f"A{i+1}: {result['answer'][:100]}...")
        print(f"Confidence: {result['confidence']}\n")

# Run async example
# asyncio.run(main())
```

## Configuration Guide

### Environment Variables

```bash
# Core Configuration
GROQ_API_KEY=your_api_key_here
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# Performance Tuning
CHUNK_SIZE=1000          # Larger chunks = more context, slower processing
CHUNK_OVERLAP=200        # More overlap = better context continuity
RETRIEVAL_TOP_K=20       # More results = better recall, slower processing
RERANK_TOP_K=5          # Final number of sources for answer generation

# Caching
ENABLE_CACHING=true
CACHE_MEMORY_SIZE=1000   # Number of items in memory cache
EMBEDDING_CACHE_TTL=604800  # 1 week
QUERY_CACHE_TTL=3600     # 1 hour

# Storage
VECTOR_STORE_PATH=./data/vector_store
UPLOAD_PATH=./data/uploads
CACHE_DIR=./data/cache
```

### Custom Configuration Class

```python
from src.config import Settings

# Custom configuration
class CustomSettings(Settings):
    # Override defaults
    chunk_size: int = 1500
    chunk_overlap: int = 300
    retrieval_top_k: int = 25
    
    # Custom settings
    max_file_size_mb: int = 100
    enable_debug_logging: bool = True

# Use custom settings
settings = CustomSettings()
```

## Performance Optimization

### Chunking Strategy

```python
# For technical documents
CHUNK_SIZE=1200
CHUNK_OVERLAP=150

# For narrative documents
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# For code documentation
CHUNK_SIZE=1500
CHUNK_OVERLAP=200
```

### Retrieval Tuning

```python
# Balanced (default)
RETRIEVAL_TOP_K=20
RERANK_TOP_K=5
BM25_WEIGHT=0.3
SEMANTIC_WEIGHT=0.7

# Favor keyword search
BM25_WEIGHT=0.5
SEMANTIC_WEIGHT=0.5

# Favor semantic search
BM25_WEIGHT=0.2
SEMANTIC_WEIGHT=0.8
```

### Caching Strategy

```python
# Development
CACHE_MEMORY_SIZE=500
EMBEDDING_CACHE_TTL=3600    # 1 hour
QUERY_CACHE_TTL=1800        # 30 minutes

# Production
CACHE_MEMORY_SIZE=5000
EMBEDDING_CACHE_TTL=2592000 # 30 days
QUERY_CACHE_TTL=7200        # 2 hours
```

## Monitoring and Metrics

### Health Monitoring Script

```python
import requests
import time
import json

def monitor_system(duration_minutes=10, interval_seconds=30):
    """Monitor system health over time."""
    base_url = "http://localhost:8000"
    end_time = time.time() + (duration_minutes * 60)
    
    print(f"Monitoring system for {duration_minutes} minutes...")
    
    while time.time() < end_time:
        try:
            # Get health status
            health = requests.get(f"{base_url}/health").json()
            
            # Get metrics
            metrics = requests.get(f"{base_url}/metrics").json()
            
            print(f"Time: {time.strftime('%H:%M:%S')}")
            print(f"  Status: {health['status']}")
            print(f"  Memory: {health['memory_usage_mb']:.1f}MB")
            print(f"  Active Requests: {health['active_requests']}")
            print(f"  Total Requests: {metrics['system']['total_requests']}")
            print(f"  Cache Enabled: {metrics['workflow']['enable_caching']}")
            print("-" * 40)
            
        except Exception as e:
            print(f"Monitoring error: {e}")
        
        time.sleep(interval_seconds)

# Run monitoring
# monitor_system(duration_minutes=5, interval_seconds=10)
```

### Performance Benchmarking

```python
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

def benchmark_queries(queries, concurrent_requests=1, iterations=3):
    """Benchmark query performance."""
    base_url = "http://localhost:8000"
    
    def single_query(query):
        start_time = time.time()
        response = requests.post(f"{base_url}/query", json={"query": query})
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            return {
                "query": query,
                "time": end_time - start_time,
                "processing_time_ms": result['processing_time_ms'],
                "confidence": result['confidence'],
                "citations": len(result['citations'])
            }
        else:
            return {"query": query, "error": response.status_code}
    
    all_results = []
    
    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}/{iterations}")
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            results = list(executor.map(single_query, queries))
            all_results.extend(results)
    
    # Calculate statistics
    times = [r['time'] for r in all_results if 'time' in r]
    processing_times = [r['processing_time_ms'] for r in all_results if 'processing_time_ms' in r]
    
    print(f"\nBenchmark Results:")
    print(f"  Total queries: {len(all_results)}")
    print(f"  Successful: {len(times)}")
    print(f"  Average time: {statistics.mean(times):.2f}s")
    print(f"  Median time: {statistics.median(times):.2f}s")
    print(f"  Min time: {min(times):.2f}s")
    print(f"  Max time: {max(times):.2f}s")
    print(f"  Average processing time: {statistics.mean(processing_times):.1f}ms")

# Example usage
test_queries = [
    "What are the main features?",
    "How does it work?",
    "What are the requirements?",
    "What is the warranty?"
]

# benchmark_queries(test_queries, concurrent_requests=2, iterations=3)
```

## Error Handling

### Common Error Scenarios

```python
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

def robust_api_call(endpoint, method="GET", **kwargs):
    """Make API call with comprehensive error handling."""
    base_url = "http://localhost:8000"
    
    try:
        response = requests.request(method, f"{base_url}{endpoint}", 
                                  timeout=30, **kwargs)
        
        # Handle HTTP errors
        if response.status_code == 400:
            print("Bad request - check your input parameters")
        elif response.status_code == 413:
            print("File too large - reduce file size or split document")
        elif response.status_code == 422:
            print("Processing failed - check document format or content")
        elif response.status_code == 500:
            print("Server error - check system health and logs")
        else:
            response.raise_for_status()
            return response.json()
    
    except Timeout:
        print("Request timed out - try again or increase timeout")
    except ConnectionError:
        print("Cannot connect to server - check if API is running")
    except RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return None

# Usage examples
result = robust_api_call("/health")
result = robust_api_call("/query", method="POST", 
                        json={"query": "test", "include_citations": True})
```

### Retry Logic

```python
import time
import random

def retry_with_backoff(func, max_retries=3, base_delay=1, max_delay=60):
    """Retry function with exponential backoff and jitter."""
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            total_delay = delay + jitter
            
            print(f"Attempt {attempt + 1} failed: {e}")
            print(f"Retrying in {total_delay:.1f} seconds...")
            time.sleep(total_delay)

# Usage
def upload_document():
    with open("document.pdf", "rb") as f:
        files = {"file": f}
        response = requests.post("http://localhost:8000/upload", files=files)
        response.raise_for_status()
        return response.json()

# Retry upload with backoff
result = retry_with_backoff(upload_document, max_retries=3)
```

## Production Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data/vector_store /app/data/uploads /app/data/cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "run_api.py"]
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pdf-rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pdf-rag-system
  template:
    metadata:
      labels:
        app: pdf-rag-system
    spec:
      containers:
      - name: pdf-rag-api
        image: pdf-rag-system:latest
        ports:
        - containerPort: 8000
        env:
        - name: GROQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: pdf-rag-secrets
              key: groq-api-key
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: pdf-rag-data-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: pdf-rag-service
spec:
  selector:
    app: pdf-rag-system
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Load Balancer Configuration

```nginx
# nginx.conf
upstream pdf_rag_backend {
    server pdf-rag-1:8000;
    server pdf-rag-2:8000;
    server pdf-rag-3:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://pdf_rag_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Timeouts for long-running requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # File upload size limit
        client_max_body_size 50M;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://pdf_rag_backend/health;
        access_log off;
    }
}
```

## Troubleshooting

### Common Issues and Solutions

#### 1. API Server Won't Start

```bash
# Check if port is in use
lsof -i :8000

# Check environment variables
env | grep GROQ

# Check logs
python run_api.py 2>&1 | tee api.log
```

#### 2. Document Upload Fails

```bash
# Check file size
ls -lh document.pdf

# Check file format
file document.pdf

# Test with curl
curl -v -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

#### 3. Queries Return No Results

```bash
# Check if documents are loaded
curl http://localhost:8000/health | jq '.total_chunks'

# Check system metrics
curl http://localhost:8000/metrics | jq '.workflow'

# Clear cache and try again
curl -X DELETE http://localhost:8000/cache
```

#### 4. Performance Issues

```bash
# Check system resources
curl http://localhost:8000/metrics/system | jq '.system_health'

# Monitor memory usage
curl http://localhost:8000/metrics | jq '.system.memory_usage_mb'

# Check cache hit rates
curl http://localhost:8000/metrics | jq '.cache'
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python run_api.py

# Or in .env file
LOG_LEVEL=DEBUG
```

### Log Analysis

```bash
# Monitor logs in real-time
tail -f api.log | grep ERROR

# Search for specific errors
grep "WorkflowError" api.log

# Analyze performance
grep "processing_time_ms" api.log | tail -10
```

This comprehensive usage guide covers the most common scenarios and provides practical examples for integrating the PDF RAG System into your applications. For additional help, refer to the API documentation at `/docs` when the server is running.