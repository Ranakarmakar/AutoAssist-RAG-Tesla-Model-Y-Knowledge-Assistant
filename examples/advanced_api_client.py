"""
Advanced API Client Example for PDF RAG System

This example demonstrates advanced usage patterns including:
- Error handling and retries
- Batch document processing
- Performance monitoring
- Async operations
- Custom configurations
"""

import asyncio
import aiohttp
import requests
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFRAGClient:
    """Advanced client for PDF RAG System API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        
    def health_check(self) -> Dict[str, Any]:
        """Check system health and readiness."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    def upload_document(self, file_path: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Upload a document with retry logic.
        
        Args:
            file_path: Path to PDF file
            max_retries: Maximum number of retry attempts
            
        Returns:
            Upload response data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.suffix.lower() == '.pdf':
            raise ValueError("Only PDF files are supported")
        
        for attempt in range(max_retries + 1):
            try:
                with open(file_path, 'rb') as f:
                    files = {'file': (file_path.name, f, 'application/pdf')}
                    response = self.session.post(
                        f"{self.base_url}/upload",
                        files=files
                    )
                    response.raise_for_status()
                    
                result = response.json()
                logger.info(f"Document uploaded successfully: {file_path.name}")
                logger.info(f"  - Pages processed: {result['pages_processed']}")
                logger.info(f"  - Chunks created: {result['chunks_created']}")
                logger.info(f"  - Processing time: {result['processing_time_ms']:.1f}ms")
                
                return result
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Upload attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Upload failed after {max_retries + 1} attempts: {e}")
                    raise
    
    def query_documents(
        self,
        query: str,
        include_citations: bool = True,
        check_sufficiency: bool = True
    ) -> Dict[str, Any]:
        """
        Query the uploaded documents.
        
        Args:
            query: Question to ask
            include_citations: Whether to include source citations
            check_sufficiency: Whether to check information sufficiency
            
        Returns:
            Query response data
        """
        payload = {
            "query": query,
            "include_citations": include_citations,
            "check_sufficiency": check_sufficiency
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/query",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Query processed successfully")
            logger.info(f"  - Query: {query}")
            logger.info(f"  - Answer length: {len(result['answer'])} characters")
            logger.info(f"  - Confidence: {result['confidence']}")
            logger.info(f"  - Citations: {len(result['citations'])}")
            logger.info(f"  - Processing time: {result['processing_time_ms']:.1f}ms")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def query_with_document(
        self,
        query: str,
        file_path: str,
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Query with simultaneous document upload.
        
        Args:
            query: Question to ask
            file_path: Path to PDF file
            include_citations: Whether to include citations
            
        Returns:
            Query response data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'application/pdf')}
                data = {
                    'query': query,
                    'include_citations': str(include_citations).lower()
                }
                
                response = self.session.post(
                    f"{self.base_url}/query-with-document",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                
            result = response.json()
            logger.info(f"Query with document processed successfully")
            logger.info(f"  - Document: {file_path.name}")
            logger.info(f"  - Query: {query}")
            logger.info(f"  - Answer length: {len(result['answer'])} characters")
            logger.info(f"  - Processing time: {result['processing_time_ms']:.1f}ms")
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Query with document failed: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        try:
            response = self.session.get(f"{self.base_url}/metrics")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get metrics: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        try:
            response = self.session.get(f"{self.base_url}/metrics/performance")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get performance metrics: {e}")
            raise
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear all system caches."""
        try:
            response = self.session.delete(f"{self.base_url}/cache")
            response.raise_for_status()
            result = response.json()
            logger.info("Cache cleared successfully")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to clear cache: {e}")
            raise
    
    def clear_documents(self) -> Dict[str, Any]:
        """Clear all documents from the system."""
        try:
            response = self.session.delete(f"{self.base_url}/documents")
            response.raise_for_status()
            result = response.json()
            logger.info("Documents cleared successfully")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to clear documents: {e}")
            raise
    
    def batch_upload(self, file_paths: List[str], max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Upload multiple documents with concurrency control.
        
        Args:
            file_paths: List of PDF file paths
            max_concurrent: Maximum concurrent uploads
            
        Returns:
            List of upload results
        """
        results = []
        
        # Process in batches to avoid overwhelming the server
        for i in range(0, len(file_paths), max_concurrent):
            batch = file_paths[i:i + max_concurrent]
            batch_results = []
            
            for file_path in batch:
                try:
                    result = self.upload_document(file_path)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to upload {file_path}: {e}")
                    batch_results.append({"error": str(e), "file_path": file_path})
            
            results.extend(batch_results)
            
            # Small delay between batches
            if i + max_concurrent < len(file_paths):
                time.sleep(1)
        
        return results
    
    def benchmark_queries(self, queries: List[str], iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark query performance.
        
        Args:
            queries: List of queries to benchmark
            iterations: Number of iterations per query
            
        Returns:
            Benchmark results
        """
        results = {
            "queries": [],
            "summary": {
                "total_queries": len(queries) * iterations,
                "total_time": 0,
                "average_time": 0,
                "min_time": float('inf'),
                "max_time": 0
            }
        }
        
        all_times = []
        
        for query in queries:
            query_results = {
                "query": query,
                "iterations": [],
                "average_time": 0,
                "min_time": float('inf'),
                "max_time": 0
            }
            
            for i in range(iterations):
                start_time = time.time()
                try:
                    result = self.query_documents(query)
                    end_time = time.time()
                    
                    query_time = end_time - start_time
                    all_times.append(query_time)
                    
                    iteration_result = {
                        "iteration": i + 1,
                        "time_seconds": query_time,
                        "processing_time_ms": result['processing_time_ms'],
                        "confidence": result['confidence'],
                        "citations": len(result['citations'])
                    }
                    
                    query_results["iterations"].append(iteration_result)
                    query_results["min_time"] = min(query_results["min_time"], query_time)
                    query_results["max_time"] = max(query_results["max_time"], query_time)
                    
                except Exception as e:
                    logger.error(f"Query failed during benchmark: {e}")
                    query_results["iterations"].append({
                        "iteration": i + 1,
                        "error": str(e)
                    })
            
            # Calculate averages
            successful_times = [it["time_seconds"] for it in query_results["iterations"] if "time_seconds" in it]
            if successful_times:
                query_results["average_time"] = sum(successful_times) / len(successful_times)
            
            results["queries"].append(query_results)
        
        # Calculate summary statistics
        if all_times:
            results["summary"]["total_time"] = sum(all_times)
            results["summary"]["average_time"] = sum(all_times) / len(all_times)
            results["summary"]["min_time"] = min(all_times)
            results["summary"]["max_time"] = max(all_times)
        
        return results
    
    def monitor_system(self, duration_seconds: int = 60, interval_seconds: int = 5) -> List[Dict[str, Any]]:
        """
        Monitor system metrics over time.
        
        Args:
            duration_seconds: How long to monitor
            interval_seconds: Interval between measurements
            
        Returns:
            List of metric snapshots
        """
        snapshots = []
        start_time = time.time()
        
        logger.info(f"Starting system monitoring for {duration_seconds} seconds...")
        
        while time.time() - start_time < duration_seconds:
            try:
                metrics = self.get_metrics()
                snapshot = {
                    "timestamp": time.time(),
                    "uptime": metrics["system"]["uptime_seconds"],
                    "memory_mb": metrics["system"]["memory_usage_mb"],
                    "cpu_percent": metrics["system"]["cpu_percent"],
                    "active_requests": metrics["system"]["active_requests"],
                    "total_requests": metrics["system"]["total_requests"],
                    "total_chunks": metrics["workflow"]["total_chunks"]
                }
                snapshots.append(snapshot)
                
                logger.info(f"Snapshot: Memory={snapshot['memory_mb']:.1f}MB, "
                           f"CPU={snapshot['cpu_percent']:.1f}%, "
                           f"Requests={snapshot['active_requests']}")
                
            except Exception as e:
                logger.error(f"Failed to get metrics snapshot: {e}")
            
            time.sleep(interval_seconds)
        
        logger.info(f"Monitoring completed. Collected {len(snapshots)} snapshots.")
        return snapshots


def main():
    """Demonstrate advanced API client usage."""
    client = PDFRAGClient()
    
    print("=== PDF RAG System - Advanced API Client Demo ===\n")
    
    # 1. Health Check
    print("1. Checking system health...")
    try:
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Memory: {health['memory_usage_mb']:.1f}MB")
        print(f"   Uptime: {health['uptime_seconds']:.1f}s")
        print(f"   Total chunks: {health['total_chunks']}")
    except Exception as e:
        print(f"   Health check failed: {e}")
        return
    
    # 2. Upload Document (if available)
    print("\n2. Uploading document...")
    test_pdf = Path("data/Owners_Manual.pdf")
    if test_pdf.exists():
        try:
            upload_result = client.upload_document(str(test_pdf))
            print(f"   Upload successful: {upload_result['filename']}")
        except Exception as e:
            print(f"   Upload failed: {e}")
    else:
        print(f"   Test PDF not found: {test_pdf}")
    
    # 3. Query Examples
    print("\n3. Running sample queries...")
    sample_queries = [
        "What are the main safety features?",
        "How do I charge the vehicle?",
        "What is the warranty information?"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n   Query {i}: {query}")
        try:
            result = client.query_documents(query)
            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Confidence: {result['confidence']}")
            print(f"   Citations: {len(result['citations'])}")
        except Exception as e:
            print(f"   Query failed: {e}")
    
    # 4. Performance Benchmark
    print("\n4. Running performance benchmark...")
    try:
        benchmark_queries = sample_queries[:2]  # Use first 2 queries
        benchmark_results = client.benchmark_queries(benchmark_queries, iterations=2)
        
        print(f"   Total queries: {benchmark_results['summary']['total_queries']}")
        print(f"   Average time: {benchmark_results['summary']['average_time']:.2f}s")
        print(f"   Min time: {benchmark_results['summary']['min_time']:.2f}s")
        print(f"   Max time: {benchmark_results['summary']['max_time']:.2f}s")
        
    except Exception as e:
        print(f"   Benchmark failed: {e}")
    
    # 5. System Metrics
    print("\n5. Getting system metrics...")
    try:
        metrics = client.get_metrics()
        print(f"   System uptime: {metrics['system']['uptime_seconds']:.1f}s")
        print(f"   Memory usage: {metrics['system']['memory_usage_mb']:.1f}MB")
        print(f"   Total requests: {metrics['system']['total_requests']}")
        print(f"   Cache enabled: {metrics['workflow']['enable_caching']}")
        
        if 'performance_metrics' in metrics:
            perf = metrics['performance_metrics']
            if 'timing_metrics' in perf:
                print(f"   Timing metrics: {len(perf['timing_metrics'])} operations tracked")
            if 'counter_metrics' in perf:
                print(f"   Counter metrics: {len(perf['counter_metrics'])} counters tracked")
        
    except Exception as e:
        print(f"   Metrics failed: {e}")
    
    # 6. Cache Management
    print("\n6. Testing cache management...")
    try:
        # Clear cache
        clear_result = client.clear_cache()
        print(f"   Cache cleared: {clear_result['status']}")
        
        # Run a query to populate cache
        result = client.query_documents("What is this document about?")
        print(f"   Query after cache clear: {result['confidence']}")
        
    except Exception as e:
        print(f"   Cache management failed: {e}")
    
    print("\n=== Demo completed ===")


if __name__ == "__main__":
    main()