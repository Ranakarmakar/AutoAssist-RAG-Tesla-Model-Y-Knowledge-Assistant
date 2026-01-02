#!/usr/bin/env python3
"""
Example client for the PDF RAG System API.

This script demonstrates how to interact with the API endpoints.
"""

import requests
import json
import time
from pathlib import Path


class PDFRAGClient:
    """Simple client for the PDF RAG System API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client with the API base URL."""
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self):
        """Check API health status."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Health check failed: {e}")
            return None
    
    def upload_document(self, pdf_path: str):
        """Upload a PDF document to the system."""
        try:
            with open(pdf_path, "rb") as f:
                files = {"file": (Path(pdf_path).name, f, "application/pdf")}
                response = self.session.post(f"{self.base_url}/upload", files=files)
                response.raise_for_status()
                return response.json()
        except requests.RequestException as e:
            print(f"Document upload failed: {e}")
            return None
        except FileNotFoundError:
            print(f"File not found: {pdf_path}")
            return None
    
    def query(self, question: str, include_citations: bool = True):
        """Query the system with a question."""
        try:
            data = {
                "query": question,
                "include_citations": include_citations,
                "check_sufficiency": True
            }
            response = self.session.post(f"{self.base_url}/query", json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Query failed: {e}")
            return None
    
    def query_with_document(self, question: str, pdf_path: str, include_citations: bool = True):
        """Query with a new document upload."""
        try:
            with open(pdf_path, "rb") as f:
                files = {"file": (Path(pdf_path).name, f, "application/pdf")}
                data = {
                    "query": question,
                    "include_citations": str(include_citations).lower()
                }
                response = self.session.post(
                    f"{self.base_url}/query-with-document",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                return response.json()
        except requests.RequestException as e:
            print(f"Query with document failed: {e}")
            return None
        except FileNotFoundError:
            print(f"File not found: {pdf_path}")
            return None
    
    def clear_documents(self):
        """Clear all documents from the system."""
        try:
            response = self.session.delete(f"{self.base_url}/documents")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Clear documents failed: {e}")
            return None
    
    def get_metrics(self):
        """Get system metrics."""
        try:
            response = self.session.get(f"{self.base_url}/metrics")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Get metrics failed: {e}")
            return None


def print_response(title: str, response: dict):
    """Pretty print API response."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    if response:
        print(json.dumps(response, indent=2))
    else:
        print("No response received")


def main():
    """Main example function."""
    print("PDF RAG System API Client Example")
    print("=" * 50)
    
    # Initialize client
    client = PDFRAGClient()
    
    # Check API health
    print("1. Checking API health...")
    health = client.health_check()
    if not health:
        print("API is not available. Make sure the server is running.")
        return
    
    print(f"✓ API is healthy - {health['status']}")
    print(f"  Total chunks: {health['total_chunks']}")
    print(f"  Memory usage: {health['memory_usage_mb']:.1f} MB")
    
    # Check if Tesla Owner's Manual exists
    pdf_path = "data/Owners_Manual.pdf"
    if not Path(pdf_path).exists():
        print(f"\nTesla Owner's Manual not found at {pdf_path}")
        print("Please place a PDF file there to test document upload.")
        return
    
    print(f"\n2. Testing query with document upload...")
    start_time = time.time()
    
    result = client.query_with_document(
        question="What are the main safety features of this vehicle?",
        pdf_path=pdf_path
    )
    
    if result:
        elapsed = time.time() - start_time
        print(f"✓ Query completed in {elapsed:.1f} seconds")
        print(f"  Answer length: {len(result['answer'])} characters")
        print(f"  Citations: {len(result['citations'])}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Sufficient information: {result['sufficient_information']}")
        
        print(f"\nAnswer preview:")
        print("-" * 30)
        print(result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer'])
        
        if result['citations']:
            print(f"\nCitations:")
            for i, citation in enumerate(result['citations'][:3], 1):
                print(f"  {i}. {citation.get('source', 'Unknown source')}")
    
    print(f"\n3. Testing regular query (using uploaded document)...")
    result = client.query("How do I charge the vehicle?")
    
    if result:
        print(f"✓ Query completed")
        print(f"  Answer length: {len(result['answer'])} characters")
        print(f"  Confidence: {result['confidence']}")
        
        print(f"\nAnswer preview:")
        print("-" * 30)
        print(result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer'])
    
    print(f"\n4. Getting system metrics...")
    metrics = client.get_metrics()
    if metrics:
        print(f"✓ Metrics retrieved")
        print(f"  System uptime: {metrics['system']['uptime_seconds']:.1f} seconds")
        print(f"  Total chunks: {metrics['workflow']['total_chunks']}")
        print(f"  Memory usage: {metrics['system']['memory_usage_mb']:.1f} MB")
        print(f"  Total requests: {metrics['system']['total_requests']}")
    
    print(f"\n5. Clearing documents...")
    result = client.clear_documents()
    if result:
        print(f"✓ Documents cleared: {result['message']}")
    
    print(f"\nExample completed successfully!")
    print(f"Visit http://localhost:8000/docs for interactive API documentation.")


if __name__ == "__main__":
    main()