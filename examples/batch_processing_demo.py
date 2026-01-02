#!/usr/bin/env python3
"""
Batch Processing Demo for PDF RAG System

This script demonstrates batch processing capabilities:
1. Upload multiple PDF documents
2. Process multiple queries against all documents
3. Generate a comprehensive report
4. Export results to JSON/CSV

Usage:
    python examples/batch_processing_demo.py --docs-dir /path/to/pdfs --queries-file queries.txt
    python examples/batch_processing_demo.py --docs manual1.pdf manual2.pdf --queries "What is warranty?" "How to maintain?"
"""

import sys
import requests
import json
import csv
import time
from pathlib import Path
import argparse
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class BatchProcessor:
    """Batch processing client for PDF RAG System."""
    
    def __init__(self, base_url="http://localhost:8000", max_workers=3):
        self.base_url = base_url.rstrip('/')
        self.max_workers = max_workers
        self.session = requests.Session()
        self.results = []
        self.lock = threading.Lock()
        
    def check_health(self):
        """Check API health."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            health = response.json()
            print(f"✅ API server healthy (uptime: {health['uptime_seconds']:.1f}s)")
            return True
        except Exception as e:
            print(f"❌ API server not accessible: {e}")
            return False
    
    def upload_document(self, file_path: Path) -> Dict[str, Any]:
        """Upload a single document."""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'application/pdf')}
                response = self.session.post(f"{self.base_url}/upload", files=files, timeout=120)
                response.raise_for_status()
            
            result = response.json()
            result['file_path'] = str(file_path)
            result['success'] = True
            
            print(f"✅ Uploaded: {file_path.name} ({result['chunks_created']} chunks)")
            return result
            
        except Exception as e:
            error_result = {
                'file_path': str(file_path),
                'success': False,
                'error': str(e),
                'filename': file_path.name
            }
            print(f"❌ Failed to upload {file_path.name}: {e}")
            return error_result
    
    def batch_upload(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Upload multiple documents concurrently."""
        print(f"\n📄 Uploading {len(file_paths)} documents...")
        
        upload_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.upload_document, fp): fp for fp in file_paths}
            
            for future in as_completed(future_to_file):
                result = future.result()
                upload_results.append(result)
        
        successful = sum(1 for r in upload_results if r['success'])
        print(f"📊 Upload summary: {successful}/{len(file_paths)} successful")
        
        return upload_results
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a single query."""
        try:
            payload = {
                "query": query,
                "include_citations": True,
                "check_sufficiency": True
            }
            
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/query", json=payload, timeout=60)
            response.raise_for_status()
            query_time = time.time() - start_time
            
            result = response.json()
            result['total_query_time'] = query_time
            result['success'] = True
            
            return result
            
        except Exception as e:
            return {
                'query': query,
                'success': False,
                'error': str(e),
                'answer': '',
                'confidence': 'error',
                'citations': [],
                'processing_time_ms': 0,
                'total_query_time': 0
            }
    
    def batch_query(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries concurrently."""
        print(f"\n🤔 Processing {len(queries)} queries...")
        
        query_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_query = {executor.submit(self.process_query, q): q for q in queries}
            
            for i, future in enumerate(as_completed(future_to_query), 1):
                result = future.result()
                query_results.append(result)
                
                status = "✅" if result['success'] else "❌"
                query_preview = result['query'][:50] + "..." if len(result['query']) > 50 else result['query']
                print(f"   {status} Query {i}/{len(queries)}: {query_preview}")
        
        successful = sum(1 for r in query_results if r['success'])
        print(f"📊 Query summary: {successful}/{len(queries)} successful")
        
        return query_results
    
    def generate_report(self, upload_results: List[Dict], query_results: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive batch processing report."""
        
        # Upload statistics
        successful_uploads = [r for r in upload_results if r['success']]
        failed_uploads = [r for r in upload_results if not r['success']]
        
        total_chunks = sum(r.get('chunks_created', 0) for r in successful_uploads)
        total_pages = sum(r.get('pages_processed', 0) for r in successful_uploads)
        avg_upload_time = sum(r.get('processing_time_ms', 0) for r in successful_uploads) / len(successful_uploads) if successful_uploads else 0
        
        # Query statistics
        successful_queries = [r for r in query_results if r['success']]
        failed_queries = [r for r in query_results if not r['success']]
        
        if successful_queries:
            avg_query_time = sum(r.get('processing_time_ms', 0) for r in successful_queries) / len(successful_queries)
            avg_total_time = sum(r.get('total_query_time', 0) for r in successful_queries) / len(successful_queries)
            
            confidence_dist = {}
            for r in successful_queries:
                conf = r.get('confidence', 'unknown')
                confidence_dist[conf] = confidence_dist.get(conf, 0) + 1
            
            citation_counts = [len(r.get('citations', [])) for r in successful_queries]
            avg_citations = sum(citation_counts) / len(citation_counts) if citation_counts else 0
        else:
            avg_query_time = avg_total_time = avg_citations = 0
            confidence_dist = {}
        
        report = {
            "summary": {
                "total_documents": len(upload_results),
                "successful_uploads": len(successful_uploads),
                "failed_uploads": len(failed_uploads),
                "total_queries": len(query_results),
                "successful_queries": len(successful_queries),
                "failed_queries": len(failed_queries),
                "total_chunks_created": total_chunks,
                "total_pages_processed": total_pages
            },
            "upload_performance": {
                "average_upload_time_ms": avg_upload_time,
                "successful_files": [r['filename'] for r in successful_uploads],
                "failed_files": [r['filename'] for r in failed_uploads]
            },
            "query_performance": {
                "average_processing_time_ms": avg_query_time,
                "average_total_time_seconds": avg_total_time,
                "average_citations_per_query": avg_citations,
                "confidence_distribution": confidence_dist
            },
            "detailed_results": {
                "uploads": upload_results,
                "queries": query_results
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_file: str):
        """Save report to JSON file."""
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Report saved to: {output_path}")
    
    def save_csv_results(self, query_results: List[Dict], output_file: str):
        """Save query results to CSV file."""
        output_path = Path(output_file)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Query', 'Answer', 'Confidence', 'Sufficient_Info', 
                'Source_Count', 'Citations_Count', 'Processing_Time_MS', 
                'Total_Time_Seconds', 'Success'
            ])
            
            # Data rows
            for result in query_results:
                writer.writerow([
                    result.get('query', ''),
                    result.get('answer', '').replace('\n', ' '),
                    result.get('confidence', ''),
                    result.get('sufficient_information', ''),
                    result.get('source_count', 0),
                    len(result.get('citations', [])),
                    result.get('processing_time_ms', 0),
                    result.get('total_query_time', 0),
                    result.get('success', False)
                ])
        
        print(f"📊 CSV results saved to: {output_path}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a summary of the batch processing results."""
        summary = report['summary']
        upload_perf = report['upload_performance']
        query_perf = report['query_performance']
        
        print("\n" + "=" * 60)
        print("📊 BATCH PROCESSING SUMMARY")
        print("=" * 60)
        
        print(f"\n📄 Document Processing:")
        print(f"   Total documents: {summary['total_documents']}")
        print(f"   Successful uploads: {summary['successful_uploads']}")
        print(f"   Failed uploads: {summary['failed_uploads']}")
        print(f"   Total pages processed: {summary['total_pages_processed']}")
        print(f"   Total chunks created: {summary['total_chunks_created']}")
        print(f"   Average upload time: {upload_perf['average_upload_time_ms']:.1f}ms")
        
        print(f"\n🤔 Query Processing:")
        print(f"   Total queries: {summary['total_queries']}")
        print(f"   Successful queries: {summary['successful_queries']}")
        print(f"   Failed queries: {summary['failed_queries']}")
        print(f"   Average processing time: {query_perf['average_processing_time_ms']:.1f}ms")
        print(f"   Average total time: {query_perf['average_total_time_seconds']:.2f}s")
        print(f"   Average citations per query: {query_perf['average_citations_per_query']:.1f}")
        
        if query_perf['confidence_distribution']:
            print(f"\n🎯 Confidence Distribution:")
            for conf, count in query_perf['confidence_distribution'].items():
                print(f"   {conf}: {count} queries")
        
        if upload_perf['failed_files']:
            print(f"\n❌ Failed uploads:")
            for filename in upload_perf['failed_files']:
                print(f"   - {filename}")
        
        print(f"\n⏰ Completed at: {report['timestamp']}")


def load_queries_from_file(file_path: str) -> List[str]:
    """Load queries from a text file (one per line)."""
    queries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                queries.append(line)
    return queries


def find_pdf_files(directory: str) -> List[Path]:
    """Find all PDF files in a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    pdf_files = list(dir_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {directory}")
    
    return pdf_files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Batch processing demo for PDF RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process directory of PDFs with queries from file
  python examples/batch_processing_demo.py --docs-dir ./pdfs --queries-file queries.txt
  
  # Process specific PDFs with inline queries
  python examples/batch_processing_demo.py --docs manual1.pdf manual2.pdf --queries "What is warranty?" "How to maintain?"
  
  # Process with custom output
  python examples/batch_processing_demo.py --docs-dir ./pdfs --queries-file queries.txt --output results.json --csv results.csv
        """
    )
    
    # Document sources
    doc_group = parser.add_mutually_exclusive_group(required=True)
    doc_group.add_argument('--docs-dir', help='Directory containing PDF files')
    doc_group.add_argument('--docs', nargs='+', help='Specific PDF files to process')
    
    # Query sources
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument('--queries-file', help='File containing queries (one per line)')
    query_group.add_argument('--queries', nargs='+', help='Inline queries to process')
    
    # Options
    parser.add_argument('--url', default='http://localhost:8000', help='API server URL')
    parser.add_argument('--workers', type=int, default=3, help='Number of concurrent workers')
    parser.add_argument('--output', default='batch_results.json', help='Output JSON file')
    parser.add_argument('--csv', help='Output CSV file for query results')
    parser.add_argument('--clear-docs', action='store_true', help='Clear existing documents before upload')
    
    args = parser.parse_args()
    
    print("🚀 PDF RAG System - Batch Processing Demo")
    print("=" * 50)
    
    # Initialize processor
    processor = BatchProcessor(args.url, args.workers)
    
    # Check API health
    if not processor.check_health():
        sys.exit(1)
    
    # Clear existing documents if requested
    if args.clear_docs:
        try:
            response = requests.delete(f"{args.url}/documents", timeout=30)
            response.raise_for_status()
            print("🗑️  Cleared existing documents")
        except Exception as e:
            print(f"⚠️  Failed to clear documents: {e}")
    
    # Prepare document list
    try:
        if args.docs_dir:
            pdf_files = find_pdf_files(args.docs_dir)
            print(f"📁 Found {len(pdf_files)} PDF files in {args.docs_dir}")
        else:
            pdf_files = [Path(f) for f in args.docs]
            for f in pdf_files:
                if not f.exists():
                    raise FileNotFoundError(f"File not found: {f}")
                if not f.suffix.lower() == '.pdf':
                    raise ValueError(f"Not a PDF file: {f}")
            print(f"📄 Processing {len(pdf_files)} specified PDF files")
    except Exception as e:
        print(f"❌ Error preparing document list: {e}")
        sys.exit(1)
    
    # Prepare query list
    try:
        if args.queries_file:
            queries = load_queries_from_file(args.queries_file)
            print(f"📝 Loaded {len(queries)} queries from {args.queries_file}")
        else:
            queries = args.queries
            print(f"📝 Processing {len(queries)} inline queries")
    except Exception as e:
        print(f"❌ Error preparing query list: {e}")
        sys.exit(1)
    
    # Process documents
    start_time = time.time()
    upload_results = processor.batch_upload(pdf_files)
    
    # Process queries
    query_results = processor.batch_query(queries)
    
    total_time = time.time() - start_time
    
    # Generate and save report
    report = processor.generate_report(upload_results, query_results)
    report['total_processing_time_seconds'] = total_time
    
    processor.save_report(report, args.output)
    
    if args.csv:
        processor.save_csv_results(query_results, args.csv)
    
    # Print summary
    processor.print_summary(report)
    
    print(f"\n⏱️  Total processing time: {total_time:.2f} seconds")
    print("✅ Batch processing completed!")


if __name__ == "__main__":
    main()