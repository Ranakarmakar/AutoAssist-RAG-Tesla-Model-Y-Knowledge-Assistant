#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Demo Script for PDF RAG System

This script demonstrates basic usage of the PDF RAG System API:
1. Upload a PDF document
2. Ask questions about the document
3. Display answers with citations

Usage:
    python examples/simple_demo.py [pdf_file] [question]
    
Examples:
    python examples/simple_demo.py data/Owners_Manual.pdf "What are the safety features?"
    python examples/simple_demo.py manual.pdf "How do I charge this vehicle?"
"""

import sys
import requests
import json
from pathlib import Path
import argparse


def check_api_health(base_url="http://localhost:8000"):
    """Check if the API server is running."""
    try:
        response = requests.get(base_url + "/health", timeout=5)
        response.raise_for_status()
        health = response.json()
        print("✅ API server is healthy (status: {})".format(health['status']))
        return True
    except requests.exceptions.RequestException as e:
        print("❌ API server is not accessible: {}".format(e))
        print("   Make sure to start the server with: python run_api.py")
        return False


def upload_document(file_path, base_url="http://localhost:8000"):
    """Upload a PDF document to the system."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print("❌ File not found: {}".format(file_path))
        return False
    
    if not file_path.suffix.lower() == '.pdf':
        print("❌ Only PDF files are supported, got: {}".format(file_path.suffix))
        return False
    
    print("📄 Uploading document: {}".format(file_path.name))
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'application/pdf')}
            response = requests.post(base_url + "/upload", files=files, timeout=60)
            response.raise_for_status()
        
        result = response.json()
        print("✅ Upload successful!")
        print("   📊 Pages processed: {}".format(result['pages_processed']))
        print("   🧩 Chunks created: {}".format(result['chunks_created']))
        print("   ⏱️  Processing time: {:.1f}ms".format(result['processing_time_ms']))
        return True
        
    except requests.exceptions.RequestException as e:
        print("❌ Upload failed: {}".format(e))
        return False


def ask_question(question, base_url="http://localhost:8000"):
    """Ask a question about the uploaded documents."""
    print("\n🤔 Question: {}".format(question))
    
    try:
        payload = {
            "query": question,
            "include_citations": True,
            "check_sufficiency": True
        }
        
        response = requests.post(base_url + "/query", json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        print("\n💡 Answer:")
        print("   {}".format(result['answer']))
        
        print("\n📈 Confidence: {}".format(result['confidence']))
        print("🔍 Sufficient information: {}".format(result['sufficient_information']))
        print("📚 Sources used: {}".format(result['source_count']))
        print("⏱️  Processing time: {:.1f}ms".format(result['processing_time_ms']))
        
        if result['citations']:
            print("\n📖 Citations:")
            for i, citation in enumerate(result['citations'], 1):
                text_preview = citation.get('text', 'N/A')[:100]
                if len(text_preview) == 100:
                    text_preview += "..."
                print("   {}. {}".format(i, text_preview))
                if 'page' in citation:
                    print("      (Page {})".format(citation['page']))
        
        return True
        
    except requests.exceptions.RequestException as e:
        print("❌ Query failed: {}".format(e))
        return False


def interactive_mode(base_url="http://localhost:8000"):
    """Run in interactive mode for multiple questions."""
    print("\n🤖 Interactive Mode - Ask multiple questions!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            ask_question(question, base_url)
            print("\n" + "-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print("Error: {}".format(e))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Simple demo for PDF RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/simple_demo.py data/manual.pdf "What are the safety features?"
  python examples/simple_demo.py --interactive
  python examples/simple_demo.py manual.pdf --interactive
        """
    )
    
    parser.add_argument('pdf_file', nargs='?', help='PDF file to upload')
    parser.add_argument('question', nargs='?', help='Question to ask about the document')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Run in interactive mode for multiple questions')
    parser.add_argument('--url', default='http://localhost:8000',
                       help='API server URL (default: http://localhost:8000)')
    
    args = parser.parse_args()
    
    print("🚀 PDF RAG System - Simple Demo")
    print("=" * 40)
    
    # Check API health
    if not check_api_health(args.url):
        sys.exit(1)
    
    # Upload document if provided
    if args.pdf_file:
        if not upload_document(args.pdf_file, args.url):
            sys.exit(1)
    
    # Handle different modes
    if args.interactive:
        interactive_mode(args.url)
    elif args.question:
        if not args.pdf_file:
            print("⚠️  No document uploaded. Using existing documents in the system.")
        ask_question(args.question, args.url)
    elif args.pdf_file:
        # Document uploaded but no question - enter interactive mode
        print("\n📄 Document uploaded successfully!")
        print("💡 No question provided. Entering interactive mode...")
        interactive_mode(args.url)
    else:
        # No document or question - show help
        parser.print_help()
        print("\n💡 Tip: Start with uploading a document and asking a question!")


if __name__ == "__main__":
    main()