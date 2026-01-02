#!/usr/bin/env python3
"""
Simple script to run the PDF RAG System API server.

Usage:
    python run_api.py [--host HOST] [--port PORT] [--reload]

Examples:
    python run_api.py                    # Run on localhost:8000
    python run_api.py --port 8080        # Run on localhost:8080
    python run_api.py --host 0.0.0.0     # Run on all interfaces
    python run_api.py --reload           # Run with auto-reload for development
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api import run_server


def main():
    """Main entry point for API server."""
    parser = argparse.ArgumentParser(
        description="Run the PDF RAG System API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting PDF RAG System API server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"Log level: {args.log_level}")
    print()
    print(f"API will be available at: http://{args.host}:{args.port}")
    print(f"API documentation: http://{args.host}:{args.port}/docs")
    print(f"Health check: http://{args.host}:{args.port}/health")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        run_server(
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()