"""Utility functions for PDF RAG System."""

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.logging_config import get_logger

logger = get_logger(__name__)


def ensure_directory_exists(path: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)
    logger.info("Directory ensured", path=path)


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracking."""
    return str(uuid.uuid4())


def safe_filename(filename: str) -> str:
    """Generate a safe filename by removing/replacing problematic characters."""
    # Remove or replace problematic characters
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in ".-_":
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    
    return "".join(safe_chars)


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def validate_pdf_file(file_path: str) -> bool:
    """Validate that a file is a PDF."""
    if not os.path.exists(file_path):
        return False
    
    # Check file extension
    if not file_path.lower().endswith('.pdf'):
        return False
    
    # Check file size (limit to 50MB)
    if get_file_size_mb(file_path) > 50:
        logger.warning("PDF file too large", file_path=file_path, size_mb=get_file_size_mb(file_path))
        return False
    
    # Basic PDF header check
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                return False
    except Exception as e:
        logger.error("Error reading PDF header", file_path=file_path, error=str(e))
        return False
    
    return True


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to 0-1 range using min-max normalization."""
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if min_score == max_score:
        return [1.0] * len(scores)
    
    return [(score - min_score) / (max_score - min_score) for score in scores]


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"