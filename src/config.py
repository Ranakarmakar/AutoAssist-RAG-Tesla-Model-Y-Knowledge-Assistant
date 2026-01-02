"""Configuration management for PDF RAG System."""

import os
from typing import Optional, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    groq_api_key: str = Field(..., description="Groq API key for LLM operations")
    
    # Model Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Embedding model for semantic search"
    )
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="LLM model for query rewriting and answer generation"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )
    
    # Chunking Configuration
    chunk_size: int = Field(default=1000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    
    # Retrieval Configuration
    retrieval_top_k: int = Field(default=20, description="Number of chunks to retrieve")
    rerank_top_k: int = Field(default=5, description="Number of chunks after reranking")
    bm25_weight: float = Field(default=0.3, description="Weight for BM25 search")
    semantic_weight: float = Field(default=0.7, description="Weight for semantic search")
    
    # Storage Configuration
    vector_store_path: str = Field(default="./data/vector_store", description="Path to vector store")
    upload_path: str = Field(default="./data/uploads", description="Path for uploaded files")
    
    # Cache Configuration
    enable_caching: bool = Field(default=True, description="Enable caching system")
    cache_memory_size: int = Field(default=1000, description="In-memory cache size")
    enable_file_cache: bool = Field(default=True, description="Enable persistent file cache")
    cache_dir: str = Field(default="./data/cache", description="Cache directory")
    embedding_cache_ttl: int = Field(default=3600 * 24 * 7, description="Embedding cache TTL (1 week)")
    query_cache_ttl: int = Field(default=3600, description="Query cache TTL (1 hour)")
    redis_url: Optional[str] = Field(default=None, description="Redis URL for caching")
    cache_ttl: int = Field(default=3600, description="Default cache TTL in seconds")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def validate_required_settings(settings: Settings) -> None:
    """Validate that all required settings are present."""
    if not settings.groq_api_key:
        raise ValueError("Required setting 'groq_api_key' is not configured")
    
    # Validate weights sum to 1.0
    total_weight = settings.bm25_weight + settings.semantic_weight
    if abs(total_weight - 1.0) > 0.01:
        raise ValueError(
            f"BM25 and semantic weights must sum to 1.0, got {total_weight}"
        )


# Global settings instance
settings = get_settings()