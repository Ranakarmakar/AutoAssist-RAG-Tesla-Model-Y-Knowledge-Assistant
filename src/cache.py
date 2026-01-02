"""
Caching system for PDF RAG System.

This module provides caching capabilities for embeddings, query results,
and other expensive operations to improve performance.
"""

import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import threading

import numpy as np
from langchain_core.documents import Document

from src.config import settings
from src.logging_config import get_logger
from src.utils import ensure_directory_exists

logger = get_logger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass


class InMemoryCache(CacheBackend):
    """In-memory cache implementation with TTL support."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize in-memory cache.
        
        Args:
            max_size: Maximum number of entries to store
        """
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        logger.info("InMemoryCache initialized", max_size=max_size)
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        if entry.get("ttl") is None:
            return False
        return time.time() > entry["created_at"] + entry["ttl"]
    
    def _evict_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.get("ttl") is not None and current_time > entry["created_at"] + entry["ttl"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug("Evicted expired cache entries", count=len(expired_keys))
    
    def _evict_lru(self) -> None:
        """Remove least recently used entries if cache is full."""
        if len(self._cache) >= self.max_size:
            # Sort by last_accessed time and remove oldest
            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1]["last_accessed"]
            )
            
            # Remove oldest 10% of entries
            num_to_remove = max(1, len(sorted_items) // 10)
            for key, _ in sorted_items[:num_to_remove]:
                del self._cache[key]
            
            logger.debug("Evicted LRU cache entries", count=num_to_remove)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            self._evict_expired()
            
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            if self._is_expired(entry):
                del self._cache[key]
                return None
            
            # Update last accessed time
            entry["last_accessed"] = time.time()
            return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        with self._lock:
            self._evict_expired()
            self._evict_lru()
            
            current_time = time.time()
            self._cache[key] = {
                "value": value,
                "created_at": current_time,
                "last_accessed": current_time,
                "ttl": ttl
            }
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if self._is_expired(entry):
                del self._cache[key]
                return False
            
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            self._evict_expired()
            
            total_size = 0
            expired_count = 0
            
            for entry in self._cache.values():
                if self._is_expired(entry):
                    expired_count += 1
                else:
                    # Rough size estimation
                    try:
                        total_size += len(pickle.dumps(entry["value"]))
                    except Exception:
                        total_size += 1000  # Fallback estimate
            
            return {
                "total_entries": len(self._cache),
                "expired_entries": expired_count,
                "estimated_size_bytes": total_size,
                "max_size": self.max_size
            }


class FileCache(CacheBackend):
    """File-based cache implementation for persistent storage."""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Initialize file cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        ensure_directory_exists(str(self.cache_dir))
        
        # Create subdirectories for different cache types
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.queries_dir = self.cache_dir / "queries"
        self.metadata_dir = self.cache_dir / "metadata"
        
        for dir_path in [self.embeddings_dir, self.queries_dir, self.metadata_dir]:
            ensure_directory_exists(str(dir_path))
        
        logger.info("FileCache initialized", cache_dir=str(self.cache_dir))
    
    def _get_file_path(self, key: str, cache_type: str = "general") -> Path:
        """Get file path for cache key."""
        # Use hash of key as filename to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        
        if cache_type == "embeddings":
            return self.embeddings_dir / f"{key_hash}.pkl"
        elif cache_type == "queries":
            return self.queries_dir / f"{key_hash}.pkl"
        elif cache_type == "metadata":
            return self.metadata_dir / f"{key_hash}.json"
        else:
            return self.cache_dir / f"{key_hash}.pkl"
    
    def _load_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Load metadata for cache entry."""
        metadata_path = self._get_file_path(key, "metadata")
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load cache metadata", key=key, error=str(e))
            return None
    
    def _save_metadata(self, key: str, metadata: Dict[str, Any]) -> None:
        """Save metadata for cache entry."""
        metadata_path = self._get_file_path(key, "metadata")
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            logger.warning("Failed to save cache metadata", key=key, error=str(e))
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Check metadata first
        metadata = self._load_metadata(key)
        if not metadata:
            return None
        
        # Check TTL
        if metadata.get("ttl") is not None:
            if time.time() > metadata["created_at"] + metadata["ttl"]:
                self.delete(key)
                return None
        
        # Determine cache type and load data
        cache_type = metadata.get("cache_type", "general")
        file_path = self._get_file_path(key, cache_type)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning("Failed to load cache entry", key=key, error=str(e))
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, cache_type: str = "general") -> None:
        """Set value in cache with optional TTL."""
        try:
            # Save data
            file_path = self._get_file_path(key, cache_type)
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Save metadata
            metadata = {
                "created_at": time.time(),
                "ttl": ttl,
                "cache_type": cache_type,
                "key": key
            }
            self._save_metadata(key, metadata)
            
        except Exception as e:
            logger.error("Failed to save cache entry", key=key, error=str(e))
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        # Load metadata to determine cache type
        metadata = self._load_metadata(key)
        cache_type = metadata.get("cache_type", "general") if metadata else "general"
        
        # Delete data file
        file_path = self._get_file_path(key, cache_type)
        if file_path.exists():
            file_path.unlink()
        
        # Delete metadata file
        metadata_path = self._get_file_path(key, "metadata")
        if metadata_path.exists():
            metadata_path.unlink()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        import shutil
        
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                ensure_directory_exists(str(self.cache_dir))
                
                # Recreate subdirectories
                for dir_path in [self.embeddings_dir, self.queries_dir, self.metadata_dir]:
                    ensure_directory_exists(str(dir_path))
                
                logger.info("File cache cleared")
        except Exception as e:
            logger.error("Failed to clear file cache", error=str(e))
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        metadata = self._load_metadata(key)
        if not metadata:
            return False
        
        # Check TTL
        if metadata.get("ttl") is not None:
            if time.time() > metadata["created_at"] + metadata["ttl"]:
                self.delete(key)
                return False
        
        return True


class CacheManager:
    """Main cache manager that coordinates different cache backends."""
    
    def __init__(
        self,
        memory_cache_size: int = 1000,
        enable_file_cache: bool = True,
        cache_dir: str = "./data/cache"
    ):
        """
        Initialize cache manager.
        
        Args:
            memory_cache_size: Size of in-memory cache
            enable_file_cache: Whether to enable persistent file cache
            cache_dir: Directory for file cache
        """
        self.memory_cache = InMemoryCache(max_size=memory_cache_size)
        self.file_cache = FileCache(cache_dir) if enable_file_cache else None
        
        # Cache configuration
        self.embedding_ttl = 3600 * 24 * 7  # 1 week for embeddings
        self.query_ttl = 3600  # 1 hour for query results
        
        logger.info(
            "CacheManager initialized",
            memory_cache_size=memory_cache_size,
            enable_file_cache=enable_file_cache,
            cache_dir=cache_dir
        )
    
    def _generate_key(self, prefix: str, data: Union[str, Dict, List]) -> str:
        """Generate cache key from data."""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        
        key_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def get_embedding_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for embeddings."""
        return self._generate_key("embedding", f"{model_name}:{text}")
    
    def get_query_cache_key(self, query: str, context_hash: str) -> str:
        """Generate cache key for query results."""
        return self._generate_key("query", f"{query}:{context_hash}")
    
    def get_document_cache_key(self, document_path: str, chunk_config: Dict) -> str:
        """Generate cache key for document chunks."""
        return self._generate_key("document", f"{document_path}:{chunk_config}")
    
    def cache_embedding(self, text: str, model_name: str, embedding: np.ndarray) -> None:
        """Cache embedding vector."""
        key = self.get_embedding_cache_key(text, model_name)
        
        # Store in memory cache
        self.memory_cache.set(key, embedding, ttl=self.embedding_ttl)
        
        # Store in file cache for persistence
        if self.file_cache:
            self.file_cache.set(key, embedding, ttl=self.embedding_ttl, cache_type="embeddings")
        
        logger.debug("Cached embedding", text_length=len(text), model=model_name)
    
    def get_cached_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding vector."""
        key = self.get_embedding_cache_key(text, model_name)
        
        # Try memory cache first
        embedding = self.memory_cache.get(key)
        if embedding is not None:
            logger.debug("Embedding cache hit (memory)", text_length=len(text))
            return embedding
        
        # Try file cache
        if self.file_cache:
            embedding = self.file_cache.get(key)
            if embedding is not None:
                # Store back in memory cache
                self.memory_cache.set(key, embedding, ttl=self.embedding_ttl)
                logger.debug("Embedding cache hit (file)", text_length=len(text))
                return embedding
        
        logger.debug("Embedding cache miss", text_length=len(text))
        return None
    
    def cache_query_result(self, query: str, context_hash: str, result: Dict[str, Any]) -> None:
        """Cache query result."""
        key = self.get_query_cache_key(query, context_hash)
        
        # Store in memory cache
        self.memory_cache.set(key, result, ttl=self.query_ttl)
        
        # Store in file cache
        if self.file_cache:
            self.file_cache.set(key, result, ttl=self.query_ttl, cache_type="queries")
        
        logger.debug("Cached query result", query_length=len(query))
    
    def get_cached_query_result(self, query: str, context_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached query result."""
        key = self.get_query_cache_key(query, context_hash)
        
        # Try memory cache first
        result = self.memory_cache.get(key)
        if result is not None:
            logger.debug("Query cache hit (memory)", query_length=len(query))
            return result
        
        # Try file cache
        if self.file_cache:
            result = self.file_cache.get(key)
            if result is not None:
                # Store back in memory cache
                self.memory_cache.set(key, result, ttl=self.query_ttl)
                logger.debug("Query cache hit (file)", query_length=len(query))
                return result
        
        logger.debug("Query cache miss", query_length=len(query))
        return None
    
    def cache_document_chunks(self, document_path: str, chunk_config: Dict, chunks: List[Document]) -> None:
        """Cache document chunks."""
        key = self.get_document_cache_key(document_path, chunk_config)
        
        # Store in file cache (chunks can be large)
        if self.file_cache:
            self.file_cache.set(key, chunks, ttl=self.embedding_ttl, cache_type="general")
        
        logger.debug("Cached document chunks", path=document_path, chunk_count=len(chunks))
    
    def get_cached_document_chunks(self, document_path: str, chunk_config: Dict) -> Optional[List[Document]]:
        """Get cached document chunks."""
        key = self.get_document_cache_key(document_path, chunk_config)
        
        # Try file cache
        if self.file_cache:
            chunks = self.file_cache.get(key)
            if chunks is not None:
                logger.debug("Document chunks cache hit", path=document_path, chunk_count=len(chunks))
                return chunks
        
        logger.debug("Document chunks cache miss", path=document_path)
        return None
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        if self.file_cache:
            self.file_cache.clear()
        
        logger.info("All caches cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "memory_cache": self.memory_cache.get_stats(),
            "file_cache_enabled": self.file_cache is not None
        }
        
        if self.file_cache:
            # Count files in cache directories
            try:
                embeddings_count = len(list(self.file_cache.embeddings_dir.glob("*.pkl")))
                queries_count = len(list(self.file_cache.queries_dir.glob("*.pkl")))
                metadata_count = len(list(self.file_cache.metadata_dir.glob("*.json")))
                
                stats["file_cache"] = {
                    "embeddings_count": embeddings_count,
                    "queries_count": queries_count,
                    "metadata_count": metadata_count,
                    "cache_dir": str(self.file_cache.cache_dir)
                }
            except Exception as e:
                logger.warning("Failed to get file cache stats", error=str(e))
                stats["file_cache"] = {"error": str(e)}
        
        return stats


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager instance."""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager(
            memory_cache_size=getattr(settings, 'cache_memory_size', 1000),
            enable_file_cache=getattr(settings, 'enable_file_cache', True),
            cache_dir=getattr(settings, 'cache_dir', './data/cache')
        )
    
    return _cache_manager


def clear_all_caches() -> None:
    """Clear all caches."""
    cache_manager = get_cache_manager()
    cache_manager.clear_all_caches()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    cache_manager = get_cache_manager()
    return cache_manager.get_cache_stats()