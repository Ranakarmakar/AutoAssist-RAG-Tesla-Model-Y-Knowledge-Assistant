"""Unit tests for cache module."""

import pytest
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from langchain_core.documents import Document

from src.cache import (
    InMemoryCache, FileCache, CacheManager,
    get_cache_manager, clear_all_caches, get_cache_stats
)


class TestInMemoryCache:
    """Test InMemoryCache implementation."""
    
    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = InMemoryCache(max_size=10)
        
        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test non-existent key
        assert cache.get("nonexistent") is None
        
        # Test exists
        assert cache.exists("key1") is True
        assert cache.exists("nonexistent") is False
        
        # Test delete
        cache.delete("key1")
        assert cache.get("key1") is None
        assert cache.exists("key1") is False
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = InMemoryCache()
        
        # Set with short TTL
        cache.set("key1", "value1", ttl=1)
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None
        assert cache.exists("key1") is False
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = InMemoryCache(max_size=3)
        
        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new key to trigger eviction
        cache.set("key4", "value4")
        
        # key2 should be evicted (least recently used)
        assert cache.exists("key1") is True  # Recently accessed
        assert cache.exists("key3") is True  # Recently set
        assert cache.exists("key4") is True  # Just added
    
    def test_clear(self):
        """Test cache clearing."""
        cache = InMemoryCache()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_stats(self):
        """Test cache statistics."""
        cache = InMemoryCache(max_size=10)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2", ttl=1)
        
        stats = cache.get_stats()
        
        assert stats["max_size"] == 10
        assert stats["total_entries"] >= 2
        assert "estimated_size_bytes" in stats
    
    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        cache = InMemoryCache(max_size=100)
        results = []
        
        def worker(thread_id):
            for i in range(10):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                cache.set(key, value)
                retrieved = cache.get(key)
                results.append(retrieved == value)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All operations should succeed
        assert all(results)


class TestFileCache:
    """Test FileCache implementation."""
    
    def test_basic_operations(self):
        """Test basic file cache operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FileCache(cache_dir=temp_dir)
            
            # Test set and get
            cache.set("key1", "value1")
            assert cache.get("key1") == "value1"
            
            # Test non-existent key
            assert cache.get("nonexistent") is None
            
            # Test exists
            assert cache.exists("key1") is True
            assert cache.exists("nonexistent") is False
            
            # Test delete
            cache.delete("key1")
            assert cache.get("key1") is None
    
    def test_persistence(self):
        """Test cache persistence across instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first cache instance
            cache1 = FileCache(cache_dir=temp_dir)
            cache1.set("key1", "value1")
            
            # Create second cache instance with same directory
            cache2 = FileCache(cache_dir=temp_dir)
            assert cache2.get("key1") == "value1"
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration in file cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FileCache(cache_dir=temp_dir)
            
            # Set with short TTL
            cache.set("key1", "value1", ttl=1)
            assert cache.get("key1") == "value1"
            
            # Wait for expiration
            time.sleep(1.1)
            assert cache.get("key1") is None
    
    def test_cache_types(self):
        """Test different cache types (embeddings, queries, etc.)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FileCache(cache_dir=temp_dir)
            
            # Test different cache types
            cache.set("embed_key", np.array([1, 2, 3]), cache_type="embeddings")
            cache.set("query_key", {"answer": "test"}, cache_type="queries")
            
            # Verify retrieval
            embedding = cache.get("embed_key")
            assert isinstance(embedding, np.ndarray)
            assert np.array_equal(embedding, np.array([1, 2, 3]))
            
            query_result = cache.get("query_key")
            assert query_result == {"answer": "test"}
    
    def test_clear(self):
        """Test file cache clearing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FileCache(cache_dir=temp_dir)
            
            cache.set("key1", "value1")
            cache.set("key2", "value2")
            
            # Verify files exist
            assert len(list(Path(temp_dir).glob("**/*.pkl"))) > 0
            
            cache.clear()
            
            # Verify files are cleared
            assert cache.get("key1") is None
            assert cache.get("key2") is None
    
    def test_error_handling(self):
        """Test error handling in file cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FileCache(cache_dir=temp_dir)
            
            # Test with corrupted metadata
            metadata_file = cache.metadata_dir / "test.json"
            metadata_file.write_text("invalid json")
            
            # Should handle gracefully
            assert cache.get("test") is None


class TestCacheManager:
    """Test CacheManager coordination."""
    
    def test_initialization(self):
        """Test cache manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(
                memory_cache_size=100,
                enable_file_cache=True,
                cache_dir=temp_dir
            )
            
            assert manager.memory_cache is not None
            assert manager.file_cache is not None
    
    def test_embedding_caching(self):
        """Test embedding caching functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=temp_dir)
            
            text = "test text"
            model_name = "test-model"
            embedding = np.array([0.1, 0.2, 0.3])
            
            # Cache embedding
            manager.cache_embedding(text, model_name, embedding)
            
            # Retrieve embedding
            cached_embedding = manager.get_cached_embedding(text, model_name)
            assert cached_embedding is not None
            assert np.array_equal(cached_embedding, embedding)
            
            # Test cache miss
            assert manager.get_cached_embedding("other text", model_name) is None
    
    def test_query_result_caching(self):
        """Test query result caching functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=temp_dir)
            
            query = "test query"
            context_hash = "test_hash"
            result = {
                "answer": "test answer",
                "citations": [],
                "confidence": "high"
            }
            
            # Cache query result
            manager.cache_query_result(query, context_hash, result)
            
            # Retrieve query result
            cached_result = manager.get_cached_query_result(query, context_hash)
            assert cached_result is not None
            assert cached_result["answer"] == result["answer"]
            
            # Test cache miss
            assert manager.get_cached_query_result("other query", context_hash) is None
    
    def test_document_chunks_caching(self):
        """Test document chunks caching functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=temp_dir)
            
            document_path = "/path/to/doc.pdf"
            chunk_config = {"chunk_size": 1000, "overlap": 200}
            chunks = [
                Document(page_content="chunk 1", metadata={"page": 1}),
                Document(page_content="chunk 2", metadata={"page": 2})
            ]
            
            # Cache document chunks
            manager.cache_document_chunks(document_path, chunk_config, chunks)
            
            # Retrieve document chunks
            cached_chunks = manager.get_cached_document_chunks(document_path, chunk_config)
            assert cached_chunks is not None
            assert len(cached_chunks) == 2
            assert cached_chunks[0].page_content == "chunk 1"
            
            # Test cache miss
            other_config = {"chunk_size": 500, "overlap": 100}
            assert manager.get_cached_document_chunks(document_path, other_config) is None
    
    def test_key_generation(self):
        """Test cache key generation."""
        manager = CacheManager()
        
        # Test embedding key generation
        key1 = manager.get_embedding_cache_key("text", "model")
        key2 = manager.get_embedding_cache_key("text", "model")
        key3 = manager.get_embedding_cache_key("other text", "model")
        
        assert key1 == key2  # Same inputs should generate same key
        assert key1 != key3  # Different inputs should generate different keys
        
        # Test query key generation
        query_key1 = manager.get_query_cache_key("query", "hash")
        query_key2 = manager.get_query_cache_key("query", "hash")
        query_key3 = manager.get_query_cache_key("query", "other_hash")
        
        assert query_key1 == query_key2
        assert query_key1 != query_key3
    
    def test_memory_file_cache_coordination(self):
        """Test coordination between memory and file caches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=temp_dir)
            
            text = "test text"
            model_name = "test-model"
            embedding = np.array([0.1, 0.2, 0.3])
            
            # Cache embedding (should go to both memory and file cache)
            manager.cache_embedding(text, model_name, embedding)
            
            # Clear memory cache
            manager.memory_cache.clear()
            
            # Should still retrieve from file cache and restore to memory
            cached_embedding = manager.get_cached_embedding(text, model_name)
            assert cached_embedding is not None
            assert np.array_equal(cached_embedding, embedding)
            
            # Should now be in memory cache again
            memory_embedding = manager.memory_cache.get(
                manager.get_embedding_cache_key(text, model_name)
            )
            assert memory_embedding is not None
    
    def test_clear_all_caches(self):
        """Test clearing all caches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=temp_dir)
            
            # Add some data
            manager.cache_embedding("text", "model", np.array([1, 2, 3]))
            manager.cache_query_result("query", "hash", {"answer": "test"})
            
            # Clear all caches
            manager.clear_all_caches()
            
            # Verify everything is cleared
            assert manager.get_cached_embedding("text", "model") is None
            assert manager.get_cached_query_result("query", "hash") is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=temp_dir)
            
            # Add some data
            manager.cache_embedding("text", "model", np.array([1, 2, 3]))
            
            stats = manager.get_cache_stats()
            
            assert "memory_cache" in stats
            assert "file_cache_enabled" in stats
            assert stats["file_cache_enabled"] is True
            
            if "file_cache" in stats:
                assert "embeddings_count" in stats["file_cache"]
    
    def test_disabled_file_cache(self):
        """Test manager with disabled file cache."""
        manager = CacheManager(enable_file_cache=False)
        
        assert manager.file_cache is None
        
        # Should still work with memory cache only
        manager.cache_embedding("text", "model", np.array([1, 2, 3]))
        cached = manager.get_cached_embedding("text", "model")
        assert cached is not None


class TestGlobalCacheFunctions:
    """Test global cache functions."""
    
    @patch('src.cache._cache_manager', None)
    def test_get_cache_manager(self):
        """Test global cache manager getter."""
        # Should create new instance
        manager1 = get_cache_manager()
        assert manager1 is not None
        
        # Should return same instance
        manager2 = get_cache_manager()
        assert manager1 is manager2
    
    @patch('src.cache.get_cache_manager')
    def test_clear_all_caches_function(self, mock_get_manager):
        """Test global clear all caches function."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        clear_all_caches()
        
        mock_manager.clear_all_caches.assert_called_once()
    
    @patch('src.cache.get_cache_manager')
    def test_get_cache_stats_function(self, mock_get_manager):
        """Test global get cache stats function."""
        mock_manager = MagicMock()
        mock_stats = {"test": "stats"}
        mock_manager.get_cache_stats.return_value = mock_stats
        mock_get_manager.return_value = mock_manager
        
        stats = get_cache_stats()
        
        assert stats == mock_stats
        mock_manager.get_cache_stats.assert_called_once()


class TestCacheIntegration:
    """Test cache integration scenarios."""
    
    def test_concurrent_access(self):
        """Test concurrent cache access."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=temp_dir)
            results = []
            
            def worker(worker_id):
                for i in range(10):
                    text = f"worker_{worker_id}_text_{i}"
                    embedding = np.array([worker_id, i, 0.5])
                    
                    # Cache and retrieve
                    manager.cache_embedding(text, "model", embedding)
                    cached = manager.get_cached_embedding(text, "model")
                    
                    results.append(cached is not None and np.array_equal(cached, embedding))
            
            threads = []
            for i in range(3):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # All operations should succeed
            assert all(results)
    
    def test_large_data_handling(self):
        """Test handling of large data objects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=temp_dir)
            
            # Create large embedding
            large_embedding = np.random.rand(1000, 768)  # Typical BERT embedding size
            
            manager.cache_embedding("large_text", "model", large_embedding)
            cached = manager.get_cached_embedding("large_text", "model")
            
            assert cached is not None
            assert np.array_equal(cached, large_embedding)
    
    def test_cache_with_special_characters(self):
        """Test caching with special characters in keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=temp_dir)
            
            # Text with special characters
            special_text = "Text with special chars: àáâãäåæçèéêë 中文 русский 🚀"
            embedding = np.array([1, 2, 3])
            
            manager.cache_embedding(special_text, "model", embedding)
            cached = manager.get_cached_embedding(special_text, "model")
            
            assert cached is not None
            assert np.array_equal(cached, embedding)