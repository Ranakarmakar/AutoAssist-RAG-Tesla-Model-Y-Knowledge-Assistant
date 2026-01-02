"""Tests for configuration management."""

import pytest
import os
from unittest.mock import patch
from pydantic import ValidationError
from src.config import Settings, validate_required_settings
from src.exceptions import ConfigurationError


def test_settings_default_values():
    """Test that settings have appropriate default values."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
        settings = Settings()
        
        assert settings.groq_api_key == "test_key"
        assert settings.embedding_model == "sentence-transformers/all-mpnet-base-v2"
        assert settings.llm_model == "llama-3.3-70b-versatile"
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.retrieval_top_k == 20
        assert settings.bm25_weight == 0.3
        assert settings.semantic_weight == 0.7


def test_validate_required_settings_success():
    """Test successful validation of required settings."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
        settings = Settings()
        # Should not raise an exception
        validate_required_settings(settings)


def test_validate_required_settings_missing_api_key():
    """Test validation failure when API key is missing."""
    with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=True):
        settings = Settings()
        with pytest.raises(ValueError, match="Required setting 'groq_api_key'"):
            validate_required_settings(settings)


def test_validate_weights_sum():
    """Test that BM25 and semantic weights sum to 1.0."""
    with patch.dict(os.environ, {
        "GROQ_API_KEY": "test_key",
        "BM25_WEIGHT": "0.4",
        "SEMANTIC_WEIGHT": "0.5"
    }):
        settings = Settings()
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            validate_required_settings(settings)