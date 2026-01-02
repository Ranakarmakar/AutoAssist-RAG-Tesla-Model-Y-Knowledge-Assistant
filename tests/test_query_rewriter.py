"""Tests for query rewriter."""

import pytest
from unittest.mock import patch, MagicMock

from src.query_rewriter import QueryRewriter
from src.exceptions import QueryRewritingError


class TestQueryRewriter:
    """Test cases for QueryRewriter class."""
    
    def setup_method(self):
        """Set up test fixtures with mocked Groq LLM."""
        with patch('src.query_rewriter.ChatGroq') as mock_groq_class:
            # Mock the Groq LLM
            self.mock_llm = MagicMock()
            mock_groq_class.return_value = self.mock_llm
            
            # Create query rewriter
            self.query_rewriter = QueryRewriter(
                model_name="test-model",
                temperature=0.1
            )
    
    def test_initialization_default_settings(self):
        """Test query rewriter initialization with default settings."""
        with patch('src.query_rewriter.settings') as mock_settings:
            mock_settings.llm_model = "default-model"
            mock_settings.groq_api_key = "test-key"
            
            with patch('src.query_rewriter.ChatGroq') as mock_groq_class:
                mock_llm = MagicMock()
                mock_groq_class.return_value = mock_llm
                
                rewriter = QueryRewriter()
                
                assert rewriter.model_name == "default-model"
                assert rewriter.temperature == 0.1
                assert rewriter.max_tokens is None
    
    def test_initialization_custom_settings(self):
        """Test query rewriter initialization with custom settings."""
        with patch('src.query_rewriter.ChatGroq') as mock_groq_class:
            mock_llm = MagicMock()
            mock_groq_class.return_value = mock_llm
            
            rewriter = QueryRewriter(
                model_name="custom-model",
                temperature=0.5,
                max_tokens=1000
            )
            
            assert rewriter.model_name == "custom-model"
            assert rewriter.temperature == 0.5
            assert rewriter.max_tokens == 1000
    
    def test_enhance_query_success(self):
        """Test successful query enhancement."""
        original_query = "How to charge car?"
        enhanced_response = "How to charge an electric vehicle battery?"
        
        # Mock the chain response
        self.query_rewriter.enhancement_chain = MagicMock()
        self.query_rewriter.enhancement_chain.invoke.return_value = enhanced_response
        
        result = self.query_rewriter.enhance_query(original_query)
        
        assert result == enhanced_response
        self.query_rewriter.enhancement_chain.invoke.assert_called_once_with({
            "original_query": original_query
        })
    
    def test_enhance_query_empty_input(self):
        """Test query enhancement with empty input."""
        with pytest.raises(QueryRewritingError, match="Query cannot be empty"):
            self.query_rewriter.enhance_query("")
        
        with pytest.raises(QueryRewritingError, match="Query cannot be empty"):
            self.query_rewriter.enhance_query("   ")
    
    def test_enhance_query_empty_response(self):
        """Test query enhancement with empty LLM response."""
        original_query = "test query"
        
        # Mock empty response
        self.query_rewriter.enhancement_chain = MagicMock()
        self.query_rewriter.enhancement_chain.invoke.return_value = ""
        
        result = self.query_rewriter.enhance_query(original_query)
        
        # Should return original query when enhancement is empty
        assert result == original_query
    
    def test_expand_query_success(self):
        """Test successful query expansion."""
        original_query = "vehicle charging"
        expansion_response = """1. How to charge an electric vehicle
2. Electric car charging methods
3. Vehicle battery charging process"""
        
        # Mock the chain response
        self.query_rewriter.expansion_chain = MagicMock()
        self.query_rewriter.expansion_chain.invoke.return_value = expansion_response
        
        result = self.query_rewriter.expand_query(original_query)
        
        expected_variations = [
            "How to charge an electric vehicle",
            "Electric car charging methods", 
            "Vehicle battery charging process"
        ]
        
        assert result == expected_variations
        self.query_rewriter.expansion_chain.invoke.assert_called_once_with({
            "original_query": original_query
        })
    
    def test_expand_query_empty_response(self):
        """Test query expansion with empty LLM response."""
        original_query = "test query"
        
        # Mock empty response
        self.query_rewriter.expansion_chain = MagicMock()
        self.query_rewriter.expansion_chain.invoke.return_value = ""
        
        result = self.query_rewriter.expand_query(original_query)
        
        # Should return original query when expansion fails
        assert result == [original_query]
    
    def test_clarify_query_success(self):
        """Test successful query clarification."""
        original_query = "charging issues"
        context = "Tesla Model Y electric vehicle"
        clarified_response = "Tesla Model Y electric vehicle charging problems and troubleshooting"
        
        # Mock the chain response
        self.query_rewriter.clarification_chain = MagicMock()
        self.query_rewriter.clarification_chain.invoke.return_value = clarified_response
        
        result = self.query_rewriter.clarify_query(original_query, context)
        
        assert result == clarified_response
        self.query_rewriter.clarification_chain.invoke.assert_called_once_with({
            "original_query": original_query,
            "context": context
        })
    
    def test_clarify_query_no_context(self):
        """Test query clarification without context."""
        original_query = "charging issues"
        clarified_response = "Electric vehicle charging problems"
        
        # Mock the chain response
        self.query_rewriter.clarification_chain = MagicMock()
        self.query_rewriter.clarification_chain.invoke.return_value = clarified_response
        
        result = self.query_rewriter.clarify_query(original_query, None)
        
        assert result == clarified_response
        self.query_rewriter.clarification_chain.invoke.assert_called_once_with({
            "original_query": original_query,
            "context": ""
        })
    
    def test_decompose_query_success(self):
        """Test successful query decomposition."""
        complex_query = "How to charge the vehicle and what are the safety precautions?"
        decomposition_response = """1. How to charge an electric vehicle?
2. What are the safety precautions for vehicle charging?"""
        
        # Mock the chain response
        self.query_rewriter.decomposition_chain = MagicMock()
        self.query_rewriter.decomposition_chain.invoke.return_value = decomposition_response
        
        result = self.query_rewriter.decompose_query(complex_query)
        
        expected_sub_questions = [
            "How to charge an electric vehicle?",
            "What are the safety precautions for vehicle charging?"
        ]
        
        assert result == expected_sub_questions
        self.query_rewriter.decomposition_chain.invoke.assert_called_once_with({
            "complex_query": complex_query
        })
    
    def test_parse_numbered_list_success(self):
        """Test parsing numbered list from LLM response."""
        text = """1. First item here
2. Second item here
3. Third item here"""
        
        result = self.query_rewriter._parse_numbered_list(text)
        
        expected = [
            "First item here",
            "Second item here", 
            "Third item here"
        ]
        
        assert result == expected
    
    def test_parse_numbered_list_with_dashes(self):
        """Test parsing list with dashes."""
        text = """- First item
- Second item
- Third item"""
        
        result = self.query_rewriter._parse_numbered_list(text)
        
        expected = [
            "First item",
            "Second item",
            "Third item"
        ]
        
        assert result == expected
    
    def test_parse_numbered_list_mixed_format(self):
        """Test parsing list with mixed formatting."""
        text = """1. First item
2. Second item

3. Third item with extra spaces"""
        
        result = self.query_rewriter._parse_numbered_list(text)
        
        expected = [
            "First item",
            "Second item",
            "Third item with extra spaces"
        ]
        
        assert result == expected
    
    def test_parse_numbered_list_empty(self):
        """Test parsing empty or invalid list."""
        assert self.query_rewriter._parse_numbered_list("") == []
        assert self.query_rewriter._parse_numbered_list("No numbered items here") == []
    
    def test_is_complex_query_simple(self):
        """Test complexity detection for simple queries."""
        simple_queries = [
            "vehicle charging",
            "How to charge?",
            "battery information"
        ]
        
        for query in simple_queries:
            assert not self.query_rewriter._is_complex_query(query)
    
    def test_is_complex_query_complex(self):
        """Test complexity detection for complex queries."""
        complex_queries = [
            "How to charge the vehicle and what are safety precautions?",
            "What is the difference between AC and DC charging?",
            "Explain how to charge the vehicle or use the mobile connector",
            "This is a very long query with many words that should be considered complex due to its length"
        ]
        
        for query in complex_queries:
            assert self.query_rewriter._is_complex_query(query)
    
    def test_rewrite_query_comprehensive_full(self):
        """Test comprehensive query rewriting with all features."""
        original_query = "How to charge vehicle and safety?"
        
        # Mock all chains
        self.query_rewriter.enhancement_chain = MagicMock()
        self.query_rewriter.enhancement_chain.invoke.return_value = "Enhanced query"
        
        self.query_rewriter.clarification_chain = MagicMock()
        self.query_rewriter.clarification_chain.invoke.return_value = "Clarified query"
        
        self.query_rewriter.expansion_chain = MagicMock()
        self.query_rewriter.expansion_chain.invoke.return_value = "1. Variation 1\n2. Variation 2"
        
        self.query_rewriter.decomposition_chain = MagicMock()
        self.query_rewriter.decomposition_chain.invoke.return_value = "1. Sub-question 1\n2. Sub-question 2"
        
        result = self.query_rewriter.rewrite_query_comprehensive(
            original_query,
            include_variations=True,
            include_enhancement=True,
            context="Tesla vehicle"
        )
        
        assert result["original_query"] == original_query
        assert result["enhanced_query"] == "Enhanced query"
        assert result["clarified_query"] == "Clarified query"
        assert result["query_variations"] == ["Variation 1", "Variation 2"]
        assert result["sub_questions"] == ["Sub-question 1", "Sub-question 2"]
    
    def test_rewrite_query_comprehensive_minimal(self):
        """Test comprehensive query rewriting with minimal features."""
        original_query = "simple query"
        
        result = self.query_rewriter.rewrite_query_comprehensive(
            original_query,
            include_variations=False,
            include_enhancement=False,
            context=None
        )
        
        assert result["original_query"] == original_query
        assert result["enhanced_query"] is None
        assert result["clarified_query"] is None
        assert result["query_variations"] == []
        assert result["sub_questions"] == []  # Simple query, no decomposition
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.query_rewriter.get_model_info()
        
        assert info["model_name"] == "test-model"
        assert info["temperature"] == 0.1
        assert info["provider"] == "Groq"
        assert "query_enhancement" in info["capabilities"]
        assert "query_expansion" in info["capabilities"]
        assert "query_clarification" in info["capabilities"]
        assert "query_decomposition" in info["capabilities"]
    
    @patch('src.query_rewriter.ChatGroq')
    def test_initialization_failure(self, mock_groq_class):
        """Test query rewriter initialization failure."""
        mock_groq_class.side_effect = Exception("Groq initialization failed")
        
        with pytest.raises(QueryRewritingError, match="Failed to initialize query rewriter"):
            QueryRewriter()
    
    def test_enhance_query_failure(self):
        """Test query enhancement failure."""
        self.query_rewriter.enhancement_chain = MagicMock()
        self.query_rewriter.enhancement_chain.invoke.side_effect = Exception("LLM error")
        
        with pytest.raises(QueryRewritingError, match="Failed to enhance query"):
            self.query_rewriter.enhance_query("test query")
    
    def test_expand_query_failure(self):
        """Test query expansion failure."""
        self.query_rewriter.expansion_chain = MagicMock()
        self.query_rewriter.expansion_chain.invoke.side_effect = Exception("LLM error")
        
        with pytest.raises(QueryRewritingError, match="Failed to expand query"):
            self.query_rewriter.expand_query("test query")
    
    def test_clarify_query_failure(self):
        """Test query clarification failure."""
        self.query_rewriter.clarification_chain = MagicMock()
        self.query_rewriter.clarification_chain.invoke.side_effect = Exception("LLM error")
        
        with pytest.raises(QueryRewritingError, match="Failed to clarify query"):
            self.query_rewriter.clarify_query("test query", "context")
    
    def test_decompose_query_failure(self):
        """Test query decomposition failure."""
        self.query_rewriter.decomposition_chain = MagicMock()
        self.query_rewriter.decomposition_chain.invoke.side_effect = Exception("LLM error")
        
        with pytest.raises(QueryRewritingError, match="Failed to decompose query"):
            self.query_rewriter.decompose_query("test query")