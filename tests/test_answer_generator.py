"""Tests for answer generator component."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.answer_generator import AnswerGenerator
from src.exceptions import RetrievalError


class TestAnswerGenerator:
    """Test cases for AnswerGenerator class."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="Machine learning is a method of data analysis that automates analytical model building.",
                metadata={"source": "ai_guide.pdf", "page": 1, "chunk_id": "ml_1", "rerank_score": 0.9}
            ),
            Document(
                page_content="Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
                metadata={"source": "ai_guide.pdf", "page": 2, "chunk_id": "dl_1", "rerank_score": 0.8}
            ),
            Document(
                page_content="Natural language processing (NLP) is a subfield of linguistics and artificial intelligence.",
                metadata={"source": "nlp_guide.pdf", "page": 1, "chunk_id": "nlp_1", "rerank_score": 0.7}
            )
        ]
    
    @patch('src.answer_generator.ChatGroq')
    def test_initialization_default_settings(self, mock_chat_groq):
        """Test answer generator initialization with default settings."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        assert generator.model_name == "llama-3.3-70b-versatile"  # From settings
        assert generator.temperature == 0.1
        assert generator.max_tokens is None
        assert generator.llm == mock_llm_instance
        
        mock_chat_groq.assert_called_once()
    
    @patch('src.answer_generator.ChatGroq')
    def test_initialization_custom_settings(self, mock_chat_groq):
        """Test answer generator initialization with custom settings."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator(
            model_name="custom-model",
            temperature=0.5,
            max_tokens=1000
        )
        
        assert generator.model_name == "custom-model"
        assert generator.temperature == 0.5
        assert generator.max_tokens == 1000
    
    @patch('src.answer_generator.ChatGroq')
    def test_initialization_failure(self, mock_chat_groq):
        """Test answer generator initialization failure."""
        mock_chat_groq.side_effect = Exception("LLM initialization failed")
        
        with pytest.raises(RetrievalError, match="Failed to initialize answer generator"):
            AnswerGenerator()
    
    @patch('src.answer_generator.ChatGroq')
    def test_generate_answer_success(self, mock_chat_groq, sample_documents):
        """Test successful answer generation."""
        # Setup mock LLM
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        # Mock chain responses
        mock_answer_response = "Machine learning is a data analysis method that builds analytical models automatically. [Source: ai_guide.pdf, Page: 1]"
        mock_sufficiency_response = "SUFFICIENT\nThe context provides adequate information about machine learning."
        
        generator = AnswerGenerator()
        
        # Mock the chain invocations
        generator.answer_chain = Mock()
        generator.answer_chain.invoke.return_value = mock_answer_response
        
        generator.sufficiency_chain = Mock()
        generator.sufficiency_chain.invoke.return_value = mock_sufficiency_response
        
        # Generate answer
        result = generator.generate_answer("What is machine learning?", sample_documents)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "answer" in result
        assert "citations" in result
        assert "sufficient_information" in result
        assert "source_count" in result
        assert "confidence" in result
        
        # Verify content
        assert result["answer"] == mock_answer_response
        assert result["source_count"] == 3
        assert result["sufficient_information"] is True
        assert len(result["citations"]) == 3
        
        # Verify citations
        for i, citation in enumerate(result["citations"]):
            assert "source" in citation
            assert "page" in citation
            assert "chunk_id" in citation
            assert "relevance_score" in citation
    
    @patch('src.answer_generator.ChatGroq')
    def test_generate_answer_empty_question(self, mock_chat_groq):
        """Test answer generation with empty question."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        with pytest.raises(RetrievalError, match="Question cannot be empty"):
            generator.generate_answer("", [])
        
        with pytest.raises(RetrievalError, match="Question cannot be empty"):
            generator.generate_answer("   ", [])
    
    @patch('src.answer_generator.ChatGroq')
    def test_generate_answer_no_documents(self, mock_chat_groq):
        """Test answer generation with no documents."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        result = generator.generate_answer("What is machine learning?", [])
        
        assert "I don't have any relevant documents" in result["answer"]
        assert result["citations"] == []
        assert result["sufficient_information"] is False
        assert result["source_count"] == 0
        assert result["confidence"] == "low"
    
    @patch('src.answer_generator.ChatGroq')
    def test_generate_answer_insufficient_info(self, mock_chat_groq, sample_documents):
        """Test answer generation with insufficient information."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        # Mock insufficient information response
        mock_answer_response = "I don't have sufficient information to answer this question."
        mock_sufficiency_response = "INSUFFICIENT\nThe context lacks specific details needed to answer the question."
        
        generator.answer_chain = Mock()
        generator.answer_chain.invoke.return_value = mock_answer_response
        
        generator.sufficiency_chain = Mock()
        generator.sufficiency_chain.invoke.return_value = mock_sufficiency_response
        
        result = generator.generate_answer("What is quantum computing?", sample_documents)
        
        assert result["sufficient_information"] is False
        assert result["confidence"] == "low"
        assert "INSUFFICIENT" in result["sufficiency_explanation"]
    
    @patch('src.answer_generator.ChatGroq')
    def test_prepare_context(self, mock_chat_groq, sample_documents):
        """Test context preparation from documents."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        context = generator._prepare_context(sample_documents)
        
        # Verify context contains document information
        assert "Document 1:" in context
        assert "Document 2:" in context
        assert "Document 3:" in context
        assert "ai_guide.pdf" in context
        assert "nlp_guide.pdf" in context
        assert "Machine learning is a method" in context
        assert "Deep learning is part" in context
        assert "Natural language processing" in context
    
    @patch('src.answer_generator.ChatGroq')
    def test_extract_citations(self, mock_chat_groq, sample_documents):
        """Test citation extraction from documents."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        citations = generator._extract_citations(sample_documents)
        
        assert len(citations) == 3
        
        for i, citation in enumerate(citations):
            assert citation["id"] == i + 1
            assert "source" in citation
            assert "page" in citation
            assert "chunk_id" in citation
            assert "content_preview" in citation
            assert "relevance_score" in citation
        
        # Check specific citation content
        assert citations[0]["source"] == "ai_guide.pdf"
        assert citations[0]["page"] == 1
        assert citations[0]["chunk_id"] == "ml_1"
        assert citations[0]["relevance_score"] == 0.9
    
    @patch('src.answer_generator.ChatGroq')
    def test_assess_confidence(self, mock_chat_groq, sample_documents):
        """Test confidence assessment."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        # Test high confidence - needs 3+ docs and 200+ chars
        high_conf = generator._assess_confidence(
            "This is a comprehensive answer with detailed information that provides thorough coverage of the topic with extensive explanations and multiple supporting points that demonstrate deep understanding and expertise.",
            sample_documents,
            True
        )
        assert high_conf == "high"
        
        # Test medium confidence - 2+ docs and 100+ chars
        medium_conf = generator._assess_confidence(
            "This is a moderately detailed answer that provides good coverage of the main points with sufficient explanation.",
            sample_documents[:2],
            True
        )
        assert medium_conf == "medium"
        
        # Test low confidence - insufficient info
        low_conf = generator._assess_confidence(
            "Any answer",
            sample_documents,
            False
        )
        assert low_conf == "low"
        
        # Test low confidence - insufficient answer
        low_conf2 = generator._assess_confidence(
            "I don't have sufficient information to answer this.",
            sample_documents,
            True
        )
        assert low_conf2 == "low"
    
    @patch('src.answer_generator.ChatGroq')
    def test_generate_answer_with_retriever(self, mock_chat_groq, sample_documents):
        """Test answer generation with retriever."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = sample_documents
        
        # Mock answer generation
        generator.generate_answer = Mock()
        generator.generate_answer.return_value = {
            "answer": "Test answer",
            "citations": [],
            "sufficient_information": True,
            "source_count": 3,
            "confidence": "high"
        }
        
        result = generator.generate_answer_with_retriever("What is AI?", mock_retriever)
        
        # Verify retriever was called
        mock_retriever.invoke.assert_called_once_with("What is AI?")
        
        # Verify answer generation was called
        generator.generate_answer.assert_called_once_with("What is AI?", sample_documents)
        
        assert result["answer"] == "Test answer"
    
    @patch('src.answer_generator.ChatGroq')
    def test_generate_answer_with_retriever_legacy(self, mock_chat_groq, sample_documents):
        """Test answer generation with legacy retriever (no invoke method)."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        # Mock legacy retriever (no invoke method)
        mock_retriever = Mock()
        del mock_retriever.invoke  # Remove invoke method
        mock_retriever.get_relevant_documents.return_value = sample_documents
        
        # Mock answer generation
        generator.generate_answer = Mock()
        generator.generate_answer.return_value = {"answer": "Test answer"}
        
        result = generator.generate_answer_with_retriever("What is AI?", mock_retriever)
        
        # Verify legacy method was called
        mock_retriever.get_relevant_documents.assert_called_once_with("What is AI?")
        
        assert result["answer"] == "Test answer"
    
    @patch('src.answer_generator.ChatGroq')
    def test_format_answer_with_citations(self, mock_chat_groq):
        """Test answer formatting with citations."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        result = {
            "answer": "Machine learning is a data analysis method.",
            "citations": [
                {
                    "source": "ai_guide.pdf",
                    "page": 1,
                    "relevance_score": 0.9
                },
                {
                    "source": "ml_book.pdf",
                    "page": 5,
                    "relevance_score": 0.8
                }
            ],
            "sufficient_information": True,
            "sufficiency_explanation": "",
            "confidence": "high"
        }
        
        formatted = generator.format_answer_with_citations(result)
        
        assert "Machine learning is a data analysis method." in formatted
        assert "**Sources:**" in formatted
        assert "ai_guide.pdf, Page 1" in formatted
        assert "ml_book.pdf, Page 5" in formatted
        assert "Relevance: 0.900" in formatted
        assert "**Confidence:** High" in formatted
    
    @patch('src.answer_generator.ChatGroq')
    def test_format_answer_insufficient_info(self, mock_chat_groq):
        """Test answer formatting with insufficient information."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        result = {
            "answer": "I cannot answer this question.",
            "citations": [],
            "sufficient_information": False,
            "sufficiency_explanation": "The context lacks necessary information.",
            "confidence": "low"
        }
        
        formatted = generator.format_answer_with_citations(result)
        
        assert "I cannot answer this question." in formatted
        assert "**Note:** The context lacks necessary information." in formatted
        assert "**Confidence:** Low" in formatted
    
    @patch('src.answer_generator.ChatGroq')
    def test_batch_generate_answers(self, mock_chat_groq, sample_documents):
        """Test batch answer generation."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        # Mock individual answer generation
        generator.generate_answer = Mock()
        generator.generate_answer.side_effect = [
            {"answer": "Answer 1", "confidence": "high"},
            {"answer": "Answer 2", "confidence": "medium"}
        ]
        
        questions = ["What is ML?", "What is AI?"]
        documents_list = [sample_documents[:2], sample_documents[1:]]
        
        results = generator.batch_generate_answers(questions, documents_list)
        
        assert len(results) == 2
        assert results[0]["answer"] == "Answer 1"
        assert results[1]["answer"] == "Answer 2"
        
        # Verify individual calls
        assert generator.generate_answer.call_count == 2
    
    @patch('src.answer_generator.ChatGroq')
    def test_batch_generate_answers_mismatch(self, mock_chat_groq):
        """Test batch generation with mismatched inputs."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        questions = ["What is ML?", "What is AI?"]
        documents_list = [[]]  # Only one document list for two questions
        
        with pytest.raises(RetrievalError, match="Number of questions must match"):
            generator.batch_generate_answers(questions, documents_list)
    
    @patch('src.answer_generator.ChatGroq')
    def test_update_temperature(self, mock_chat_groq):
        """Test updating temperature parameter."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator(temperature=0.1)
        assert generator.temperature == 0.1
        
        generator.update_temperature(0.5)
        assert generator.temperature == 0.5
        
        # Verify LLM was recreated
        assert mock_chat_groq.call_count == 2  # Once during init, once during update
        
        # Test invalid temperature
        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 1.0"):
            generator.update_temperature(1.5)
        
        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 1.0"):
            generator.update_temperature(-0.1)
    
    @patch('src.answer_generator.ChatGroq')
    def test_get_model_info(self, mock_chat_groq):
        """Test getting model information."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator(
            model_name="test-model",
            temperature=0.3,
            max_tokens=500
        )
        
        info = generator.get_model_info()
        
        assert info["model_name"] == "test-model"
        assert info["temperature"] == 0.3
        assert info["max_tokens"] == 500
        assert info["provider"] == "Groq"
        assert "capabilities" in info
        assert "answer_generation" in info["capabilities"]
    
    @patch('src.answer_generator.ChatGroq')
    def test_validate_documents_valid(self, mock_chat_groq, sample_documents):
        """Test document validation with valid documents."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        assert generator.validate_documents(sample_documents) is True
    
    @patch('src.answer_generator.ChatGroq')
    def test_validate_documents_invalid(self, mock_chat_groq):
        """Test document validation with invalid documents."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        # Test empty list
        assert generator.validate_documents([]) is False
        
        # Test document with empty content
        invalid_docs = [Document(page_content="", metadata={})]
        assert generator.validate_documents(invalid_docs) is False
        
        # Test document with whitespace-only content
        whitespace_docs = [Document(page_content="   ", metadata={})]
        assert generator.validate_documents(whitespace_docs) is False
        
        # Test document with invalid metadata
        valid_doc = Document(page_content="test", metadata={})
        valid_doc.metadata = "not_dict"  # Manually set invalid metadata
        invalid_metadata_docs = [valid_doc]
        assert generator.validate_documents(invalid_metadata_docs) is False
    
    @patch('src.answer_generator.ChatGroq')
    def test_assess_sufficiency_parsing(self, mock_chat_groq):
        """Test sufficiency assessment parsing."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        # Mock sufficiency chain
        generator.sufficiency_chain = Mock()
        
        # Test sufficient response
        generator.sufficiency_chain.invoke.return_value = "SUFFICIENT\nThe context provides adequate information."
        result = generator._assess_sufficiency("test question", "test context")
        assert result["sufficient"] is True
        
        # Test insufficient response
        generator.sufficiency_chain.invoke.return_value = "INSUFFICIENT\nThe context lacks key information."
        result = generator._assess_sufficiency("test question", "test context")
        assert result["sufficient"] is False
        
        # Test error handling
        generator.sufficiency_chain.invoke.side_effect = Exception("Assessment failed")
        result = generator._assess_sufficiency("test question", "test context")
        assert result["sufficient"] is True  # Default to sufficient on error
    
    @patch('src.answer_generator.ChatGroq')
    def test_answer_generation_failure(self, mock_chat_groq, sample_documents):
        """Test answer generation when LLM fails."""
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        generator = AnswerGenerator()
        
        # Mock chain failure
        generator.answer_chain = Mock()
        generator.answer_chain.invoke.side_effect = Exception("LLM failed")
        
        with pytest.raises(RetrievalError, match="Failed to generate answer"):
            generator.generate_answer("What is AI?", sample_documents)


class TestAnswerGeneratorIntegration:
    """Integration tests for AnswerGenerator with real components."""
    
    def test_context_preparation_consistency(self):
        """Test that context preparation is consistent."""
        from src.answer_generator import AnswerGenerator
        from unittest.mock import Mock, patch
        
        with patch('src.answer_generator.ChatGroq') as mock_chat_groq:
            mock_llm_instance = Mock()
            mock_chat_groq.return_value = mock_llm_instance
            
            generator = AnswerGenerator()
            
            docs = [
                Document(
                    page_content="Test content",
                    metadata={"source": "test.pdf", "page": 1, "chunk_id": "test_1"}
                )
            ]
            
            context1 = generator._prepare_context(docs)
            context2 = generator._prepare_context(docs)
            
            assert context1 == context2
    
    def test_citation_extraction_completeness(self):
        """Test that all document information is captured in citations."""
        from src.answer_generator import AnswerGenerator
        from unittest.mock import Mock, patch
        
        with patch('src.answer_generator.ChatGroq') as mock_chat_groq:
            mock_llm_instance = Mock()
            mock_chat_groq.return_value = mock_llm_instance
            
            generator = AnswerGenerator()
            
            docs = [
                Document(
                    page_content="Content 1",
                    metadata={"source": "doc1.pdf", "page": 1, "chunk_id": "c1", "rerank_score": 0.9}
                ),
                Document(
                    page_content="Content 2",
                    metadata={"source": "doc2.pdf", "page": 2, "chunk_id": "c2", "retrieval_score": 0.8}
                )
            ]
            
            citations = generator._extract_citations(docs)
            
            assert len(citations) == 2
            assert citations[0]["relevance_score"] == 0.9  # Uses rerank_score
            assert citations[1]["relevance_score"] == 0.8  # Falls back to retrieval_score