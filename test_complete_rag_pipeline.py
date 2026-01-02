"""End-to-end test for complete RAG pipeline."""

import pytest
import os
from pathlib import Path

from src.document_processor import DocumentProcessor
from src.chunking_engine import ChunkingEngine
from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore
from src.query_rewriter import QueryRewriter
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.answer_generator import AnswerGenerator
from src.config import settings
from src.exceptions import RetrievalError


class TestCompleteRAGPipeline:
    """Test the complete RAG pipeline end-to-end."""
    
    @pytest.fixture(autouse=True)
    def setup_api_key(self):
        """Ensure API key is available for integration tests."""
        if not settings.groq_api_key:
            pytest.skip("GROQ_API_KEY not set - skipping integration tests")
    
    @pytest.fixture
    def sample_pdf_path(self):
        """Get path to sample PDF for testing."""
        pdf_path = Path("data/Owners_Manual.pdf")
        if not pdf_path.exists():
            pytest.skip("Sample PDF not found - skipping end-to-end test")
        return str(pdf_path)
    
    def test_complete_rag_pipeline(self, sample_pdf_path):
        """Test the complete RAG pipeline from PDF to answer."""
        print("\n=== Testing Complete RAG Pipeline ===")
        
        # Step 1: Document Processing
        print("1. Processing PDF document...")
        doc_processor = DocumentProcessor()
        documents = doc_processor.process_pdf_file(sample_pdf_path)
        
        assert len(documents) > 0, "Should extract documents from PDF"
        print(f"   Extracted {len(documents)} pages from PDF")
        
        # Step 2: Text Chunking
        print("2. Chunking documents...")
        chunker = ChunkingEngine(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split_documents(documents[:3])  # Use first 3 pages for speed
        
        assert len(chunks) > 0, "Should create chunks from documents"
        print(f"   Created {len(chunks)} chunks from documents")
        
        # Step 3: Embedding and Vector Storage
        print("3. Creating embeddings and vector store...")
        embedding_model = EmbeddingModel()
        vector_store = VectorStore(embedding_model.embeddings, store_path="data/test_pipeline_vector_store")
        
        # Clear any existing data
        vector_store.clear_store()
        
        # Add documents to vector store
        vector_store.add_documents(chunks)
        
        doc_count = vector_store.get_document_count()
        assert doc_count == len(chunks), f"Vector store should contain {len(chunks)} documents"
        print(f"   Added {doc_count} documents to vector store")
        
        # Step 4: Query Rewriting
        print("4. Testing query rewriting...")
        query_rewriter = QueryRewriter()
        original_query = "How do I charge the vehicle?"
        
        enhanced_query = query_rewriter.enhance_query(original_query)
        assert len(enhanced_query) > len(original_query), "Enhanced query should be longer"
        print(f"   Original: {original_query}")
        print(f"   Enhanced: {enhanced_query}")
        
        # Step 5: Hybrid Retrieval
        print("5. Testing hybrid retrieval...")
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_weight=0.3,
            semantic_weight=0.7
        )
        
        # Initialize with documents
        hybrid_retriever.add_documents(chunks)
        
        # Retrieve relevant documents
        retrieved_docs = hybrid_retriever.retrieve(enhanced_query, top_k=5)
        
        assert len(retrieved_docs) > 0, "Should retrieve relevant documents"
        print(f"   Retrieved {len(retrieved_docs)} relevant documents")
        
        # Step 6: Reranking
        print("6. Testing reranking...")
        reranker = Reranker(top_k=3)
        
        reranked_docs = reranker.rerank(enhanced_query, retrieved_docs)
        
        assert len(reranked_docs) <= 3, "Should return top 3 reranked documents"
        assert len(reranked_docs) > 0, "Should have reranked documents"
        print(f"   Reranked to top {len(reranked_docs)} documents")
        
        # Verify rerank scores are added
        for doc in reranked_docs:
            assert 'rerank_score' in doc.metadata, "Documents should have rerank scores"
        
        # Step 7: Answer Generation
        print("7. Testing answer generation...")
        answer_generator = AnswerGenerator()
        
        result = answer_generator.generate_answer(original_query, reranked_docs)
        
        # Verify answer quality
        assert isinstance(result, dict), "Should return result dictionary"
        assert "answer" in result, "Should contain answer"
        assert "citations" in result, "Should contain citations"
        assert "confidence" in result, "Should contain confidence"
        
        assert len(result["answer"]) > 50, "Answer should be substantial"
        assert result["source_count"] > 0, "Should have source documents"
        
        print(f"   Generated answer ({len(result['answer'])} chars)")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Citations: {len(result['citations'])}")
        
        # Step 8: Format final answer
        print("8. Formatting final answer...")
        formatted_answer = answer_generator.format_answer_with_citations(result)
        
        assert "**Sources:**" in formatted_answer, "Should include source citations"
        assert "**Confidence:**" in formatted_answer, "Should include confidence"
        
        print("\n=== Complete RAG Pipeline Results ===")
        print(f"Question: {original_query}")
        print(f"Enhanced Query: {enhanced_query}")
        print(f"Documents Retrieved: {len(retrieved_docs)}")
        print(f"Documents Reranked: {len(reranked_docs)}")
        print(f"Answer Length: {len(result['answer'])} characters")
        print(f"Confidence: {result['confidence']}")
        print(f"Citations: {len(result['citations'])}")
        print("\nFormatted Answer:")
        print("-" * 50)
        print(formatted_answer)
        print("-" * 50)
        
        # Cleanup
        vector_store.clear_store()
        
        print("\n✅ Complete RAG Pipeline Test PASSED!")
        
        return {
            "original_query": original_query,
            "enhanced_query": enhanced_query,
            "documents_processed": len(documents),
            "chunks_created": len(chunks),
            "documents_retrieved": len(retrieved_docs),
            "documents_reranked": len(reranked_docs),
            "answer_result": result,
            "formatted_answer": formatted_answer
        }
    
    def test_pipeline_with_multiple_queries(self, sample_pdf_path):
        """Test pipeline with multiple different queries."""
        print("\n=== Testing Pipeline with Multiple Queries ===")
        
        # Setup pipeline components (abbreviated setup)
        doc_processor = DocumentProcessor()
        chunker = ChunkingEngine(chunk_size=500, chunk_overlap=50)
        embedding_model = EmbeddingModel()
        vector_store = VectorStore(embedding_model.embeddings, store_path="data/test_multi_query_vector_store")
        query_rewriter = QueryRewriter()
        hybrid_retriever = HybridRetriever(vector_store=vector_store)
        reranker = Reranker(top_k=3)
        answer_generator = AnswerGenerator()
        
        # Process documents once
        documents = doc_processor.process_pdf_file(sample_pdf_path)
        chunks = chunker.split_documents(documents[:2])  # Use fewer pages for speed
        
        vector_store.clear_store()
        vector_store.add_documents(chunks)
        hybrid_retriever.add_documents(chunks)
        
        # Test multiple queries
        test_queries = [
            "How do I start the vehicle?",
            "What are the safety features?",
            "How do I maintain the battery?"
        ]
        
        results = []
        
        for query in test_queries:
            print(f"\nProcessing query: {query}")
            
            # Full pipeline for each query
            enhanced_query = query_rewriter.enhance_query(query)
            retrieved_docs = hybrid_retriever.retrieve(enhanced_query, top_k=5)
            reranked_docs = reranker.rerank(enhanced_query, retrieved_docs)
            answer_result = answer_generator.generate_answer(query, reranked_docs)
            
            results.append({
                "query": query,
                "enhanced_query": enhanced_query,
                "retrieved_count": len(retrieved_docs),
                "reranked_count": len(reranked_docs),
                "answer_length": len(answer_result["answer"]),
                "confidence": answer_result["confidence"]
            })
            
            print(f"  Retrieved: {len(retrieved_docs)}, Reranked: {len(reranked_docs)}")
            print(f"  Answer length: {len(answer_result['answer'])}, Confidence: {answer_result['confidence']}")
        
        # Verify all queries got reasonable results
        for result in results:
            assert result["retrieved_count"] > 0, f"Should retrieve docs for: {result['query']}"
            assert result["answer_length"] > 20, f"Should generate substantial answer for: {result['query']}"
            assert result["confidence"] in ["low", "medium", "high"], f"Should have valid confidence for: {result['query']}"
        
        # Cleanup
        vector_store.clear_store()
        
        print(f"\n✅ Multi-Query Pipeline Test PASSED! Processed {len(test_queries)} queries successfully.")
        
        return results
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling and edge cases."""
        print("\n=== Testing Pipeline Error Handling ===")
        
        # Test with empty documents
        chunker = ChunkingEngine()
        empty_chunks = chunker.split_documents([])
        assert len(empty_chunks) == 0, "Should handle empty document list"
        
        # Test answer generation with no documents
        answer_generator = AnswerGenerator()
        result = answer_generator.generate_answer("Test question", [])
        assert "I don't have any relevant documents" in result["answer"]
        assert result["confidence"] == "low"
        
        # Test hybrid retriever with no documents
        embedding_model = EmbeddingModel()
        vector_store = VectorStore(embedding_model.embeddings)
        hybrid_retriever = HybridRetriever(vector_store=vector_store)
        
        # Should handle gracefully - expect RetrievalError
        try:
            retrieved = hybrid_retriever.retrieve("test query", top_k=5)
            assert False, "Should have raised RetrievalError"
        except RetrievalError:
            print("✓ Hybrid retriever correctly handles empty document store")
        
        print("✅ Error Handling Test PASSED!")


if __name__ == "__main__":
    # Quick test if run directly
    import sys
    
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not set - cannot run pipeline test")
        sys.exit(1)
    
    pdf_path = "data/Owners_Manual.pdf"
    if not os.path.exists(pdf_path):
        print(f"Sample PDF not found at {pdf_path}")
        sys.exit(1)
    
    # Run quick pipeline test
    test = TestCompleteRAGPipeline()
    test.setup_api_key()
    
    try:
        result = test.test_complete_rag_pipeline(pdf_path)
        print("\n🎉 Quick pipeline test completed successfully!")
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        sys.exit(1)