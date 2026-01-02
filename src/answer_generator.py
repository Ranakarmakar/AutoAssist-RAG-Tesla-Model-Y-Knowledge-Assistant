"""Answer generation component for PDF RAG System using Groq LLM."""

from typing import List, Dict, Any, Optional, Tuple
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.config import settings
from src.logging_config import get_logger
from src.exceptions import RetrievalError

logger = get_logger(__name__)


class AnswerGenerator:
    """Generates comprehensive answers with citations using Groq LLM."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the answer generator.
        
        Args:
            model_name: Groq model name to use
            temperature: Temperature for text generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name or settings.llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger
        
        try:
            self.logger.info(
                "Initializing Groq LLM for answer generation",
                model_name=self.model_name,
                temperature=self.temperature
            )
            
            # Initialize Groq LLM
            self.llm = ChatGroq(
                groq_api_key=settings.groq_api_key,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Initialize prompt templates
            self._setup_prompts()
            
            # Initialize chains
            self._setup_chains()
            
            self.logger.info("Answer generator initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize answer generator: {str(e)}"
            self.logger.error("Answer generator initialization failed", error=str(e))
            raise RetrievalError(error_msg) from e
    
    def _setup_prompts(self) -> None:
        """Set up prompt templates for answer generation."""
        
        # Main answer generation prompt with citations
        self.answer_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert assistant that provides comprehensive, accurate answers based on provided document excerpts.

Your task is to answer the user's question using ONLY the information provided in the context below. Follow these guidelines:

1. **Answer Accuracy**: Base your answer strictly on the provided context. Do not add information from your general knowledge.

2. **Citations**: For every piece of information you use, include a citation in the format [Source: filename, Page: X] where X is the page number from the document metadata.

3. **Comprehensive Response**: Provide a thorough answer that synthesizes information from multiple sources when available.

4. **Insufficient Information**: If the provided context does not contain enough information to answer the question, clearly state "I don't have sufficient information in the provided documents to fully answer this question" and explain what information is missing.

5. **Factual Accuracy**: Ensure all statements are directly supported by the source material.

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Prompt for insufficient information detection
        self.sufficiency_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Analyze whether the provided context contains sufficient information to answer the user's question.

Context:
{context}

Question: {question}

Respond with either:
- "SUFFICIENT" if the context contains enough information to provide a meaningful answer
- "INSUFFICIENT" if the context lacks key information needed to answer the question

Also provide a brief explanation of your assessment.

Assessment:"""
        )
        
        # Prompt for citation extraction and formatting
        self.citation_prompt = PromptTemplate(
            input_variables=["documents"],
            template="""Extract citation information from the provided documents and format them properly.

For each document, create a citation in the format:
[Source: filename, Page: X, Chunk: Y]

Documents:
{documents}

Citations:"""
        )
    
    def _setup_chains(self) -> None:
        """Set up LangChain chains for answer generation."""
        
        # Answer generation chain
        self.answer_chain = (
            self.answer_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Sufficiency assessment chain
        self.sufficiency_chain = (
            self.sufficiency_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Citation extraction chain
        self.citation_chain = (
            self.citation_prompt 
            | self.llm 
            | StrOutputParser()
        )
    
    def generate_answer(
        self, 
        question: str, 
        documents: List[Document],
        include_citations: bool = True,
        check_sufficiency: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive answer with citations based on retrieved documents.
        
        Args:
            question: User's question
            documents: List of relevant documents
            include_citations: Whether to include citation information
            check_sufficiency: Whether to check information sufficiency
            
        Returns:
            Dictionary containing answer, citations, and metadata
            
        Raises:
            RetrievalError: If answer generation fails
        """
        if not question or not question.strip():
            raise RetrievalError("Question cannot be empty")
        
        if not documents:
            return {
                "answer": "I don't have any relevant documents to answer this question. Please provide more context or try a different query.",
                "citations": [],
                "sufficient_information": False,
                "source_count": 0,
                "confidence": "low",
                "model_used": self.model_name,
                "question": question.strip()
            }
        
        try:
            self.logger.info(
                "Generating answer",
                question=question,
                document_count=len(documents)
            )
            
            # Prepare context from documents
            context = self._prepare_context(documents)
            
            # Check information sufficiency if requested
            sufficient_info = True
            sufficiency_explanation = ""
            
            if check_sufficiency:
                sufficiency_result = self._assess_sufficiency(question, context)
                sufficient_info = sufficiency_result["sufficient"]
                sufficiency_explanation = sufficiency_result["explanation"]
            
            # Generate answer
            answer = self.answer_chain.invoke({
                "context": context,
                "question": question.strip()
            })
            
            # Extract and format citations
            citations = []
            if include_citations:
                citations = self._extract_citations(documents)
            
            # Determine confidence level
            confidence = self._assess_confidence(answer, documents, sufficient_info)
            
            result = {
                "answer": answer.strip(),
                "citations": citations,
                "sufficient_information": sufficient_info,
                "sufficiency_explanation": sufficiency_explanation,
                "source_count": len(documents),
                "confidence": confidence,
                "model_used": self.model_name,
                "question": question.strip()
            }
            
            self.logger.info(
                "Answer generated successfully",
                question=question,
                answer_length=len(answer),
                citation_count=len(citations),
                sufficient_info=sufficient_info,
                confidence=confidence
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to generate answer: {str(e)}"
            self.logger.error("Answer generation failed", error=str(e), question=question)
            raise RetrievalError(error_msg) from e
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """
        Prepare context string from documents with proper formatting.
        
        Args:
            documents: List of documents to format
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            # Extract metadata
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', f'chunk_{i}')
            
            # Format document section
            context_part = f"""
Document {i}:
Source: {source}
Page: {page}
Chunk ID: {chunk_id}
Content: {doc.page_content}
---"""
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _assess_sufficiency(self, question: str, context: str) -> Dict[str, Any]:
        """
        Assess whether the context contains sufficient information to answer the question.
        
        Args:
            question: User's question
            context: Formatted context from documents
            
        Returns:
            Dictionary with sufficiency assessment
        """
        try:
            assessment = self.sufficiency_chain.invoke({
                "context": context,
                "question": question
            })
            
            # Parse assessment
            lines = assessment.strip().split('\n')
            first_line = lines[0].strip().upper()
            
            # Check for insufficient first, then sufficient
            if "INSUFFICIENT" in first_line:
                sufficient = False
            elif "SUFFICIENT" in first_line:
                sufficient = True
            else:
                # Default to sufficient if unclear
                sufficient = True
            
            explanation = assessment.strip()
            
            return {
                "sufficient": sufficient,
                "explanation": explanation,
                "raw_assessment": assessment
            }
            
        except Exception as e:
            self.logger.warning("Failed to assess sufficiency", error=str(e))
            return {
                "sufficient": True,  # Default to sufficient if assessment fails
                "explanation": "Could not assess information sufficiency",
                "raw_assessment": ""
            }
    
    def _extract_citations(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract and format citations from documents.
        
        Args:
            documents: List of documents to extract citations from
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        for i, doc in enumerate(documents, 1):
            citation = {
                "id": i,
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'Unknown'),
                "chunk_id": doc.metadata.get('chunk_id', f'chunk_{i}'),
                "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                "relevance_score": doc.metadata.get('rerank_score', doc.metadata.get('retrieval_score', 0.0))
            }
            citations.append(citation)
        
        return citations
    
    def _assess_confidence(
        self, 
        answer: str, 
        documents: List[Document], 
        sufficient_info: bool
    ) -> str:
        """
        Assess confidence level of the generated answer.
        
        Args:
            answer: Generated answer
            documents: Source documents
            sufficient_info: Whether information was sufficient
            
        Returns:
            Confidence level string
        """
        # Simple heuristic-based confidence assessment
        if not sufficient_info:
            return "low"
        
        if "I don't have sufficient information" in answer or "insufficient information" in answer.lower():
            return "low"
        
        # Adjust thresholds for more accurate confidence assessment
        if len(documents) >= 3 and len(answer) > 200:
            return "high"
        elif len(documents) >= 2 and len(answer) > 100:
            return "medium"
        else:
            return "low"
    
    def generate_answer_with_retriever(
        self, 
        question: str, 
        retriever,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate answer using a retriever to get relevant documents.
        
        Args:
            question: User's question
            retriever: LangChain retriever instance
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary containing answer and metadata
            
        Raises:
            RetrievalError: If retrieval or answer generation fails
        """
        try:
            self.logger.info("Generating answer with retriever", question=question)
            
            # Retrieve relevant documents
            if hasattr(retriever, 'invoke'):
                documents = retriever.invoke(question)
            else:
                documents = retriever.get_relevant_documents(question)
            
            # Limit documents if top_k specified
            if top_k and len(documents) > top_k:
                documents = documents[:top_k]
            
            # Generate answer using retrieved documents
            return self.generate_answer(question, documents)
            
        except Exception as e:
            error_msg = f"Failed to generate answer with retriever: {str(e)}"
            self.logger.error("Answer generation with retriever failed", error=str(e))
            raise RetrievalError(error_msg) from e
    
    def format_answer_with_citations(self, result: Dict[str, Any]) -> str:
        """
        Format the answer result into a readable string with citations.
        
        Args:
            result: Answer generation result dictionary
            
        Returns:
            Formatted answer string
        """
        answer = result["answer"]
        citations = result["citations"]
        
        formatted_parts = [answer]
        
        if citations:
            formatted_parts.append("\n\n**Sources:**")
            for citation in citations:
                source_line = f"- {citation['source']}, Page {citation['page']}"
                if citation.get('relevance_score'):
                    source_line += f" (Relevance: {citation['relevance_score']:.3f})"
                formatted_parts.append(source_line)
        
        if not result["sufficient_information"]:
            formatted_parts.append(f"\n\n**Note:** {result['sufficiency_explanation']}")
        
        formatted_parts.append(f"\n\n**Confidence:** {result['confidence'].title()}")
        
        return "\n".join(formatted_parts)
    
    def batch_generate_answers(
        self, 
        questions: List[str], 
        documents_list: List[List[Document]]
    ) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple questions efficiently.
        
        Args:
            questions: List of questions
            documents_list: List of document lists (one per question)
            
        Returns:
            List of answer generation results
            
        Raises:
            RetrievalError: If batch generation fails
        """
        if len(questions) != len(documents_list):
            raise RetrievalError("Number of questions must match number of document lists")
        
        try:
            self.logger.info("Starting batch answer generation", question_count=len(questions))
            
            results = []
            for question, documents in zip(questions, documents_list):
                result = self.generate_answer(question, documents)
                results.append(result)
            
            self.logger.info("Batch answer generation completed", result_count=len(results))
            
            return results
            
        except Exception as e:
            error_msg = f"Failed to perform batch answer generation: {str(e)}"
            self.logger.error("Batch answer generation failed", error=str(e))
            raise RetrievalError(error_msg) from e
    
    def update_temperature(self, new_temperature: float) -> None:
        """
        Update the temperature parameter for text generation.
        
        Args:
            new_temperature: New temperature value (0.0-1.0)
        """
        if not 0.0 <= new_temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        
        old_temperature = self.temperature
        self.temperature = new_temperature
        
        # Reinitialize LLM with new temperature
        self.llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Reinitialize chains
        self._setup_chains()
        
        self.logger.info(
            "Updated temperature parameter",
            old_temperature=old_temperature,
            new_temperature=new_temperature
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the answer generator configuration.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": "Groq",
            "capabilities": [
                "answer_generation",
                "citation_formatting",
                "sufficiency_assessment",
                "confidence_scoring",
                "batch_processing"
            ]
        }
    
    def validate_documents(self, documents: List[Document]) -> bool:
        """
        Validate that documents are suitable for answer generation.
        
        Args:
            documents: List of documents to validate
            
        Returns:
            True if documents are valid for answer generation
        """
        if not documents:
            return False
        
        for doc in documents:
            if not doc.page_content or not doc.page_content.strip():
                return False
            
            if not isinstance(doc.metadata, dict):
                return False
        
        return True