"""Query rewriting component for PDF RAG System using Groq LLM."""

from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import settings
from src.logging_config import get_logger
from src.exceptions import QueryRewritingError

logger = get_logger(__name__)


class QueryRewriter:
    """Handles query rewriting and enhancement using Groq LLM."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the query rewriter.
        
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
                "Initializing Groq LLM for query rewriting",
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
            
            self.logger.info("Query rewriter initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize query rewriter: {str(e)}"
            self.logger.error("Query rewriter initialization failed", error=str(e))
            raise QueryRewritingError(error_msg) from e
    
    def _setup_prompts(self) -> None:
        """Set up prompt templates for different query rewriting tasks."""
        
        # Query enhancement prompt
        self.enhancement_prompt = PromptTemplate(
            input_variables=["original_query"],
            template="""You are an expert at improving search queries for document retrieval systems.

Your task is to enhance the following query to make it more effective for finding relevant information in technical documents.

Original Query: {original_query}

Please rewrite this query to:
1. Use more specific and technical terminology when appropriate
2. Add relevant synonyms or alternative phrasings
3. Make it more likely to match document content
4. Keep the core intent and meaning intact

Enhanced Query:"""
        )
        
        # Query expansion prompt (multiple variations)
        self.expansion_prompt = PromptTemplate(
            input_variables=["original_query"],
            template="""You are an expert at creating multiple search query variations for document retrieval.

Original Query: {original_query}

Generate 3 different variations of this query that would help find the same information but using different words, phrasings, or approaches. Each variation should:
1. Maintain the original intent
2. Use different terminology or synonyms
3. Approach the topic from slightly different angles
4. Be suitable for searching technical documents

Format your response as a numbered list:
1. [First variation]
2. [Second variation]  
3. [Third variation]"""
        )
        
        # Query clarification prompt
        self.clarification_prompt = PromptTemplate(
            input_variables=["original_query", "context"],
            template="""You are helping to clarify and improve a search query based on additional context.

Original Query: {original_query}
Context: {context}

Based on the context provided, create an improved version of the query that:
1. Incorporates relevant context information
2. Is more specific and targeted
3. Uses appropriate technical terminology
4. Maintains the user's original intent

Improved Query:"""
        )
        
        # Question decomposition prompt
        self.decomposition_prompt = PromptTemplate(
            input_variables=["complex_query"],
            template="""You are an expert at breaking down complex questions into simpler, more focused sub-questions.

Complex Query: {complex_query}

Break this query down into 2-4 simpler, more focused questions that together would provide a complete answer to the original query. Each sub-question should:
1. Focus on a specific aspect of the original query
2. Be clear and unambiguous
3. Be suitable for document search
4. Together cover all aspects of the original query

Format your response as a numbered list:
1. [First sub-question]
2. [Second sub-question]
3. [Third sub-question] (if needed)
4. [Fourth sub-question] (if needed)"""
        )
    
    def _setup_chains(self) -> None:
        """Set up LangChain chains for different query rewriting operations."""
        
        # Enhancement chain
        self.enhancement_chain = (
            self.enhancement_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Expansion chain
        self.expansion_chain = (
            self.expansion_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Clarification chain
        self.clarification_chain = (
            self.clarification_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Decomposition chain
        self.decomposition_chain = (
            self.decomposition_prompt 
            | self.llm 
            | StrOutputParser()
        )
    
    def enhance_query(self, query: str) -> str:
        """
        Enhance a single query to make it more effective for retrieval.
        
        Args:
            query: Original query string
            
        Returns:
            Enhanced query string
            
        Raises:
            QueryRewritingError: If enhancement fails
        """
        if not query or not query.strip():
            raise QueryRewritingError("Query cannot be empty")
        
        try:
            self.logger.info("Enhancing query", original_query=query)
            
            enhanced = self.enhancement_chain.invoke({"original_query": query.strip()})
            enhanced = enhanced.strip()
            
            if not enhanced:
                self.logger.warning("Empty enhanced query returned, using original")
                enhanced = query.strip()
            
            self.logger.info(
                "Query enhanced successfully",
                original_query=query,
                enhanced_query=enhanced
            )
            
            return enhanced
            
        except Exception as e:
            error_msg = f"Failed to enhance query: {str(e)}"
            self.logger.error("Query enhancement failed", error=str(e), query=query)
            raise QueryRewritingError(error_msg) from e
    
    def expand_query(self, query: str) -> List[str]:
        """
        Generate multiple variations of a query.
        
        Args:
            query: Original query string
            
        Returns:
            List of query variations
            
        Raises:
            QueryRewritingError: If expansion fails
        """
        if not query or not query.strip():
            raise QueryRewritingError("Query cannot be empty")
        
        try:
            self.logger.info("Expanding query", original_query=query)
            
            expanded_text = self.expansion_chain.invoke({"original_query": query.strip()})
            
            # Parse the numbered list response
            variations = self._parse_numbered_list(expanded_text)
            
            if not variations:
                self.logger.warning("No variations generated, using original query")
                variations = [query.strip()]
            
            self.logger.info(
                "Query expanded successfully",
                original_query=query,
                variation_count=len(variations)
            )
            
            return variations
            
        except Exception as e:
            error_msg = f"Failed to expand query: {str(e)}"
            self.logger.error("Query expansion failed", error=str(e), query=query)
            raise QueryRewritingError(error_msg) from e
    
    def clarify_query(self, query: str, context: str) -> str:
        """
        Clarify a query using additional context.
        
        Args:
            query: Original query string
            context: Additional context information
            
        Returns:
            Clarified query string
            
        Raises:
            QueryRewritingError: If clarification fails
        """
        if not query or not query.strip():
            raise QueryRewritingError("Query cannot be empty")
        
        try:
            self.logger.info(
                "Clarifying query with context",
                original_query=query,
                context_length=len(context) if context else 0
            )
            
            clarified = self.clarification_chain.invoke({
                "original_query": query.strip(),
                "context": context or ""
            })
            clarified = clarified.strip()
            
            if not clarified:
                self.logger.warning("Empty clarified query returned, using original")
                clarified = query.strip()
            
            self.logger.info(
                "Query clarified successfully",
                original_query=query,
                clarified_query=clarified
            )
            
            return clarified
            
        except Exception as e:
            error_msg = f"Failed to clarify query: {str(e)}"
            self.logger.error("Query clarification failed", error=str(e), query=query)
            raise QueryRewritingError(error_msg) from e
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose a complex query into simpler sub-questions.
        
        Args:
            query: Complex query string
            
        Returns:
            List of simpler sub-questions
            
        Raises:
            QueryRewritingError: If decomposition fails
        """
        if not query or not query.strip():
            raise QueryRewritingError("Query cannot be empty")
        
        try:
            self.logger.info("Decomposing complex query", complex_query=query)
            
            decomposed_text = self.decomposition_chain.invoke({"complex_query": query.strip()})
            
            # Parse the numbered list response
            sub_questions = self._parse_numbered_list(decomposed_text)
            
            if not sub_questions:
                self.logger.warning("No sub-questions generated, using original query")
                sub_questions = [query.strip()]
            
            self.logger.info(
                "Query decomposed successfully",
                complex_query=query,
                sub_question_count=len(sub_questions)
            )
            
            return sub_questions
            
        except Exception as e:
            error_msg = f"Failed to decompose query: {str(e)}"
            self.logger.error("Query decomposition failed", error=str(e), query=query)
            raise QueryRewritingError(error_msg) from e
    
    def rewrite_query_comprehensive(
        self, 
        query: str, 
        include_variations: bool = True,
        include_enhancement: bool = True,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive query rewriting with multiple techniques.
        
        Args:
            query: Original query string
            include_variations: Whether to generate query variations
            include_enhancement: Whether to enhance the query
            context: Optional context for clarification
            
        Returns:
            Dictionary containing all rewritten queries
            
        Raises:
            QueryRewritingError: If rewriting fails
        """
        if not query or not query.strip():
            raise QueryRewritingError("Query cannot be empty")
        
        try:
            self.logger.info(
                "Starting comprehensive query rewriting",
                original_query=query,
                include_variations=include_variations,
                include_enhancement=include_enhancement,
                has_context=bool(context)
            )
            
            result = {
                "original_query": query.strip(),
                "enhanced_query": None,
                "clarified_query": None,
                "query_variations": [],
                "sub_questions": []
            }
            
            # Enhance query if requested
            if include_enhancement:
                result["enhanced_query"] = self.enhance_query(query)
            
            # Clarify with context if provided
            if context:
                result["clarified_query"] = self.clarify_query(query, context)
            
            # Generate variations if requested
            if include_variations:
                result["query_variations"] = self.expand_query(query)
            
            # Decompose if query seems complex (heuristic: contains "and", "or", multiple questions)
            if self._is_complex_query(query):
                result["sub_questions"] = self.decompose_query(query)
            
            self.logger.info(
                "Comprehensive query rewriting completed",
                original_query=query,
                enhanced=bool(result["enhanced_query"]),
                clarified=bool(result["clarified_query"]),
                variations_count=len(result["query_variations"]),
                sub_questions_count=len(result["sub_questions"])
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to perform comprehensive query rewriting: {str(e)}"
            self.logger.error("Comprehensive query rewriting failed", error=str(e), query=query)
            raise QueryRewritingError(error_msg) from e
    
    def _parse_numbered_list(self, text: str) -> List[str]:
        """
        Parse a numbered list from LLM response.
        
        Args:
            text: Text containing numbered list
            
        Returns:
            List of parsed items
        """
        if not text:
            return []
        
        lines = text.strip().split('\n')
        items = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for numbered items (1., 2., etc.)
            if line and (line[0].isdigit() or line.startswith('- ')):
                # Remove numbering and clean up
                if '.' in line and line.split('.')[0].isdigit():
                    item = '.'.join(line.split('.')[1:]).strip()
                elif line.startswith('- '):
                    item = line[2:].strip()
                else:
                    item = line
                
                if item:
                    items.append(item)
        
        return items
    
    def _is_complex_query(self, query: str) -> bool:
        """
        Heuristic to determine if a query is complex and should be decomposed.
        
        Args:
            query: Query string to analyze
            
        Returns:
            True if query appears complex
        """
        query_lower = query.lower()
        
        # Check for complexity indicators
        complexity_indicators = [
            ' and ', ' or ', ' but ', ' however ',
            'difference between', 'compare', 'versus'
        ]
        
        # Count actual complexity indicators
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in query_lower)
        
        # Check for multiple questions (more than one question mark)
        question_count = query.count('?')
        
        # Consider complex if:
        # - Multiple complexity indicators
        # - Multiple questions
        # - Very long query (more than 15 words)
        return (
            indicator_count >= 1 or 
            question_count > 1 or 
            len(query.split()) > 15
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the query rewriter configuration.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": "Groq",
            "capabilities": [
                "query_enhancement",
                "query_expansion", 
                "query_clarification",
                "query_decomposition",
                "comprehensive_rewriting"
            ]
        }