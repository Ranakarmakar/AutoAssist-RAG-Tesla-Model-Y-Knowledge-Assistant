"""Text chunking engine for PDF RAG System."""

from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import settings
from src.logging_config import get_logger
from src.exceptions import DocumentProcessingError

logger = get_logger(__name__)


class ChunkingEngine:
    """Handles document chunking using LangChain's RecursiveCharacterTextSplitter."""
    
    def __init__(
        self, 
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the chunking engine.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )
        
        self.logger = logger
        self.logger.info(
            "ChunkingEngine initialized",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into smaller chunks.
        
        Args:
            documents: List of LangChain Document objects to split
            
        Returns:
            List of chunked Document objects with preserved and enhanced metadata
            
        Raises:
            DocumentProcessingError: If chunking fails
        """
        if not documents:
            self.logger.warning("No documents provided for chunking")
            return []
        
        try:
            self.logger.info("Starting document chunking", document_count=len(documents))
            
            # Split documents using LangChain's text splitter
            chunked_docs = self.text_splitter.split_documents(documents)
            
            # Enhance metadata for each chunk
            enhanced_chunks = []
            for i, chunk in enumerate(chunked_docs):
                enhanced_chunk = self._enhance_chunk_metadata(chunk, i)
                enhanced_chunks.append(enhanced_chunk)
            
            self.logger.info(
                "Document chunking completed",
                original_documents=len(documents),
                total_chunks=len(enhanced_chunks),
                avg_chunk_size=sum(len(chunk.page_content) for chunk in enhanced_chunks) / len(enhanced_chunks) if enhanced_chunks else 0
            )
            
            return enhanced_chunks
            
        except Exception as e:
            error_msg = f"Failed to chunk documents: {str(e)}"
            self.logger.error("Document chunking failed", error=str(e))
            raise DocumentProcessingError(error_msg) from e
    
    def _enhance_chunk_metadata(self, chunk: Document, chunk_index: int) -> Document:
        """
        Enhance chunk metadata with additional information.
        
        Args:
            chunk: Original chunk document
            chunk_index: Index of the chunk in the overall sequence
            
        Returns:
            Document with enhanced metadata
        """
        # Create enhanced metadata
        enhanced_metadata = chunk.metadata.copy()
        
        # Add chunking-specific metadata
        enhanced_metadata.update({
            'chunk_index': chunk_index,
            'chunk_size': len(chunk.page_content),
            'chunk_word_count': len(chunk.page_content.split()),
            'chunk_char_count': len(chunk.page_content),
            'chunking_method': 'RecursiveCharacterTextSplitter',
            'chunk_overlap_size': self.chunk_overlap,
            'max_chunk_size': self.chunk_size,
        })
        
        # Generate unique chunk ID
        source_file = enhanced_metadata.get('source_file', 'unknown')
        page_number = enhanced_metadata.get('page_number', 'unknown')
        chunk_id = f"{source_file}_page_{page_number}_chunk_{chunk_index}"
        enhanced_metadata['chunk_id'] = chunk_id
        
        # Create new document with enhanced metadata
        enhanced_chunk = Document(
            page_content=chunk.page_content,
            metadata=enhanced_metadata
        )
        
        return enhanced_chunk
    
    def create_chunk_metadata(self, chunk: Document) -> Dict[str, Any]:
        """
        Extract and format chunk metadata for external use.
        
        Args:
            chunk: Document chunk
            
        Returns:
            Dictionary containing chunk metadata
        """
        metadata = chunk.metadata.copy()
        
        # Add computed metadata
        metadata.update({
            'content_preview': chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content,
            'has_content': bool(chunk.page_content.strip()),
            'is_valid_chunk': self._validate_chunk(chunk),
        })
        
        return metadata
    
    def _validate_chunk(self, chunk: Document) -> bool:
        """
        Validate that a chunk meets quality criteria.
        
        Args:
            chunk: Document chunk to validate
            
        Returns:
            True if chunk is valid, False otherwise
        """
        # Check if chunk has content
        if not chunk.page_content or not chunk.page_content.strip():
            return False
        
        # Check minimum content length (at least 10 characters)
        if len(chunk.page_content.strip()) < 10:
            return False
        
        # Check that chunk is not just whitespace or special characters
        printable_chars = sum(1 for c in chunk.page_content if c.isalnum())
        if printable_chars < 5:  # At least 5 alphanumeric characters
            return False
        
        return True
    
    def get_chunk_statistics(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Generate statistics about the chunked documents.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary containing chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'total_words': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'valid_chunks': 0,
                'invalid_chunks': 0
            }
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        word_counts = [len(chunk.page_content.split()) for chunk in chunks]
        valid_chunks = sum(1 for chunk in chunks if self._validate_chunk(chunk))
        
        stats = {
            'total_chunks': len(chunks),
            'total_characters': sum(chunk_sizes),
            'total_words': sum(word_counts),
            'avg_chunk_size': sum(chunk_sizes) / len(chunks),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'valid_chunks': valid_chunks,
            'invalid_chunks': len(chunks) - valid_chunks,
            'chunk_size_distribution': {
                'small_chunks': sum(1 for size in chunk_sizes if size < self.chunk_size * 0.5),
                'medium_chunks': sum(1 for size in chunk_sizes if self.chunk_size * 0.5 <= size < self.chunk_size * 0.9),
                'large_chunks': sum(1 for size in chunk_sizes if size >= self.chunk_size * 0.9),
            }
        }
        
        return stats
    
    def chunk_single_document(self, document: Document) -> List[Document]:
        """
        Chunk a single document.
        
        Args:
            document: Single document to chunk
            
        Returns:
            List of chunked documents
        """
        return self.split_documents([document])
    
    def reconfigure(
        self, 
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ) -> None:
        """
        Reconfigure the chunking engine with new parameters.
        
        Args:
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap
            separators: New separators list
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
        if separators is not None:
            self.separators = separators
        
        # Recreate text splitter with new configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )
        
        self.logger.info(
            "ChunkingEngine reconfigured",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )