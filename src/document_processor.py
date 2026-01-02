"""Document processing component for PDF RAG System."""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.config import settings
from src.logging_config import get_logger
from src.exceptions import DocumentProcessingError
from src.utils import validate_pdf_file, get_file_size_mb

logger = get_logger(__name__)


class DocumentProcessor:
    """Handles PDF document loading and processing using LangChain."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.logger = logger
        
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a PDF file and extract its content as LangChain Documents.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of LangChain Document objects with content and metadata
            
        Raises:
            DocumentProcessingError: If PDF loading fails
        """
        try:
            # Validate the PDF file first
            if not validate_pdf_file(pdf_path):
                raise DocumentProcessingError(f"Invalid PDF file: {pdf_path}")
            
            self.logger.info("Loading PDF document", pdf_path=pdf_path)
            
            # Use LangChain's PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if not documents:
                raise DocumentProcessingError(f"No content extracted from PDF: {pdf_path}")
            
            # Enhance metadata
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source_file': os.path.basename(pdf_path),
                    'file_path': pdf_path,
                    'file_size_mb': get_file_size_mb(pdf_path),
                    'page_number': i + 1,
                    'total_pages': len(documents),
                    'chunk_id': f"{os.path.basename(pdf_path)}_page_{i + 1}"
                })
            
            self.logger.info(
                "PDF loaded successfully", 
                pdf_path=pdf_path, 
                pages_extracted=len(documents),
                total_chars=sum(len(doc.page_content) for doc in documents)
            )
            
            return documents
            
        except Exception as e:
            error_msg = f"Failed to load PDF {pdf_path}: {str(e)}"
            self.logger.error("PDF loading failed", pdf_path=pdf_path, error=str(e))
            raise DocumentProcessingError(error_msg) from e
    
    def validate_documents(self, documents: List[Document]) -> bool:
        """
        Validate that documents contain readable, non-empty content.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            True if documents are valid, False otherwise
        """
        if not documents:
            self.logger.warning("No documents to validate")
            return False
        
        for i, doc in enumerate(documents):
            # Check if content is non-empty
            if not doc.page_content or not doc.page_content.strip():
                self.logger.warning("Empty content found", page_number=i + 1)
                return False
            
            # Check if content is readable (contains printable characters)
            printable_chars = sum(1 for c in doc.page_content if c.isprintable() or c.isspace())
            total_chars = len(doc.page_content)
            
            if total_chars > 0:
                readability_ratio = printable_chars / total_chars
                if readability_ratio < 0.8:  # At least 80% printable characters
                    self.logger.warning(
                        "Low readability content", 
                        page_number=i + 1,
                        readability_ratio=readability_ratio
                    )
                    return False
        
        self.logger.info("Document validation passed", document_count=len(documents))
        return True
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file without loading full content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        try:
            if not os.path.exists(pdf_path):
                raise DocumentProcessingError(f"PDF file not found: {pdf_path}")
            
            # Basic file metadata
            file_stat = os.stat(pdf_path)
            metadata = {
                'file_name': os.path.basename(pdf_path),
                'file_path': pdf_path,
                'file_size_bytes': file_stat.st_size,
                'file_size_mb': get_file_size_mb(pdf_path),
                'created_time': file_stat.st_ctime,
                'modified_time': file_stat.st_mtime,
            }
            
            # Try to get PDF-specific metadata using PyPDFLoader
            try:
                loader = PyPDFLoader(pdf_path)
                # Load just the first page to get metadata
                first_page = loader.load()[0] if loader.load() else None
                
                if first_page and hasattr(first_page, 'metadata'):
                    pdf_metadata = first_page.metadata
                    metadata.update({
                        'pdf_metadata': pdf_metadata,
                        'source': pdf_metadata.get('source', pdf_path)
                    })
            except Exception as e:
                self.logger.warning("Could not extract PDF metadata", error=str(e))
            
            return metadata
            
        except Exception as e:
            error_msg = f"Failed to extract metadata from {pdf_path}: {str(e)}"
            self.logger.error("Metadata extraction failed", pdf_path=pdf_path, error=str(e))
            raise DocumentProcessingError(error_msg) from e
    
    def process_pdf_file(self, pdf_path: str) -> List[Document]:
        """
        Complete PDF processing pipeline: load, validate, and return documents.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of validated LangChain Document objects
            
        Raises:
            DocumentProcessingError: If any step in processing fails
        """
        try:
            # Load the PDF
            documents = self.load_pdf(pdf_path)
            
            # Validate the documents
            if not self.validate_documents(documents):
                raise DocumentProcessingError(f"Document validation failed for {pdf_path}")
            
            self.logger.info(
                "PDF processing completed successfully",
                pdf_path=pdf_path,
                document_count=len(documents)
            )
            
            return documents
            
        except DocumentProcessingError:
            raise
        except Exception as e:
            error_msg = f"PDF processing failed for {pdf_path}: {str(e)}"
            self.logger.error("PDF processing pipeline failed", pdf_path=pdf_path, error=str(e))
            raise DocumentProcessingError(error_msg) from e