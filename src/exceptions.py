"""Custom exceptions for PDF RAG System."""


class PDFRAGException(Exception):
    """Base exception for PDF RAG System."""
    pass


class DocumentProcessingError(PDFRAGException):
    """Raised when document processing fails."""
    pass


class EmbeddingError(PDFRAGException):
    """Raised when embedding generation fails."""
    pass


class RetrievalError(PDFRAGException):
    """Raised when retrieval fails."""
    pass


class QueryRewritingError(PDFRAGException):
    """Raised when query rewriting fails."""
    pass


class AnswerGenerationError(PDFRAGException):
    """Raised when answer generation fails."""
    pass


class ConfigurationError(PDFRAGException):
    """Raised when configuration is invalid."""
    pass


class VectorStoreError(PDFRAGException):
    """Raised when vector store operations fail."""
    pass


class WorkflowError(PDFRAGException):
    """Raised when workflow execution fails."""
    pass