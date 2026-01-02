"""Main entry point for PDF RAG System."""

import asyncio
from src.config import settings, validate_required_settings
from src.logging_config import configure_logging, get_logger
from src.utils import ensure_directory_exists
from src.exceptions import ConfigurationError

# Import all components for availability
from src.document_processor import DocumentProcessor
from src.chunking_engine import ChunkingEngine
from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore
from src.query_rewriter import QueryRewriter
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.answer_generator import AnswerGenerator
from src.rag_workflow import RAGWorkflow

logger = get_logger(__name__)


def initialize_system() -> None:
    """Initialize the PDF RAG system."""
    # Configure logging
    configure_logging()
    logger.info("PDF RAG System starting up")
    
    # Validate configuration
    try:
        validate_required_settings(settings)
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error("Configuration validation failed", error=str(e))
        raise ConfigurationError(f"Configuration error: {e}")
    
    # Ensure required directories exist
    ensure_directory_exists(settings.vector_store_path)
    ensure_directory_exists(settings.upload_path)
    
    logger.info("System initialization complete")


async def main() -> None:
    """Main application entry point."""
    try:
        initialize_system()
        logger.info("PDF RAG System initialized successfully")
        
        # TODO: Start API server when implemented
        logger.info("System ready for requests")
        
    except Exception as e:
        logger.error("Failed to start PDF RAG System", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())