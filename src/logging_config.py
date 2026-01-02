"""Logging configuration for PDF RAG System."""

import logging
import structlog
from typing import Any, Dict
from src.config import settings


def configure_logging() -> None:
    """Configure structured logging for the application."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper()),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class CorrelationIDProcessor:
    """Add correlation ID to log entries for request tracking."""
    
    def __init__(self):
        self.correlation_id = None
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for current context."""
        self.correlation_id = correlation_id
    
    def __call__(self, logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add correlation ID to event dict."""
        if self.correlation_id:
            event_dict["correlation_id"] = self.correlation_id
        return event_dict


# Global correlation ID processor
correlation_processor = CorrelationIDProcessor()