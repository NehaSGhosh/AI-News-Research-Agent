"""Custom exceptions for the news search agent application."""


class NewsSearchError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, *args, **kwargs):
        """
        Initialize the exception with an error message.

        Args:
            message (str): Human-readable error message describing the issue.
            *args: Additional positional arguments for the base Exception.
            **kwargs: Additional keyword arguments for the base Exception.
        """
        self.message = message
        super().__init__(message, *args, **kwargs)


class ConfigurationError(NewsSearchError):
    """Raised when configuration is invalid or missing."""


class IngestionError(NewsSearchError):
    """Raised when document ingestion fails."""


class VectorStoreError(NewsSearchError):
    """Raised when vector store operations fail."""


class RetrievalError(NewsSearchError):
    """Raised when document retrieval fails."""


class ModelError(NewsSearchError):
    """Raised when LLM or embedding model invocation fails."""


class ValidationError(NewsSearchError):
    """Raised when input validation fails."""
