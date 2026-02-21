from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.exceptions import IngestionError, ValidationError


class NewsChunkingService:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 60):
        """
        Initialize the chunking service with specified chunk size and overlap.

        Args:
            chunk_size (int): Maximum number of characters per chunk (must be > 0).
            chunk_overlap (int): Number of overlapping characters between chunks (>= 0 and < chunk_size).

        Raises:
            ValidationError: If parameters are invalid.
        """
        if chunk_size <= 0 or chunk_overlap < 0:
            raise ValidationError(
                "chunk_size must be positive and chunk_overlap must be non-negative"
            )
        if chunk_overlap >= chunk_size:
            raise ValidationError("chunk_overlap must be less than chunk_size")
        try:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", " ", ""],
            )
        except Exception as e:
            raise ValidationError(f"Invalid chunking parameters: {e}") from e

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into smaller chunks.

        Args:
            documents (List[Document]): List of Document objects to be chunked.

        Returns:
            List[Document]: List of chunked Document objects with the same metadata.

        Raises:
            ValidationError: If no documents are provided.
            IngestionError: If chunking fails due to internal errors.
        """
        if not documents:
            raise ValidationError("No documents provided for chunking")
        try:
            return self.splitter.split_documents(documents)
        except Exception as e:
            raise IngestionError(f"Failed to chunk documents: {e}") from e