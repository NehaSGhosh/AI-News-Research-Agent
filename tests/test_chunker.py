import pytest

from app.exceptions import IngestionError, ValidationError
from app.indexing.chunker import NewsChunkingService
from langchain_core.documents import Document


@pytest.mark.unit
class TestNewsChunkingService:

    def test_init_with_invalid_chunk_size(self):
        with pytest.raises(ValidationError, match="chunk_size must be positive"):
            NewsChunkingService(chunk_size=0)

    def test_init_with_negative_chunk_size(self):
        with pytest.raises(ValidationError, match="chunk_size must be positive"):
            NewsChunkingService(chunk_size=-1)

    def test_init_with_negative_chunk_overlap(self):
        with pytest.raises(ValidationError, match="chunk_overlap must be non-negative"):
            NewsChunkingService(chunk_overlap=-1)

    def test_init_with_overlap_greater_than_size(self):
        with pytest.raises(ValidationError, match="chunk_overlap must be less than chunk_size"):
            NewsChunkingService(chunk_size=100, chunk_overlap=100)

    def test_init_with_valid_parameters(self):
        service = NewsChunkingService(chunk_size=100, chunk_overlap=20)
        assert service.splitter is not None

    def test_chunk_with_empty_documents(self):
        service = NewsChunkingService()
        with pytest.raises(ValidationError, match="No documents provided"):
            service.chunk([])

    def test_chunk_with_none_documents(self):
        service = NewsChunkingService()
        with pytest.raises(ValidationError):
            service.chunk(None)

    def test_chunk_single_document(self, sample_documents: list[Document]):
        service = NewsChunkingService(chunk_size=50, chunk_overlap=10)
        chunks = service.chunk([sample_documents[0]])
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_chunk_multiple_documents(self):
        long_docs = [
            Document(
                page_content="This is a very long article about technology and AI advances. " * 20,
                metadata={"source": "long1.txt"},
            ),
            Document(
                page_content="Science news covering various breakthroughs in space exploration. " * 20,
                metadata={"source": "long2.txt"},
            ),
        ]
        service = NewsChunkingService(chunk_size=50, chunk_overlap=10)
        chunks = service.chunk(long_docs)
        assert len(chunks) > len(long_docs)
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_chunk_preserves_metadata(self, sample_documents: list[Document]):
        service = NewsChunkingService(chunk_size=50, chunk_overlap=10)
        chunks = service.chunk(sample_documents)
        assert all(hasattr(chunk, "metadata") for chunk in chunks)
        assert all("source" in chunk.metadata for chunk in chunks)

    def test_chunk_respects_chunk_size(self, sample_documents: list[Document]):
        chunk_size = 30
        service = NewsChunkingService(chunk_size=chunk_size, chunk_overlap=5)
        chunks = service.chunk(sample_documents)

        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size * 2

    def test_chunk_with_large_document(self):
        large_content = "This is a very long article. " * 1000
        large_doc = Document(
            page_content=large_content,
            metadata={"source": "large.txt"},
        )

        service = NewsChunkingService(chunk_size=100, chunk_overlap=20)
        chunks = service.chunk([large_doc])
        assert len(chunks) > 1
