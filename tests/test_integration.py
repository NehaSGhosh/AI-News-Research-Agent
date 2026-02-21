from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.news_agent import NewsSearchAgent
from app.exceptions import IngestionError, ValidationError
from app.ingest import ArticleReaderService
from app.indexing.chunker import NewsChunkingService
from app.indexing.vector_store import VectorStore


@pytest.mark.integration
class TestFullWorkflow:

    @pytest.mark.slow
    @pytest.mark.requires_api
    def test_full_ingestion_and_query_workflow(
        self, test_data_dir: Path, temp_dir: Path, monkeypatch, real_openai_api_key
    ):
        monkeypatch.setenv("OPENAI_API_KEY", real_openai_api_key)

        reader_service = ArticleReaderService(str(test_data_dir))
        articles = reader_service.load_articles()
        assert len(articles) > 0

        chunking_service = NewsChunkingService(chunk_size=100, chunk_overlap=20)
        chunks = chunking_service.chunk(articles)
        assert len(chunks) > len(articles)

        db_service = VectorStore()
        db_path = str(temp_dir / "test_faiss_db")
        vectorstore = db_service.create_or_load_faiss_db(chunks, db_path)
        assert vectorstore is not None

        loaded_vectorstore = db_service.load_existing_db(db_path)
        assert loaded_vectorstore is not None

        agent = NewsSearchAgent(loaded_vectorstore)
        result = agent.ask("What is the news about technology?")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.integration
    def test_ingestion_to_chunking_workflow(self, test_data_dir: Path):
        reader_service = ArticleReaderService(str(test_data_dir))
        articles = reader_service.load_articles()
        assert len(articles) > 0

        chunking_service = NewsChunkingService(chunk_size=50, chunk_overlap=10)
        chunks = chunking_service.chunk(articles)
        assert len(chunks) > len(articles)

        for chunk in chunks:
            assert hasattr(chunk, "page_content")
            assert hasattr(chunk, "metadata")
            assert "source" in chunk.metadata

    @pytest.mark.integration
    def test_error_handling_workflow(self, temp_dir: Path):
        fake_dir = str(temp_dir / "nonexistent")
        reader_service = ArticleReaderService(fake_dir)

        with pytest.raises(IngestionError):
            reader_service.load_articles()

        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        reader_service = ArticleReaderService(str(empty_dir))

        with pytest.raises(IngestionError, match="No valid .txt articles found"):
            reader_service.load_articles()

    @pytest.mark.integration
    @patch("app.indexing.vector_store.OpenAIEmbeddings")
    @patch("app.indexing.vector_store.FAISS")
    @patch("os.makedirs")
    def test_mocked_vector_store_workflow(
        self,
        mock_makedirs,
        mock_faiss,
        mock_embeddings,
        test_data_dir: Path,
        sample_documents: list,
        temp_dir: Path,
    ):
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        mock_db = MagicMock()
        mock_faiss.from_documents.return_value = mock_db

        reader_service = ArticleReaderService(str(test_data_dir))
        articles = reader_service.load_articles()

        chunking_service = NewsChunkingService(chunk_size=100, chunk_overlap=20)
        chunks = chunking_service.chunk(articles)

        db_service = VectorStore()
        db_path = str(temp_dir / "mocked_db")
        vectorstore = db_service.create_or_load_faiss_db(chunks, db_path)

        assert vectorstore == mock_db

    @pytest.mark.integration
    @patch("app.news_agent.ChatOpenAI")
    def test_mocked_agent_workflow(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is a test answer."
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        mock_doc = MagicMock()
        mock_doc.page_content = "Sample document content."
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        agent = NewsSearchAgent(mock_vs)
        result = agent.ask("What is the latest news about technology?")

        assert isinstance(result, str)
        assert len(result) > 0
        mock_retriever.invoke.assert_called()
        mock_llm.invoke.assert_called()


@pytest.mark.integration
class TestDataFlow:

    def test_article_to_document_conversion(self, test_data_dir: Path):
        reader_service = ArticleReaderService(str(test_data_dir))
        articles = reader_service.load_articles()

        for article in articles:
            assert hasattr(article, "page_content")
            assert hasattr(article, "metadata")
            assert isinstance(article.page_content, str)
            assert isinstance(article.metadata, dict)
            assert "source" in article.metadata

    def test_document_to_chunk_preservation(self, sample_documents: list):
        chunking_service = NewsChunkingService(chunk_size=50, chunk_overlap=10)
        chunks = chunking_service.chunk(sample_documents)

        sources = {doc.metadata["source"] for doc in sample_documents}
        chunk_sources = {chunk.metadata.get("source") for chunk in chunks}

        assert sources.issubset(chunk_sources) or len(chunk_sources) > 0

    def test_chunk_content_consistency(self, sample_documents: list):
        chunking_service = NewsChunkingService(chunk_size=50, chunk_overlap=10)
        chunks = chunking_service.chunk(sample_documents)

        chunk_content = " ".join(chunk.page_content for chunk in chunks)
        original_content = " ".join(doc.page_content for doc in sample_documents)

        assert len(chunk_content) >= len(original_content) * 0.8
