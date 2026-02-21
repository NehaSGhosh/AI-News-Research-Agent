import os
from unittest.mock import MagicMock, patch

import pytest

from app.exceptions import ConfigurationError, VectorStoreError, ValidationError
from app.indexing.vector_store import VectorStore


@pytest.mark.unit
class TestVectorStoreInit:

    @patch("app.indexing.vector_store.OpenAIEmbeddings")
    def test_init_success(self, mock_embeddings_class):
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance

        store = VectorStore()
        assert store.embeddings == mock_embeddings_instance
        mock_embeddings_class.assert_called_once()

    @patch("app.indexing.vector_store.OpenAIEmbeddings")
    def test_init_failure(self, mock_embeddings_class):
        mock_embeddings_class.side_effect = Exception("API key error")

        with pytest.raises(ConfigurationError, match="Failed to initialize embeddings"):
            VectorStore()


@pytest.mark.unit
class TestCreateOrLoadFaissDb:

    @patch("app.indexing.vector_store.OpenAIEmbeddings")
    def test_empty_documents(self, mock_embeddings):
        mock_embeddings.return_value = MagicMock()
        store = VectorStore()
        with pytest.raises(ValidationError, match="No text chunks provided"):
            store.create_or_load_faiss_db([], "./test_db")

    @patch("app.indexing.vector_store.OpenAIEmbeddings")
    def test_empty_path(self, mock_embeddings, sample_documents):
        mock_embeddings.return_value = MagicMock()
        store = VectorStore()
        with pytest.raises(ValidationError, match="Database path cannot be empty"):
            store.create_or_load_faiss_db(sample_documents, "")

    @patch("app.indexing.vector_store.OpenAIEmbeddings")
    def test_whitespace_path(self, mock_embeddings, sample_documents):
        mock_embeddings.return_value = MagicMock()
        store = VectorStore()
        with pytest.raises(ValidationError, match="Database path cannot be empty"):
            store.create_or_load_faiss_db(sample_documents, "   ")

    @patch("app.indexing.vector_store.OpenAIEmbeddings")
    @patch("app.indexing.vector_store.FAISS")
    @patch("os.makedirs")
    def test_success(self, mock_makedirs, mock_faiss, mock_embeddings, sample_documents, temp_dir):
        mock_embeddings.return_value = MagicMock()
        mock_db = MagicMock()
        mock_faiss.from_documents.return_value = mock_db

        store = VectorStore()
        db_path = str(temp_dir / "test_db")
        result = store.create_or_load_faiss_db(sample_documents, db_path)

        assert result == mock_db
        mock_faiss.from_documents.assert_called_once()
        mock_db.save_local.assert_called_once_with(db_path)

    @patch("app.indexing.vector_store.OpenAIEmbeddings")
    @patch("app.indexing.vector_store.FAISS")
    def test_faiss_from_documents_failure(self, mock_faiss, mock_embeddings, sample_documents):
        mock_embeddings.return_value = MagicMock()
        mock_faiss.from_documents.side_effect = Exception("embedding fail")

        store = VectorStore()
        with pytest.raises(VectorStoreError, match="Failed to create FAISS index"):
            store.create_or_load_faiss_db(sample_documents, "./test_db")

    @patch("app.indexing.vector_store.OpenAIEmbeddings")
    @patch("app.indexing.vector_store.FAISS")
    @patch("os.makedirs")
    def test_permission_error_on_save(
        self, mock_makedirs, mock_faiss, mock_embeddings, sample_documents, temp_dir
    ):
        mock_embeddings.return_value = MagicMock()
        mock_db = MagicMock()
        mock_faiss.from_documents.return_value = mock_db
        mock_makedirs.side_effect = PermissionError("Permission denied")

        store = VectorStore()
        db_path = str(temp_dir / "test_db")
        with pytest.raises(VectorStoreError, match="permission denied"):
            store.create_or_load_faiss_db(sample_documents, db_path)

    @patch("app.indexing.vector_store.OpenAIEmbeddings")
    @patch("app.indexing.vector_store.FAISS")
    @patch("os.makedirs")
    def test_os_error_on_save(
        self, mock_makedirs, mock_faiss, mock_embeddings, sample_documents, temp_dir
    ):
        mock_embeddings.return_value = MagicMock()
        mock_db = MagicMock()
        mock_faiss.from_documents.return_value = mock_db
        mock_makedirs.side_effect = OSError("disk error")

        store = VectorStore()
        db_path = str(temp_dir / "test_db")
        with pytest.raises(VectorStoreError, match="Failed to save FAISS index"):
            store.create_or_load_faiss_db(sample_documents, db_path)


@pytest.mark.unit
class TestLoadExistingDb:

    @patch("app.indexing.vector_store.OpenAIEmbeddings")
    def test_empty_path(self, mock_embeddings):
        mock_embeddings.return_value = MagicMock()
        store = VectorStore()
        with pytest.raises(ValidationError, match="Database path cannot be empty"):
            store.load_existing_db("")

    @patch("app.indexing.vector_store.OpenAIEmbeddings")
    def test_nonexistent_path(self, mock_embeddings, temp_dir):
        mock_embeddings.return_value = MagicMock()
        store = VectorStore()
        fake_path = str(temp_dir / "nonexistent_db")
        with pytest.raises(VectorStoreError, match="FAISS index not found"):
            store.load_existing_db(fake_path)

    @patch("app.indexing.vector_store.OpenAIEmbeddings")
    @patch("app.indexing.vector_store.FAISS")
    @patch("os.path.exists")
    def test_success(self, mock_exists, mock_faiss, mock_embeddings, temp_dir):
        mock_embeddings.return_value = MagicMock()
        mock_exists.return_value = True
        mock_db = MagicMock()
        mock_faiss.load_local.return_value = mock_db

        store = VectorStore()
        db_path = str(temp_dir / "existing_db")
        result = store.load_existing_db(db_path)

        assert result == mock_db
        mock_faiss.load_local.assert_called_once()

    @patch("app.indexing.vector_store.OpenAIEmbeddings")
    @patch("app.indexing.vector_store.FAISS")
    @patch("os.path.exists")
    def test_load_error(self, mock_exists, mock_faiss, mock_embeddings, temp_dir):
        mock_embeddings.return_value = MagicMock()
        mock_exists.return_value = True
        mock_faiss.load_local.side_effect = Exception("Load error")

        store = VectorStore()
        db_path = str(temp_dir / "existing_db")
        with pytest.raises(VectorStoreError, match="Failed to load FAISS index"):
            store.load_existing_db(db_path)
