from unittest.mock import MagicMock, patch

import pytest

from app.exceptions import (
    ConfigurationError,
    IngestionError,
    ModelError,
    NewsSearchError,
    RetrievalError,
    ValidationError,
    VectorStoreError,
)


@pytest.mark.unit
class TestMain:

    @patch("app.main.VectorStore")
    @patch("app.main.logger")
    def test_config_error_exits(self, mock_logger, mock_vs_cls):
        mock_vs_cls.side_effect = ConfigurationError("bad config")

        from app.main import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        mock_logger.error.assert_called()

    @patch("app.main.NewsSearchAgent")
    @patch("app.main.VectorStore")
    @patch("app.main.os.path.isdir", return_value=True)
    @patch("app.main.logger")
    def test_load_existing_vectorstore(self, mock_logger, mock_isdir, mock_vs_cls, mock_agent_cls):
        mock_db_service = MagicMock()
        mock_vs_cls.return_value = mock_db_service
        mock_vectorstore = MagicMock()
        mock_db_service.load_existing_db.return_value = mock_vectorstore

        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent

        from app.main import main

        with patch("builtins.input", side_effect=["exit"]):
            main()

        mock_db_service.load_existing_db.assert_called_once()

    @patch("app.main.NewsSearchAgent")
    @patch("app.main.NewsChunkingService")
    @patch("app.main.ArticleReaderService")
    @patch("app.main.VectorStore")
    @patch("app.main.os.path.isdir", return_value=False)
    @patch("app.main.logger")
    def test_create_new_vectorstore(
        self, mock_logger, mock_isdir, mock_vs_cls, mock_reader_cls, mock_chunk_cls, mock_agent_cls
    ):
        mock_db_service = MagicMock()
        mock_vs_cls.return_value = mock_db_service

        mock_reader = MagicMock()
        mock_reader.load_articles.return_value = [MagicMock()]
        mock_reader_cls.return_value = mock_reader

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [MagicMock()]
        mock_chunk_cls.return_value = mock_chunker

        mock_vectorstore = MagicMock()
        mock_db_service.create_or_load_faiss_db.return_value = mock_vectorstore

        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent

        from app.main import main

        with patch("builtins.input", side_effect=["exit"]):
            main()

        mock_db_service.create_or_load_faiss_db.assert_called_once()

    @patch("app.main.VectorStore")
    @patch("app.main.os.path.isdir", return_value=True)
    @patch("app.main.logger")
    def test_setup_error_exits(self, mock_logger, mock_isdir, mock_vs_cls):
        mock_db_service = MagicMock()
        mock_vs_cls.return_value = mock_db_service
        mock_db_service.load_existing_db.side_effect = VectorStoreError("load fail")

        from app.main import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch("app.main.NewsSearchAgent")
    @patch("app.main.VectorStore")
    @patch("app.main.os.path.isdir", return_value=True)
    @patch("app.main.logger")
    def test_agent_init_error_exits(self, mock_logger, mock_isdir, mock_vs_cls, mock_agent_cls):
        mock_db_service = MagicMock()
        mock_vs_cls.return_value = mock_db_service
        mock_db_service.load_existing_db.return_value = MagicMock()
        mock_agent_cls.side_effect = ModelError("model fail")

        from app.main import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch("app.main.NewsSearchAgent")
    @patch("app.main.VectorStore")
    @patch("app.main.os.path.isdir", return_value=True)
    @patch("app.main.logger")
    def test_eof_error_exits_gracefully(self, mock_logger, mock_isdir, mock_vs_cls, mock_agent_cls):
        mock_db_service = MagicMock()
        mock_vs_cls.return_value = mock_db_service
        mock_db_service.load_existing_db.return_value = MagicMock()
        mock_agent_cls.return_value = MagicMock()

        from app.main import main

        with patch("builtins.input", side_effect=EOFError):
            main()

        mock_logger.info.assert_any_call("Exiting.")

    @patch("app.main.NewsSearchAgent")
    @patch("app.main.VectorStore")
    @patch("app.main.os.path.isdir", return_value=True)
    @patch("app.main.logger")
    def test_keyboard_interrupt_on_input(self, mock_logger, mock_isdir, mock_vs_cls, mock_agent_cls):
        mock_db_service = MagicMock()
        mock_vs_cls.return_value = mock_db_service
        mock_db_service.load_existing_db.return_value = MagicMock()
        mock_agent_cls.return_value = MagicMock()

        from app.main import main

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            main()

        mock_logger.info.assert_any_call("Exiting.")

    @patch("app.main.NewsSearchAgent")
    @patch("app.main.VectorStore")
    @patch("app.main.os.path.isdir", return_value=True)
    @patch("app.main.logger")
    def test_validation_error_in_query(self, mock_logger, mock_isdir, mock_vs_cls, mock_agent_cls):
        mock_db_service = MagicMock()
        mock_vs_cls.return_value = mock_db_service
        mock_db_service.load_existing_db.return_value = MagicMock()

        mock_agent = MagicMock()
        mock_agent.ask.side_effect = ValidationError("too short")
        mock_agent_cls.return_value = mock_agent

        from app.main import main

        with patch("builtins.input", side_effect=["a valid query here", "exit"]):
            main()

        mock_logger.warning.assert_called()

    @patch("app.main.NewsSearchAgent")
    @patch("app.main.VectorStore")
    @patch("app.main.os.path.isdir", return_value=True)
    @patch("app.main.logger")
    def test_model_error_in_query(self, mock_logger, mock_isdir, mock_vs_cls, mock_agent_cls):
        mock_db_service = MagicMock()
        mock_vs_cls.return_value = mock_db_service
        mock_db_service.load_existing_db.return_value = MagicMock()

        mock_agent = MagicMock()
        mock_agent.ask.side_effect = ModelError("llm fail")
        mock_agent_cls.return_value = mock_agent

        from app.main import main

        with patch("builtins.input", side_effect=["a valid query here", "exit"]):
            main()

        mock_logger.error.assert_called()

    @patch("app.main.NewsSearchAgent")
    @patch("app.main.VectorStore")
    @patch("app.main.os.path.isdir", return_value=True)
    @patch("app.main.logger")
    def test_news_search_error_in_query(self, mock_logger, mock_isdir, mock_vs_cls, mock_agent_cls):
        mock_db_service = MagicMock()
        mock_vs_cls.return_value = mock_db_service
        mock_db_service.load_existing_db.return_value = MagicMock()

        mock_agent = MagicMock()
        mock_agent.ask.side_effect = NewsSearchError("generic error")
        mock_agent_cls.return_value = mock_agent

        from app.main import main

        with patch("builtins.input", side_effect=["a valid query here", "exit"]):
            main()

        mock_logger.error.assert_called()

    @patch("app.main.NewsSearchAgent")
    @patch("app.main.VectorStore")
    @patch("app.main.os.path.isdir", return_value=True)
    @patch("app.main.logger")
    def test_keyboard_interrupt_during_query(self, mock_logger, mock_isdir, mock_vs_cls, mock_agent_cls):
        mock_db_service = MagicMock()
        mock_vs_cls.return_value = mock_db_service
        mock_db_service.load_existing_db.return_value = MagicMock()

        mock_agent = MagicMock()
        mock_agent.ask.side_effect = KeyboardInterrupt
        mock_agent_cls.return_value = mock_agent

        from app.main import main

        with patch("builtins.input", side_effect=["a valid query here"]):
            main()

        mock_logger.info.assert_any_call("Interrupted by user.")

    @patch("app.main.NewsSearchAgent")
    @patch("app.main.VectorStore")
    @patch("app.main.os.path.isdir", return_value=True)
    @patch("app.main.logger")
    @patch("builtins.print")
    def test_successful_query(self, mock_print, mock_logger, mock_isdir, mock_vs_cls, mock_agent_cls):
        mock_db_service = MagicMock()
        mock_vs_cls.return_value = mock_db_service
        mock_db_service.load_existing_db.return_value = MagicMock()

        mock_agent = MagicMock()
        mock_agent.ask.return_value = "This is the answer."
        mock_agent_cls.return_value = mock_agent

        from app.main import main

        with patch("builtins.input", side_effect=["a valid query here", "exit"]):
            main()

        mock_print.assert_any_call("\nAnswer:\nThis is the answer.")

    @patch("app.main.VectorStore")
    @patch("app.main.ArticleReaderService")
    @patch("app.main.os.path.isdir", return_value=False)
    @patch("app.main.logger")
    def test_ingestion_error_exits(self, mock_logger, mock_isdir, mock_vs_cls, mock_reader_cls):
        mock_db_service = MagicMock()
        mock_vs_cls.return_value = mock_db_service
        mock_reader_cls.return_value = MagicMock(
            load_articles=MagicMock(side_effect=IngestionError("no articles"))
        )

        from app.main import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch("app.main.NewsSearchAgent")
    @patch("app.main.VectorStore")
    @patch("app.main.os.path.isdir", return_value=True)
    @patch("app.main.logger")
    def test_retrieval_error_in_query(self, mock_logger, mock_isdir, mock_vs_cls, mock_agent_cls):
        mock_db_service = MagicMock()
        mock_vs_cls.return_value = mock_db_service
        mock_db_service.load_existing_db.return_value = MagicMock()

        mock_agent = MagicMock()
        mock_agent.ask.side_effect = RetrievalError("retrieval fail")
        mock_agent_cls.return_value = mock_agent

        from app.main import main

        with patch("builtins.input", side_effect=["a valid query here", "exit"]):
            main()

        mock_logger.error.assert_called()
