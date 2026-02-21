import os
from pathlib import Path
from unittest.mock import patch

import pytest

from app.exceptions import IngestionError, ValidationError
from app.ingest import ArticleReaderService


@pytest.mark.unit
class TestArticleReaderServiceInit:

    def test_init_with_empty_data_dir(self):
        with pytest.raises(ValidationError, match="data_dir cannot be empty"):
            ArticleReaderService("")

    def test_init_with_none_data_dir(self):
        with pytest.raises(ValidationError):
            ArticleReaderService(None)

    def test_init_with_whitespace_data_dir(self):
        with pytest.raises(ValidationError, match="data_dir cannot be empty"):
            ArticleReaderService("   ")

    def test_init_with_valid_data_dir(self, temp_dir: Path):
        service = ArticleReaderService(str(temp_dir))
        assert service.data_dir == str(temp_dir)


@pytest.mark.unit
class TestReadArticle:

    def test_empty_path(self, temp_dir: Path):
        service = ArticleReaderService(str(temp_dir))
        with pytest.raises(ValidationError, match="Article path cannot be empty"):
            service.read_article("")

    def test_none_path(self, temp_dir: Path):
        service = ArticleReaderService(str(temp_dir))
        with pytest.raises(ValidationError):
            service.read_article(None)

    def test_nonexistent_file(self, temp_dir: Path):
        service = ArticleReaderService(str(temp_dir))
        fake_path = str(temp_dir / "nonexistent.txt")
        with pytest.raises(IngestionError, match="Article file not found"):
            service.read_article(fake_path)

    def test_path_is_directory(self, temp_dir: Path):
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        service = ArticleReaderService(str(temp_dir))
        with pytest.raises(IngestionError, match="Path is not a file"):
            service.read_article(str(subdir))

    def test_success(self, sample_text_file: Path):
        service = ArticleReaderService(str(sample_text_file.parent))
        content = service.read_article(str(sample_text_file))
        assert isinstance(content, str)
        assert len(content) > 0

    def test_permission_error(self, temp_dir: Path):
        file_path = temp_dir / "locked.txt"
        file_path.write_text("content", encoding="utf-8")
        service = ArticleReaderService(str(temp_dir))

        with patch("builtins.open", side_effect=PermissionError("denied")):
            with pytest.raises(IngestionError, match="permission denied"):
                service.read_article(str(file_path))

    def test_unicode_decode_error(self, temp_dir: Path):
        file_path = temp_dir / "bad_encoding.txt"
        file_path.write_bytes(b"\xff\xfe invalid utf-8")
        service = ArticleReaderService(str(temp_dir))

        with patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "bad")):
            with pytest.raises(IngestionError, match="Failed to decode file"):
                service.read_article(str(file_path))

    def test_os_error(self, temp_dir: Path):
        file_path = temp_dir / "oserr.txt"
        file_path.write_text("content", encoding="utf-8")
        service = ArticleReaderService(str(temp_dir))

        with patch("builtins.open", side_effect=OSError("disk error")):
            with pytest.raises(IngestionError, match="Failed to read file"):
                service.read_article(str(file_path))


@pytest.mark.unit
class TestLoadArticles:

    def test_nonexistent_dir(self):
        service = ArticleReaderService("/nonexistent/directory")
        with pytest.raises(IngestionError, match="Data directory does not exist"):
            service.load_articles()

    def test_path_is_file_not_dir(self, temp_dir: Path):
        file_path = temp_dir / "notadir.txt"
        file_path.write_text("content", encoding="utf-8")
        service = ArticleReaderService(str(file_path))
        with pytest.raises(IngestionError, match="Path is not a directory"):
            service.load_articles()

    def test_empty_dir(self, temp_dir: Path):
        service = ArticleReaderService(str(temp_dir))
        with pytest.raises(IngestionError, match="No valid .txt articles found"):
            service.load_articles()

    def test_success(self, test_data_dir: Path):
        service = ArticleReaderService(str(test_data_dir))
        articles = service.load_articles()
        assert len(articles) > 0
        assert all(hasattr(article, "page_content") for article in articles)
        assert all(hasattr(article, "metadata") for article in articles)

    def test_loads_all_txt_files(self, temp_dir: Path):
        data_dir = temp_dir / "test_data"
        data_dir.mkdir()

        for i in range(10):
            (data_dir / f"article{i}.txt").write_text(
                f"Article {i} content. " * 10, encoding="utf-8"
            )

        service = ArticleReaderService(str(data_dir))
        articles = service.load_articles()
        assert len(articles) == 10

    def test_skips_non_txt_files(self, temp_dir: Path):
        data_dir = temp_dir / "test_data"
        data_dir.mkdir()

        (data_dir / "article.txt").write_text("Content", encoding="utf-8")
        (data_dir / "article.pdf").write_text("PDF content", encoding="utf-8")
        (data_dir / "article.doc").write_text("DOC content", encoding="utf-8")

        service = ArticleReaderService(str(data_dir))
        articles = service.load_articles()
        assert len(articles) == 1
        assert articles[0].metadata["source"] == "article.txt"

    def test_skips_empty_files(self, temp_dir: Path):
        data_dir = temp_dir / "test_data"
        data_dir.mkdir()

        (data_dir / "empty.txt").write_text("   ", encoding="utf-8")
        (data_dir / "valid.txt").write_text("Valid content", encoding="utf-8")

        service = ArticleReaderService(str(data_dir))
        articles = service.load_articles()
        assert len(articles) == 1
        assert articles[0].metadata["source"] == "valid.txt"

    def test_permission_error_on_listdir(self, temp_dir: Path):
        service = ArticleReaderService(str(temp_dir))
        with patch("os.listdir", side_effect=PermissionError("denied")):
            with pytest.raises(IngestionError, match="permission denied"):
                service.load_articles()

    def test_os_error_on_listdir(self, temp_dir: Path):
        service = ArticleReaderService(str(temp_dir))
        with patch("os.listdir", side_effect=OSError("disk error")):
            with pytest.raises(IngestionError, match="Failed to access directory"):
                service.load_articles()

    def test_generic_exception_during_article_load(self, temp_dir: Path):
        data_dir = temp_dir / "test_data"
        data_dir.mkdir()
        (data_dir / "article.txt").write_text("Content", encoding="utf-8")

        service = ArticleReaderService(str(data_dir))
        with patch.object(service, "read_article", side_effect=RuntimeError("unexpected")):
            with pytest.raises(IngestionError, match="Failed to load article"):
                service.load_articles()
