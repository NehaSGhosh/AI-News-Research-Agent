import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.exceptions import IngestionError, ValidationError


class ArticleReaderService:
    # Service class to handle reading and loading articles from a directory.

    def __init__(self, data_dir: str):
        """
        Initialize the service with the path to the data directory.

        Args:
            data_dir (str): Path to the directory containing article files.

        Raises:
            ValidationError: If the provided directory path is empty.
        """
        if not data_dir or not str(data_dir).strip():
            raise ValidationError("data_dir cannot be empty")
        self.data_dir = data_dir

    def read_article(self, path: str) -> str:
        """
        Read a single article file and return its contents as a string.

        Args:
            path (str): Full path to the article file.

        Returns:
            str: Content of the article.

        Raises:
            ValidationError: If the path is empty.
            IngestionError: If the file does not exist, is not a file,
                            cannot be read, or fails decoding.
        """
        if not path or not str(path).strip():
            raise ValidationError("Article path cannot be empty")
        if not os.path.exists(path):
            raise IngestionError(f"Article file not found: {path}")
        if not os.path.isfile(path):
            raise IngestionError(f"Path is not a file: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except PermissionError as e:
            raise IngestionError(f"Cannot read file (permission denied): {path}") from e
        except UnicodeDecodeError as e:
            raise IngestionError(
                f"Failed to decode file (invalid encoding): {path}. "
                "Ensure the file is UTF-8 encoded."
            ) from e
        except OSError as e:
            raise IngestionError(f"Failed to read file: {path}") from e

    def load_articles(self) -> List[Document]:
        """
        Load all articles (.txt) from the data directory and return
        them as a list of Document objects.

        Returns:
            List[Document]: List of Document objects with content and metadata.

        Raises:
            IngestionError: If the directory does not exist, is not accessible,
                            or contains no valid .txt files.
        """
        dir_path = Path(self.data_dir)
        if not dir_path.exists():
            raise IngestionError(f"Data directory does not exist: {self.data_dir}")
        if not dir_path.is_dir():
            raise IngestionError(f"Path is not a directory: {self.data_dir}")

        articles: List[Document] = []
        try:
            files = os.listdir(self.data_dir)
        except PermissionError as e:
            raise IngestionError(
                f"Cannot list directory (permission denied): {self.data_dir}"
            ) from e
        except OSError as e:
            raise IngestionError(f"Failed to access directory: {self.data_dir}") from e

        for file in files:
            path = os.path.join(self.data_dir, file)
            if file.endswith(".txt"):
                try:
                    text = self.read_article(path)
                    content = text.strip()
                    if content:
                        articles.append(
                            Document(
                                page_content=content,
                                metadata={"source": file},
                            )
                        )
                except IngestionError:
                    raise
                except Exception as e:
                    raise IngestionError(f"Failed to load article {file}: {e}") from e

        if not articles:
            raise IngestionError(
                f"No valid .txt articles found in {self.data_dir}. "
                "Add .txt files to the data directory."
            )
        return articles