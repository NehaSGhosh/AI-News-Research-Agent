import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List
from langchain_openai import OpenAIEmbeddings

from app.config import EMBEDDING_MODEL
from app.exceptions import ConfigurationError, VectorStoreError, ValidationError
from app.logger import logger

class VectorStore:
    def __init__(self) -> None:
        """
        Initialize the vector store service by loading the OpenAI embeddings.

        Raises:
            ConfigurationError: If embeddings fail to initialize (e.g., missing API key).
        """
        try:
            self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        except Exception as e:
            raise ConfigurationError(
                "Failed to initialize embeddings. Ensure OPENAI_API_KEY is set."
            ) from e

    def create_or_load_faiss_db(self, documents: List[Document], db_path: str):
        """
        Create a FAISS vectorstore from a list of documents and save it locally.

        Args:
            documents (List[Document]): List of chunked Document objects.
            db_path (str): Local path to save the FAISS index.

        Returns:
            FAISS: The created FAISS vectorstore instance.

        Raises:
            ValidationError: If documents list is empty or db_path is invalid.
            VectorStoreError: If FAISS index creation or saving fails.
        """
        if not documents:
            raise ValidationError("No text chunks provided to build FAISS index.")
        if not db_path or not str(db_path).strip():
            raise ValidationError("Database path cannot be empty")

        try:
            db = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings,
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to create FAISS index from documents: {e}"
            ) from e

        try:
            os.makedirs(db_path, exist_ok=True)
            db.save_local(db_path)
            logger.info(f"FAISS index created and saved at {db_path}")
        except PermissionError as e:
            raise VectorStoreError(
                f"Cannot write to database path (permission denied): {db_path}"
            ) from e
        except OSError as e:
            raise VectorStoreError(f"Failed to save FAISS index: {e}") from e

        return db

    def load_existing_db(self, db_path: str):
        """
        Load an existing FAISS vectorstore from a local path.

        Args:
            db_path (str): Path where FAISS index is stored.

        Returns:
            FAISS: The loaded FAISS vectorstore instance.

        Raises:
            ValidationError: If db_path is empty.
            VectorStoreError: If FAISS index does not exist or loading fails.
        """
        if not db_path or not str(db_path).strip():
            raise ValidationError("Database path cannot be empty")
        if not os.path.exists(db_path):
            raise VectorStoreError(f"FAISS index not found at {db_path}")

        try:
            db = FAISS.load_local(
                folder_path=db_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"FAISS index loaded from {db_path}")
        except Exception as e:
            raise VectorStoreError(
                f"Failed to load FAISS index from {db_path}: {e}"
            ) from e

        return db