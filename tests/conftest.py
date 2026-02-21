import logging
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from dotenv import load_dotenv

from langchain_core.documents import Document

load_dotenv()

_ORIGINAL_API_KEY = os.getenv("OPENAI_API_KEY", "")


@pytest.fixture(scope="session")
def real_openai_api_key() -> str:
    if not _ORIGINAL_API_KEY or not _ORIGINAL_API_KEY.strip():
        pytest.skip("OPENAI_API_KEY not set")
    return _ORIGINAL_API_KEY


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    file_path = temp_dir / "sample.txt"
    file_path.write_text("This is a sample news article. " * 50, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="This is the first news article about technology.",
            metadata={"source": "article1.txt"},
        ),
        Document(
            page_content="This is the second news article about science.",
            metadata={"source": "article2.txt"},
        ),
        Document(
            page_content="This is the third news article about politics.",
            metadata={"source": "article3.txt"},
        ),
    ]


@pytest.fixture
def test_data_dir(temp_dir: Path) -> Path:
    data_dir = temp_dir / "test_data"
    data_dir.mkdir()

    (data_dir / "article1.txt").write_text(
        "Breaking news: Technology advances in AI. " * 20, encoding="utf-8"
    )
    (data_dir / "article2.txt").write_text(
        "Science update: New discoveries in space. " * 20, encoding="utf-8"
    )
    (data_dir / "article3.txt").write_text(
        "Politics: Election results announced. " * 20, encoding="utf-8"
    )

    return data_dir


@pytest.fixture(autouse=True)
def reset_env_vars(monkeypatch):
    original_vars = {}
    test_vars = [
        "OPENAI_API_KEY",
        "FAISS_PERSIST_DIR",
        "DATA_DIR",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "EMBEDDING_MODEL",
    ]

    for var in test_vars:
        original_vars[var] = os.environ.get(var)
        if var in os.environ:
            monkeypatch.delenv(var, raising=False)

    yield

    for var, value in original_vars.items():
        if value is not None:
            monkeypatch.setenv(var, value)
        elif var in os.environ:
            monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def cleanup_test_loggers():
    yield
    for name in list(logging.Logger.manager.loggerDict):
        if name.startswith("test_"):
            logger = logging.getLogger(name)
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
