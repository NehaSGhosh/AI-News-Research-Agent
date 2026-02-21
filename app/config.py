import os
from typing import Optional
from dotenv import load_dotenv

from app.exceptions import ConfigurationError

load_dotenv()


def _get_env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Retrieve an environment variable with optional default and required checks.

    Args:
        key (str): The environment variable key to retrieve.
        default (Optional[str]): Default value if the environment variable is not set.
        required (bool): If True, raises ConfigurationError when the variable is missing or empty.

    Returns:
        str: The value of the environment variable or the default.

    Raises:
        ConfigurationError: If the variable is required but missing or empty.
    """
    value = os.getenv(key, default)
    if required and (not value or not value.strip()):
        raise ConfigurationError(
            f"Missing required environment variable: {key}. "
            "Please set it in your .env file or environment."
        )
    return value or default or ""


OPENAI_API_KEY = _get_env("OPENAI_API_KEY", "", True)

FAISS_PERSIST_DIR = _get_env("FAISS_PERSIST_DIR", "./faiss_db")
DATA_DIR = _get_env("DATA_DIR", "./articles")

try:
    CHUNK_SIZE = int(_get_env("CHUNK_SIZE", "300"))
    CHUNK_OVERLAP = int(_get_env("CHUNK_OVERLAP", "60"))
except ValueError as e:
    raise ConfigurationError(
        f"Invalid CHUNK_SIZE or CHUNK_OVERLAP: must be integers. {e}"
    ) from e

EMBEDDING_MODEL = _get_env("EMBEDDING_MODEL", "text-embedding-3-small")