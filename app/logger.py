import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_log_config(key: str, default: str) -> str:
    """
    Retrieve a logging configuration value from environment variables.

    Args:
        key (str): The environment variable key to look up.
        default (str): Default value to use if the key is not set.

    Returns:
        str: The value of the environment variable or the default, stripped of whitespace.
    """
    return os.getenv(key, default).strip() if os.getenv(key) else default


def setup_logger(
    name: str | None = None,
    level: str | None = None,
    log_file: str | None = None,
) -> logging.Logger:
    """
    Configure and return a logger with console, error, and optional file handlers.

    Args:
        name (str | None): Optional name for the logger. Defaults to env LOG_NAME or "news_search_agent".
        level (str | None): Logging level as string ("DEBUG", "INFO", etc.). Defaults to env LOG_LEVEL or "INFO".
        log_file (str | None): Optional path to a log file. Defaults to env LOG_FILE or no file logging.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger_name = name or _get_log_config("LOG_NAME", "news_search_agent")
    log_level = level or _get_log_config("LOG_LEVEL", "INFO")
    log_file_path = log_file or _get_log_config("LOG_FILE", "")

    logger = logging.getLogger(logger_name)
    
    try:
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    except AttributeError:
        logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    if log_file_path:
        try:
            log_path = Path(log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_path}")
        except Exception as e:
            # Use basicConfig as fallback since logger might not be ready
            logging.basicConfig(level=logging.WARNING)
            logging.warning(f"Failed to set up file logging: {e}")

    return logger

# Default logger instance for the application
logger = setup_logger()
