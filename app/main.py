import os
import sys

# Allow multiple instances of OpenMP to coexist without crashing
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from app.news_agent import NewsSearchAgent
from app.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, FAISS_PERSIST_DIR
from app.exceptions import (
    ConfigurationError,
    IngestionError,
    ModelError,
    NewsSearchError,
    RetrievalError,
    ValidationError,
    VectorStoreError,
)
from app.ingest import ArticleReaderService
from app.indexing.chunker import NewsChunkingService
from app.indexing.vector_store import VectorStore
from app.logger import logger


def main():
    """
    Entry point for the RAG News QA system.

    Workflow:
    1. Load or create the FAISS vectorstore with article embeddings.
    2. Initialize the NewsSearchAgent with the vectorstore.
    3. Enter a loop allowing the user to ask news-related questions.
    4. Handle graceful shutdown and exceptions.
    """ 
    try:
        db_service = VectorStore()
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    # Load existing vectorstore or create a new one from articles
    try:
        if os.path.isdir(FAISS_PERSIST_DIR):
            logger.info(f"Loading existing vector store from {FAISS_PERSIST_DIR}")
            vectorstore = db_service.load_existing_db(FAISS_PERSIST_DIR)
        else:
            logger.info(f"Creating new vector store. Reading articles from {DATA_DIR}")
            reader_service = ArticleReaderService(DATA_DIR)
            articles = reader_service.load_articles()

            chunking_service = NewsChunkingService(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            chunks = chunking_service.chunk(articles)
            logger.info(f"Total chunks created: {len(chunks)}")

            vectorstore = db_service.create_or_load_faiss_db(
                chunks, FAISS_PERSIST_DIR
            )
            logger.info(f"Vector store created and saved to {FAISS_PERSIST_DIR}")
    except (IngestionError, VectorStoreError, ValidationError) as e:
        logger.error(f"Setup error: {e}")
        sys.exit(1)

    # Initialize the NewsSearchAgent with the vectorstore.
    try:
        qa_agent = NewsSearchAgent(vectorstore)
    except (ConfigurationError, ModelError, RetrievalError, ValidationError) as e:
        logger.error(f"Failed to initialize agent: {e}")
        sys.exit(1)

    logger.info("RAG News QA agent is ready.")
    
    # Interactive loop for user queries
    while True:
        try:
            query = input("\nAsk a news question (or 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("Exiting.")
            break

        if query.lower() == "exit":
            logger.info("User requested exit.")
            break

        try:
            logger.debug(f"Processing query: {query}")
            answer = qa_agent.ask(query)
            print(f"\nAnswer:\n{answer}")
            logger.debug("Query processed successfully")
        except ValidationError as e:
            logger.warning(f"Invalid input: {e}")
        except (ModelError, RetrievalError) as e:
            logger.error(f"Error generating response: {e}")
        except NewsSearchError as e:
            logger.error(f"Error: {e}")
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            break


if __name__ == "__main__":
    main()
