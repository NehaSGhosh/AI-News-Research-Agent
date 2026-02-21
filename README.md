# AI News Research Agent

**Field:** Media / Information Retrieval

This project is a Retrieval-Augmented Generation (RAG) news search and question-answering agent. It ingests local news articles, chunks and indexes them into a vector store, and uses an LLM to **dynamically control the query-to-response flow**, including deciding whether to request more context, ask for clarification, or provide the final answer.

Note: The dataset used for this project contains the first 1001 files from [news-articles-corpus](https://www.kaggle.com/datasets/sbhatti/news-articles-corpus)

## Features

- Ingests `.txt` news articles from a data directory
- Cleans and chunks documents into semantic pieces
- Builds and persists a FAISS vector store with OpenAI embeddings
- Uses an LLM to **orchestrate the query handling**, deciding when to retrieve more information or request clarification
- Supports **multi-step retrieval**, where the number of top documents retrieved (`k`) can increase dynamically if the LLM requests more context
- Production-grade error handling and logging
- Comprehensive unit and integration tests

## Prerequisites

- Python 3.11+
- OpenAI API key

## Setup

1. **Create and activate a virtual environment**:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Configure environment variables**:

- Copy `.env.example` to `.env`
- Add your OpenAI API key

## Running the Application

From the project root:

```bash
python -m app.main
```

You will be prompted to ask news questions in the terminal. Type `exit` to quit.

## Logging

Logging is configured via `app/logger.py` and environment variables:

- `LOG_LEVEL` – `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default: `INFO`)
- `LOG_FILE` – optional path to a log file; if empty, logs are console-only
- `LOG_NAME` – logger name (default: `news_search_agent`)

## Testing

This project includes a full unit and integration test suite.

### Running Tests

#### Run all tests

```bash
pytest
```

#### Run only unit tests

```bash
pytest -m unit
```

#### Run only integration tests

```bash
pytest -m integration
```

#### Run with coverage report

```bash
pytest --cov=app --cov-report=html
```

#### Run specific test file

```bash
pytest tests/test_ingest.py
```

#### Run specific test

```bash
pytest tests/test_ingest.py::TestArticleReaderService::test_read_article_success
```

### Environment Variables for Tests

Some integration tests require API keys. Set these in your environment or `.env` file:

- `OPENAI_API_KEY` – required for vector store and agent tests

Tests marked with `@pytest.mark.requires_api` will be skipped if API keys are not available.

## Sample Input/Output

```
Ask a news question (or 'exit' to quit): Summarize a crime.
2026-02-21 01:55:31 - news_search_agent - INFO - LLM indicated more context is needed. Increasing k from 2 to 4 and retrieving additional documents.
2026-02-21 01:55:31 - news_search_agent - INFO - LLM indicated more context is needed. Increasing k from 4 to 8 and retrieving additional documents.

Answer:
The crime involved the murder of a child named Elizabeth, who was strangled and stabbed by Bustamante, described as a "truly evil individual" by the prosecution.

Ask a news question (or 'exit' to quit): Who is Merkel?

Answer:
Angela Merkel is the German Chancellor.

Ask a news question (or 'exit' to quit): Tell about the effect of inlation.     
2026-02-21 01:57:40 - news_search_agent - INFO - LLM indicated more context is needed. Increasing k from 2 to 4 and retrieving additional documents.

Answer:
The effect of inflation can be significant on various aspects of the economy. It can lead to increased prices for essential goods such as eggs, poultry, and milk, as noted in the context. For instance, the production cycle for eggs and poultry is shorter, meaning that price changes can occur more rapidly in these markets. Additionally, inflation can impact farmers, as harsh weather conditions can affect crop yields, leading to delays in harvests and potential price hikes for consumers.     

Moreover, inflation can disproportionately affect lower-income families, especially those with children under five, as tax and benefit reforms may not adequately adjust to rising costs. This can lead to increased financial strain on these households.

In summary, inflation can lead to higher prices for goods, affect agricultural production, and create disparities in economic impact across different household types.

Ask a news question (or 'exit' to quit): Who will win the next FIFA world cup?
2026-02-21 01:58:37 - news_search_agent - INFO - LLM indicated more context is needed. Increasing k from 2 to 4 and retrieving additional documents.
2026-02-21 01:58:38 - news_search_agent - INFO - LLM indicated more context is needed. Increasing k from 4 to 8 and retrieving additional documents.
2026-02-21 01:58:39 - news_search_agent - INFO - LLM indicated more context is needed. Increasing k from 8 to 16 and retrieving additional documents.
2026-02-21 01:58:40 - news_search_agent - INFO - LLM indicated more context is needed. Increasing k from 16 to 32 and retrieving additional documents.

Answer:
No relevant information found.

Ask a news question (or 'exit' to quit): exit
2026-02-21 01:59:25 - news_search_agent - INFO - User requested exit.
```
