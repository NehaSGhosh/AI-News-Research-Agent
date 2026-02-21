from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from app.news_agent import NewsSearchAgent
from app.exceptions import ModelError, RetrievalError, ValidationError


@pytest.mark.unit
class TestNewsSearchAgentInit:

    def test_init_with_none_vectorstore(self):
        with pytest.raises(ValidationError, match="vectorstore cannot be None"):
            NewsSearchAgent(None)

    @patch("app.news_agent.ChatOpenAI")
    def test_init_success(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_vs = MagicMock()

        agent = NewsSearchAgent(mock_vs)
        assert agent.vectorstore == mock_vs
        assert agent.model == mock_llm

    @patch("app.news_agent.ChatOpenAI")
    def test_init_model_load_failure(self, mock_chat_openai):
        mock_chat_openai.side_effect = Exception("API key error")
        mock_vs = MagicMock()

        with pytest.raises(ModelError, match="Failed to initialize LLM"):
            NewsSearchAgent(mock_vs)


@pytest.mark.unit
class TestLoadModel:

    @patch("app.news_agent.ChatOpenAI")
    def test_load_model_success(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_vs = MagicMock()

        agent = NewsSearchAgent(mock_vs)
        assert agent.model == mock_llm

    @patch("app.news_agent.ChatOpenAI")
    def test_load_model_failure(self, mock_chat_openai):
        mock_chat_openai.side_effect = Exception("fail")
        mock_vs = MagicMock()

        with pytest.raises(ModelError):
            NewsSearchAgent(mock_vs)


@pytest.mark.unit
class TestAsk:

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_empty_query(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        agent = NewsSearchAgent(MagicMock())
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            agent.ask("")

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_whitespace_query(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        agent = NewsSearchAgent(MagicMock())
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            agent.ask("   ")

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_short_query(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        agent = NewsSearchAgent(MagicMock())
        with pytest.raises(ValidationError, match="Query too short"):
            agent.ask("short")

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_retriever_creation_failure(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        mock_vs = MagicMock()
        mock_vs.as_retriever.side_effect = Exception("retriever fail")

        agent = NewsSearchAgent(mock_vs)
        with pytest.raises(RetrievalError, match="Failed to create retriever"):
            agent.ask("What is the latest news?")

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_retrieval_invoke_failure(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = Exception("invoke fail")
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        agent = NewsSearchAgent(mock_vs)
        with pytest.raises(RetrievalError, match="Failed to retrieve documents"):
            agent.ask("What is the latest news?")

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_no_docs_returns_fallback(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        agent = NewsSearchAgent(mock_vs)
        result = agent.ask("What is the latest news?")
        assert result == "No relevant information found."

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_invalid_doc_format(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        mock_doc = MagicMock(spec=[])
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        agent = NewsSearchAgent(mock_vs)
        with pytest.raises(RetrievalError, match="Invalid document format"):
            agent.ask("What is the latest news?")

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_prompt_format_error(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        agent = NewsSearchAgent(mock_vs)

        with patch("app.news_agent.ChatPromptTemplate") as mock_pt:
            mock_pt.from_template.side_effect = Exception("bad prompt")
            with pytest.raises(ModelError, match="Failed to format prompt"):
                agent.ask("What is the latest news?")

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_llm_invocation_failure(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API fail")
        mock_chat_openai.return_value = mock_llm
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        agent = NewsSearchAgent(mock_vs)
        with pytest.raises(ModelError, match="LLM invocation failed"):
            agent.ask("What is the latest news?")

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_result_without_content_attr(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "plain answer text"
        mock_chat_openai.return_value = mock_llm
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        agent = NewsSearchAgent(mock_vs)
        result = agent.ask("What is the latest news?")
        assert result == "plain answer text"

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_action_retrieve_more_then_answer(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_llm.invoke.side_effect = [
            MagicMock(content="ACTION: RETRIEVE_MORE"),
            MagicMock(content="Final answer here"),
        ]
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        agent = NewsSearchAgent(mock_vs)
        result = agent.ask("What is the latest news?")
        assert result == "Final answer here"

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_action_clarify(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="ACTION: CLARIFY")
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        agent = NewsSearchAgent(mock_vs)
        result = agent.ask("What is the latest news?")
        assert result == "The question appears unclear. Please clarify your request."

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_action_unknown_returns_no_info(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="ACTION: UNKNOWN_ACTION")
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        agent = NewsSearchAgent(mock_vs)
        result = agent.ask("What is the latest news?")
        assert result == "No relevant information found."

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_strips_answer_prefix(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="Answer: The real answer")
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        agent = NewsSearchAgent(mock_vs)
        result = agent.ask("What is the latest news?")
        assert result == "The real answer"

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_empty_response_returns_fallback(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="   ")
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        agent = NewsSearchAgent(mock_vs)
        result = agent.ask("What is the latest news?")
        assert result == "No relevant information found."

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_max_rounds_exhausted(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="ACTION: RETRIEVE_MORE")
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        agent = NewsSearchAgent(mock_vs)
        result = agent.ask("What is the latest news?")
        assert result == "No relevant information found."

    @patch("app.news_agent.ChatOpenAI")
    def test_ask_success_direct_answer(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="This is the answer.")
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        agent = NewsSearchAgent(mock_vs)
        result = agent.ask("What is the latest news?")
        assert result == "This is the answer."
