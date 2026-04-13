"""
Tests for RAGSystem (rag_system.py).
Verifies the end-to-end query flow including session management,
tool passing, and error propagation.
"""
import pytest
from unittest.mock import MagicMock, patch

from rag_system import RAGSystem
from search_tools import CourseSearchTool, ToolManager
from session_manager import SessionManager
from vector_store import VectorStore, SearchResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_message(content, tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    return msg


# ---------------------------------------------------------------------------
# RAGSystem initialization
# ---------------------------------------------------------------------------

class TestRAGSystemInit:
    def test_creates_all_components(self):
        with patch("rag_system.AIGenerator") as MockAIGen:
            with patch("rag_system.VectorStore") as MockVS:
                with patch("rag_system.SessionManager") as MockSM:
                    MockVS.return_value = MagicMock(spec=VectorStore)
                    MockSM.return_value = MagicMock(spec=SessionManager)

                    from config import Config
                    config = Config()

                    rag = RAGSystem(config)

                    MockAIGen.assert_called_once()
                    MockVS.assert_called_once()
                    MockSM.assert_called_once()

    def test_registers_course_search_tool(self):
        with patch("rag_system.AIGenerator") as MockAIGen:
            with patch("rag_system.VectorStore") as MockVS:
                with patch("rag_system.SessionManager") as MockSM:
                    MockVS.return_value = MagicMock(spec=VectorStore)
                    MockSM.return_value = MagicMock(spec=SessionManager)

                    from config import Config
                    config = Config()

                    rag = RAGSystem(config)

                    definitions = rag.tool_manager.get_tool_definitions()
                    tool_names = {d["name"] for d in definitions}
                    assert "search_course_content" in tool_names


# ---------------------------------------------------------------------------
# RAGSystem.query()
# ---------------------------------------------------------------------------

class TestRAGSystemQuery:
    def test_returns_response_and_sources_tuple(self):
        mock_vs = MagicMock(spec=VectorStore)
        mock_vs.search.return_value = SearchResults(
            documents=["Python content."],
            metadata=[{"course_title": "Py", "lesson_number": 1, "chunk_index": 0}],
            distances=[0.1],
        )

        mock_ai = MagicMock()
        mock_ai.generate_response.return_value = "Python is a programming language."

        with patch("rag_system.VectorStore", return_value=mock_vs):
            with patch("rag_system.AIGenerator", return_value=mock_ai):
                with patch("rag_system.SessionManager") as MockSM:
                    MockSM.return_value = MagicMock()

                    from config import Config
                    config = Config()

                    rag = RAGSystem(config)
                    rag.ai_generator = mock_ai
                    rag.vector_store = mock_vs

                    response, sources = rag.query("What is Python?")

                    assert isinstance(response, str)
                    assert response == "Python is a programming language."

    def test_passes_tool_definitions_to_ai_generator(self):
        mock_vs = MagicMock(spec=VectorStore)
        mock_ai = MagicMock()
        mock_ai.generate_response.return_value = "answer", []

        with patch("rag_system.VectorStore", return_value=mock_vs):
            with patch("rag_system.AIGenerator", return_value=mock_ai):
                with patch("rag_system.SessionManager") as MockSM:
                    MockSM.return_value = MagicMock()
                    MockSM.return_value.get_conversation_history.return_value = None

                    from config import Config
                    config = Config()

                    rag = RAGSystem(config)
                    rag.ai_generator = mock_ai
                    rag.vector_store = mock_vs

                    rag.query("What is Python?")

                    call_kwargs = mock_ai.generate_response.call_args.kwargs
                    assert "tools" in call_kwargs
                    tool_names = {t["name"] for t in call_kwargs["tools"]}
                    assert "search_course_content" in tool_names

    def test_passes_tool_manager_to_ai_generator(self):
        mock_vs = MagicMock(spec=VectorStore)
        mock_ai = MagicMock()
        mock_ai.generate_response.return_value = "answer"

        with patch("rag_system.VectorStore", return_value=mock_vs):
            with patch("rag_system.AIGenerator", return_value=mock_ai):
                with patch("rag_system.SessionManager") as MockSM:
                    MockSM.return_value = MagicMock()
                    MockSM.return_value.get_conversation_history.return_value = None

                    from config import Config
                    config = Config()

                    rag = RAGSystem(config)
                    rag.ai_generator = mock_ai
                    rag.tool_manager = ToolManager()
                    rag.tool_manager.register_tool(CourseSearchTool(mock_vs))
                    rag.vector_store = mock_vs

                    rag.query("What is Python?")

                    call_kwargs = mock_ai.generate_response.call_args.kwargs
                    assert "tool_manager" in call_kwargs
                    assert call_kwargs["tool_manager"] is not None

    def test_passes_conversation_history_when_session_exists(self):
        mock_vs = MagicMock(spec=VectorStore)
        mock_ai = MagicMock()
        mock_ai.generate_response.return_value = "second answer"

        mock_sm = MagicMock()
        mock_sm.get_conversation_history.return_value = "User: What is Python?\nAssistant: Python is great."
        mock_sm.add_exchange = MagicMock()

        with patch("rag_system.VectorStore", return_value=mock_vs):
            with patch("rag_system.AIGenerator", return_value=mock_ai):
                with patch("rag_system.SessionManager", return_value=mock_sm):
                    from config import Config
                    config = Config()

                    rag = RAGSystem(config)
                    rag.ai_generator = mock_ai
                    rag.session_manager = mock_sm
                    rag.vector_store = mock_vs

                    rag.query("What about functions?", session_id="session_1")

                    call_kwargs = mock_ai.generate_response.call_args.kwargs
                    assert "conversation_history" in call_kwargs
                    assert "What is Python" in call_kwargs["conversation_history"]

    def test_updates_conversation_history_after_query(self):
        mock_vs = MagicMock(spec=VectorStore)
        mock_ai = MagicMock()
        mock_ai.generate_response.return_value = "answer"

        mock_sm = MagicMock()
        mock_sm.get_conversation_history.return_value = None
        mock_sm.add_exchange = MagicMock()

        with patch("rag_system.VectorStore", return_value=mock_vs):
            with patch("rag_system.AIGenerator", return_value=mock_ai):
                with patch("rag_system.SessionManager", return_value=mock_sm):
                    from config import Config
                    config = Config()

                    rag = RAGSystem(config)
                    rag.ai_generator = mock_ai
                    rag.session_manager = mock_sm
                    rag.vector_store = mock_vs

                    rag.query("What is Python?", session_id="session_1")

                    mock_sm.add_exchange.assert_called_once_with(
                        "session_1",
                        "What is Python?",
                        "answer",
                    )

    def test_resets_sources_after_retrieving(self):
        mock_vs = MagicMock(spec=VectorStore)
        mock_ai = MagicMock()
        mock_ai.generate_response.return_value = "answer"

        mock_sm = MagicMock()
        mock_sm.get_conversation_history.return_value = None

        with patch("rag_system.VectorStore", return_value=mock_vs):
            with patch("rag_system.AIGenerator", return_value=mock_ai):
                with patch("rag_system.SessionManager", return_value=mock_sm):
                    from config import Config
                    config = Config()

                    rag = RAGSystem(config)
                    rag.ai_generator = mock_ai
                    rag.session_manager = mock_sm
                    rag.vector_store = mock_vs
                    rag.tool_manager = ToolManager()

                    # Ensure reset_sources is defined on tool_manager
                    rag.tool_manager.reset_sources = MagicMock()

                    rag.query("test")

                    rag.tool_manager.reset_sources.assert_called_once()

    def test_raises_runtime_error_when_ollama_fails(self):
        mock_vs = MagicMock(spec=VectorStore)
        mock_ai = MagicMock()
        mock_ai.generate_response.side_effect = RuntimeError("Ollama client error: Connection refused")

        with patch("rag_system.VectorStore", return_value=mock_vs):
            with patch("rag_system.AIGenerator", return_value=mock_ai):
                with patch("rag_system.SessionManager") as MockSM:
                    MockSM.return_value = MagicMock()

                    from config import Config
                    config = Config()

                    rag = RAGSystem(config)
                    rag.ai_generator = mock_ai
                    rag.vector_store = mock_vs

                    with pytest.raises(RuntimeError) as exc_info:
                        rag.query("test query")

                    assert "Ollama" in str(exc_info.value)

    def test_query_without_session_id_still_works(self):
        """Passing None session_id should not crash."""
        mock_vs = MagicMock(spec=VectorStore)
        mock_ai = MagicMock()
        mock_ai.generate_response.return_value = "answer"

        mock_sm = MagicMock()
        mock_sm.get_conversation_history.return_value = None

        with patch("rag_system.VectorStore", return_value=mock_vs):
            with patch("rag_system.AIGenerator", return_value=mock_ai):
                with patch("rag_system.SessionManager", return_value=mock_sm):
                    from config import Config
                    config = Config()

                    rag = RAGSystem(config)
                    rag.ai_generator = mock_ai
                    rag.session_manager = mock_sm
                    rag.vector_store = mock_vs

                    # Should not raise
                    response, sources = rag.query("What is Python?")
                    assert response == "answer"


# ---------------------------------------------------------------------------
# RAGSystem.add_course_document()
# ---------------------------------------------------------------------------

class TestRAGSystemAddCourseDocument:
    def test_returns_none_on_exception(self):
        mock_vs = MagicMock(spec=VectorStore)
        mock_ai = MagicMock()

        with patch("rag_system.VectorStore", return_value=mock_vs):
            with patch("rag_system.AIGenerator", return_value=mock_ai):
                with patch("rag_system.SessionManager") as MockSM:
                    MockSM.return_value = MagicMock()

                    from config import Config
                    config = Config()

                    rag = RAGSystem(config)
                    rag.vector_store = mock_vs

                    course, count = rag.add_course_document("/nonexistent/file.txt")
                    assert course is None
                    assert count == 0


# ---------------------------------------------------------------------------
# RAGSystem.get_course_analytics()
# ---------------------------------------------------------------------------

class TestRAGSystemGetCourseAnalytics:
    def test_returns_course_count_and_titles(self):
        mock_vs = MagicMock(spec=VectorStore)
        mock_vs.get_course_count.return_value = 3
        mock_vs.get_existing_course_titles.return_value = [
            "Course A", "Course B", "Course C"
        ]

        mock_ai = MagicMock()

        with patch("rag_system.VectorStore", return_value=mock_vs):
            with patch("rag_system.AIGenerator", return_value=mock_ai):
                with patch("rag_system.SessionManager") as MockSM:
                    MockSM.return_value = MagicMock()

                    from config import Config
                    config = Config()

                    rag = RAGSystem(config)
                    rag.vector_store = mock_vs

                    analytics = rag.get_course_analytics()

                    assert analytics["total_courses"] == 3
                    assert len(analytics["course_titles"]) == 3


# ---------------------------------------------------------------------------
# Integration: full tool execution path
# ---------------------------------------------------------------------------

class TestRAGSystemToolExecutionPath:
    def test_tool_manager_is_used_during_query(self):
        """
        Verifies that when Ollama returns tool_calls, the tool manager
        is invoked to execute them.
        """
        mock_vs = MagicMock(spec=VectorStore)
        mock_vs.search.return_value = SearchResults(
            documents=["Python functions are defined using `def`."],
            metadata=[{"course_title": "Python", "lesson_number": 1, "chunk_index": 0}],
            distances=[0.1],
        )

        tool = CourseSearchTool(mock_vs)
        tm = ToolManager()
        tm.register_tool(tool)

        mock_ai = MagicMock()
        mock_ai.generate_response.return_value = "Based on search: def keyword."

        with patch("rag_system.VectorStore", return_value=mock_vs):
            with patch("rag_system.AIGenerator", return_value=mock_ai):
                with patch("rag_system.SessionManager") as MockSM:
                    MockSM.return_value = MagicMock()

                    from config import Config
                    config = Config()

                    rag = RAGSystem(config)
                    rag.ai_generator = mock_ai
                    rag.session_manager = MockSM.return_value
                    rag.tool_manager = tm
                    rag.vector_store = mock_vs

                    response, sources = rag.query("What is def in Python?")

                    # The AI generator's tool_manager should have been passed
                    call_kwargs = mock_ai.generate_response.call_args.kwargs
                    assert call_kwargs["tool_manager"] is tm

    def test_ollama_unreachable_propagates_as_503(self):
        """
        When Ollama is unreachable, AIGenerator raises RuntimeError.
        The app.py endpoint catches this and returns HTTP 503.

        This test verifies the RuntimeError propagation chain.
        """
        mock_vs = MagicMock(spec=VectorStore)
        mock_ai = MagicMock()
        mock_ai.generate_response.side_effect = RuntimeError(
            "Ollama client error: Connection refused"
        )

        with patch("rag_system.VectorStore", return_value=mock_vs):
            with patch("rag_system.AIGenerator", return_value=mock_ai):
                with patch("rag_system.SessionManager") as MockSM:
                    MockSM.return_value = MagicMock()

                    from config import Config
                    config = Config()

                    rag = RAGSystem(config)
                    rag.ai_generator = mock_ai
                    rag.vector_store = mock_vs

                    with pytest.raises(RuntimeError) as exc_info:
                        rag.query("What is Python?")

                    assert "Connection refused" in str(exc_info.value)
                    # This is what app.py catches to return 503
                    assert "Ollama" in str(exc_info.value)