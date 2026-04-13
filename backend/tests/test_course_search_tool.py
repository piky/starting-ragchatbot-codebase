"""
Tests for CourseSearchTool (search_tools.py).
"""
import pytest
from unittest.mock import MagicMock, patch
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Fixtures (local to this module)
# ---------------------------------------------------------------------------

@pytest.fixture
def tool():
    """CourseSearchTool with a fully-mocked VectorStore."""
    mock_store = MagicMock()
    return CourseSearchTool(mock_store), mock_store


# ---------------------------------------------------------------------------
# get_tool_definition()
# ---------------------------------------------------------------------------

class TestGetToolDefinition:
    def test_returns_dict_with_required_keys(self, tool):
        tool_instance, _ = tool
        definition = tool_instance.get_tool_definition()

        assert isinstance(definition, dict)
        assert "name" in definition
        assert "description" in definition
        assert "input_schema" in definition

    def test_name_is_search_course_content(self, tool):
        tool_instance, _ = tool
        definition = tool_instance.get_tool_definition()
        assert definition["name"] == "search_course_content"

    def test_input_schema_has_query_property(self, tool):
        tool_instance, _ = tool
        definition = tool_instance.get_tool_definition()
        schema = definition["input_schema"]

        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"

    def test_input_schema_has_optional_course_name(self, tool):
        tool_instance, _ = tool
        definition = tool_instance.get_tool_definition()
        schema = definition["input_schema"]

        assert "course_name" in schema["properties"]
        assert schema["properties"]["course_name"]["type"] == "string"

    def test_input_schema_has_optional_lesson_number(self, tool):
        tool_instance, _ = tool
        definition = tool_instance.get_tool_definition()
        schema = definition["input_schema"]

        assert "lesson_number" in schema["properties"]
        assert schema["properties"]["lesson_number"]["type"] == "integer"

    def test_query_is_required(self, tool):
        tool_instance, _ = tool
        definition = tool_instance.get_tool_definition()
        assert "query" in definition["input_schema"]["required"]


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------

class TestExecute:
    def test_calls_store_search_with_query_only(self, tool):
        tool_instance, mock_store = tool
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        tool_instance.execute(query="What are Python functions?")

        mock_store.search.assert_called_once_with(
            query="What are Python functions?",
            course_name=None,
            lesson_number=None,
        )

    def test_calls_store_search_with_all_filters(self, tool):
        tool_instance, mock_store = tool
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        tool_instance.execute(
            query="What are Python functions?",
            course_name="Python Basics",
            lesson_number=2,
        )

        mock_store.search.assert_called_once_with(
            query="What are Python functions?",
            course_name="Python Basics",
            lesson_number=2,
        )

    def test_returns_error_string_when_store_returns_error(self, tool):
        tool_instance, mock_store = tool
        mock_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Search error: ChromaDB unavailable",
        )

        result = tool_instance.execute(query="anything")

        assert "Search error" in result
        assert "ChromaDB unavailable" in result

    def test_returns_no_results_message_when_empty(self, tool):
        tool_instance, mock_store = tool
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = tool_instance.execute(query="unicorn rainbow")

        assert "No relevant content found" in result

    def test_returns_no_results_with_course_filter_info(self, tool):
        tool_instance, mock_store = tool
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = tool_instance.execute(
            query="functions",
            course_name="Python Basics",
        )

        assert "No relevant content found" in result
        assert "Python Basics" in result

    def test_returns_formatted_results_when_documents_found(self, tool):
        tool_instance, mock_store = tool
        mock_store.search.return_value = SearchResults(
            documents=[
                "Python functions are defined with the def keyword.",
                "Functions can accept parameters and return values.",
            ],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 2, "chunk_index": 0},
                {"course_title": "Python Basics", "lesson_number": 2, "chunk_index": 1},
            ],
            distances=[0.1, 0.2],
        )
        mock_store.get_lesson_link.return_value = None

        result = tool_instance.execute(query="Python functions")

        assert "Python Basics" in result
        assert "def keyword" in result
        assert "parameters" in result

    def test_includes_lesson_context_in_formatted_output(self, tool):
        tool_instance, mock_store = tool
        mock_store.search.return_value = SearchResults(
            documents=["Functions are reusable blocks of code."],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 3, "chunk_index": 0},
            ],
            distances=[0.1],
        )
        mock_store.get_lesson_link.return_value = (
            "https://example.com/python/lesson3"
        )

        result = tool_instance.execute(query="functions")

        assert "Lesson 3" in result
        assert "example.com" in result

    def test_tracks_sources_after_execution(self, tool):
        tool_instance, mock_store = tool
        mock_store.search.return_value = SearchResults(
            documents=["Chunk content here."],
            metadata=[
                {"course_title": "JS Course", "lesson_number": 1, "chunk_index": 0},
            ],
            distances=[0.1],
        )
        mock_store.get_lesson_link.return_value = None

        tool_instance.execute(query="js")

        assert len(tool_instance.last_sources) == 1
        assert tool_instance.last_sources[0]["text"] == "JS Course - Lesson 1"

    def test_sources_include_url_when_available(self, tool):
        tool_instance, mock_store = tool
        mock_store.search.return_value = SearchResults(
            documents=["Chunk content."],
            metadata=[
                {"course_title": "Go Course", "lesson_number": 4, "chunk_index": 0},
            ],
            distances=[0.1],
        )
        mock_store.get_lesson_link.return_value = "https://example.com/go/4"

        tool_instance.execute(query="go")

        assert tool_instance.last_sources[0]["url"] == "https://example.com/go/4"

    def test_handles_store_exception(self, tool):
        tool_instance, mock_store = tool
        mock_store.search.side_effect = RuntimeError("Unexpected failure")

        result = tool_instance.execute(query="anything")

        # Should return the error as a string
        assert "Unexpected failure" in result

    def test_passes_lesson_number_to_get_lesson_link(self, tool):
        tool_instance, mock_store = tool
        mock_store.search.return_value = SearchResults(
            documents=["Content."],
            metadata=[
                {"course_title": "Course", "lesson_number": 7, "chunk_index": 0},
            ],
            distances=[0.1],
        )
        mock_store.get_lesson_link.return_value = None

        tool_instance.execute(query="x", lesson_number=7)

        mock_store.get_lesson_link.assert_called_once_with("Course", 7)

    def test_passes_none_lesson_to_get_lesson_link_when_no_lesson(self, tool):
        tool_instance, mock_store = tool
        mock_store.search.return_value = SearchResults(
            documents=["Content."],
            metadata=[
                {"course_title": "Course", "lesson_number": None, "chunk_index": 0},
            ],
            distances=[0.1],
        )
        mock_store.get_lesson_link.return_value = None

        tool_instance.execute(query="x")

        mock_store.get_lesson_link.assert_not_called()


# ---------------------------------------------------------------------------
# ToolManager + CourseSearchTool integration
# ---------------------------------------------------------------------------

class TestToolManagerWithCourseSearchTool:
    def test_register_and_execute_tool(self, tool):
        tool_instance, mock_store = tool
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        tm = ToolManager()
        tm.register_tool(tool_instance)

        definitions = tm.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_unknown_tool_returns_error(self):
        tm = ToolManager()
        result = tm.execute_tool("nonexistent_tool", query="test")
        assert "not found" in result

    def test_get_last_sources_returns_sources_from_tool(self, tool):
        tool_instance, mock_store = tool
        mock_store.search.return_value = SearchResults(
            documents=["doc"],
            metadata=[{"course_title": "Test", "lesson_number": 1, "chunk_index": 0}],
            distances=[0.1],
        )
        mock_store.get_lesson_link.return_value = None

        tm = ToolManager()
        tm.register_tool(tool_instance)

        tm.execute_tool("search_course_content", query="test")
        sources = tm.get_last_sources()

        assert len(sources) == 1

    def test_reset_sources_clears_all(self, tool):
        tool_instance, mock_store = tool
        mock_store.search.return_value = SearchResults(
            documents=["doc"],
            metadata=[{"course_title": "Test", "lesson_number": 1, "chunk_index": 0}],
            distances=[0.1],
        )
        mock_store.get_lesson_link.return_value = None

        tm = ToolManager()
        tm.register_tool(tool_instance)

        tm.execute_tool("search_course_content", query="test")
        assert len(tm.get_last_sources()) == 1

        tm.reset_sources()
        assert tm.get_last_sources() == []

    def test_get_last_sources_returns_empty_when_no_search(self):
        tm = ToolManager()
        assert tm.get_last_sources() == []
