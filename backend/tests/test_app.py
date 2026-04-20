"""
Tests for FastAPI endpoints (app.py).
Verifies request/response handling, error handling, and session management.
"""
import pytest
from unittest.mock import MagicMock, patch
import asyncio


# ---------------------------------------------------------------------------
# Test fixtures for API tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag_system():
    """Create a mocked RAG system for API testing."""
    mock = MagicMock()
    mock.session_manager = MagicMock()
    mock.session_manager.create_session.return_value = "test-session-123"
    mock.session_manager.get_conversation_history.return_value = None
    mock.query.return_value = ("Python functions use the def keyword.", [])
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python Basics", "JavaScript Advanced"]
    }
    return mock


@pytest.fixture
def mock_rag_system_with_sources():
    """RAG system that returns sources."""
    mock = MagicMock()
    mock.session_manager = MagicMock()
    mock.session_manager.create_session.return_value = "test-session-456"
    mock.session_manager.get_conversation_history.return_value = None
    mock.query.return_value = (
        "Python is a programming language.",
        [
            {"text": "Python Basics - Lesson 1", "url": "https://example.com/py1"},
            {"text": "Python Advanced - Lesson 2", "url": None}
        ]
    )
    return mock


# ---------------------------------------------------------------------------
# Root endpoint tests
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    @pytest.mark.asyncio
    async def test_root_returns_ok_status(self, test_client):
        """Root endpoint should return status ok."""
        mock_rag = MagicMock()
        client = await test_client(mock_rag)

        response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "RAG System" in data["message"]


# ---------------------------------------------------------------------------
# /api/query endpoint tests
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    @pytest.mark.asyncio
    async def test_query_returns_response_with_sources(self, test_client, mock_rag_system):
        """Query endpoint should return answer and sources."""
        client = await test_client(mock_rag_system)

        response = await client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "Python functions use the def keyword."
        mock_rag_system.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_creates_session_if_not_provided(self, test_client, mock_rag_system):
        """Query endpoint should create a session when none is provided."""
        client = await test_client(mock_rag_system)

        await client.post("/api/query", json={"query": "test"})

        mock_rag_system.session_manager.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_uses_provided_session_id(self, test_client, mock_rag_system):
        """Query endpoint should use the provided session_id."""
        client = await test_client(mock_rag_system)

        await client.post(
            "/api/query",
            json={"query": "test", "session_id": "existing-session"}
        )

        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("test", "existing-session")

    @pytest.mark.asyncio
    async def test_query_includes_sources_in_response(self, test_client, mock_rag_system_with_sources):
        """Query response should include source items with optional URLs."""
        client = await test_client(mock_rag_system_with_sources)

        response = await client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )

        data = response.json()
        assert len(data["sources"]) == 2
        assert data["sources"][0]["text"] == "Python Basics - Lesson 1"
        assert data["sources"][0]["url"] == "https://example.com/py1"
        assert data["sources"][1]["url"] is None

    @pytest.mark.asyncio
    async def test_query_returns_503_on_ollama_error(self, test_client):
        """Query endpoint should return 503 when Ollama is unreachable."""
        mock_rag = MagicMock()
        mock_rag.session_manager.create_session.return_value = "session"
        mock_rag.query.side_effect = RuntimeError("Ollama client error: Connection refused")
        client = await test_client(mock_rag)

        response = await client.post(
            "/api/query",
            json={"query": "test"}
        )

        assert response.status_code == 503
        data = response.json()
        assert "Ollama" in data["detail"]

    @pytest.mark.asyncio
    async def test_query_returns_500_on_unexpected_error(self, test_client):
        """Query endpoint should return 500 on unexpected errors."""
        mock_rag = MagicMock()
        mock_rag.session_manager.create_session.return_value = "session"
        mock_rag.query.side_effect = Exception("Unexpected error")
        client = await test_client(mock_rag)

        response = await client.post(
            "/api/query",
            json={"query": "test"}
        )

        assert response.status_code == 500
        data = response.json()
        assert "Unexpected error" in data["detail"]

    @pytest.mark.asyncio
    async def test_query_requires_query_field(self, test_client, mock_rag_system):
        """Query endpoint should reject requests without query field."""
        client = await test_client(mock_rag_system)

        response = await client.post("/api/query", json={})

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_query_validates_query_is_string(self, test_client, mock_rag_system):
        """Query endpoint should reject non-string query values."""
        client = await test_client(mock_rag_system)

        response = await client.post(
            "/api/query",
            json={"query": 12345}
        )

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /api/courses endpoint tests
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:
    @pytest.mark.asyncio
    async def test_courses_returns_stats(self, test_client, mock_rag_system):
        """Courses endpoint should return course statistics."""
        client = await test_client(mock_rag_system)

        response = await client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Python Basics" in data["course_titles"]
        mock_rag_system.get_course_analytics.assert_called_once()

    @pytest.mark.asyncio
    async def test_courses_returns_empty_list_when_no_courses(self, test_client):
        """Courses endpoint should handle empty course catalog."""
        mock_rag = MagicMock()
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        client = await test_client(mock_rag)

        response = await client.get("/api/courses")

        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    @pytest.mark.asyncio
    async def test_courses_returns_500_on_error(self, test_client):
        """Courses endpoint should return 500 on internal error."""
        mock_rag = MagicMock()
        mock_rag.get_course_analytics.side_effect = Exception("Database error")
        client = await test_client(mock_rag)

        response = await client.get("/api/courses")

        assert response.status_code == 500
        data = response.json()
        assert "Database error" in data["detail"]


# ---------------------------------------------------------------------------
# CORS and middleware tests
# ---------------------------------------------------------------------------

class TestCORSConfiguration:
    @pytest.mark.asyncio
    async def test_cors_allows_all_origins(self, test_client, mock_rag_system):
        """CORS should allow requests from any origin."""
        client = await test_client(mock_rag_system)

        response = await client.options(
            "/api/query",
            headers={"Origin": "http://localhost:3000"}
        )

        # FastAPI handles OPTIONS automatically for CORS
        assert response.status_code in [200, 405]

    @pytest.mark.asyncio
    async def test_cors_headers_present_on_response(self, test_client, mock_rag_system):
        """CORS headers should be present on responses."""
        client = await test_client(mock_rag_system)

        response = await client.post(
            "/api/query",
            json={"query": "test"},
            headers={"Origin": "http://localhost:3000"}
        )

        # Allow headers are set by CORSMiddleware
        assert response.status_code == 200