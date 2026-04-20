"""
Pytest configuration and shared fixtures for backend tests.
"""
import sys
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch

import pytest

# Ensure backend is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from session_manager import SessionManager
from rag_system import RAGSystem


# ---------------------------------------------------------------------------
# Mock ChromaDB / embedding function
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedding_function():
    """Return a MagicMock that can stand in for SentenceTransformerEmbeddingFunction."""
    return MagicMock()


@pytest.fixture
def mock_chroma_client():
    """Return a MagicMock ChromaDB PersistentClient with two collections."""
    client = MagicMock()

    catalog_collection = MagicMock()
    content_collection = MagicMock()

    client.get_or_create_collection.side_effect = lambda name, **kwargs: (
        catalog_collection if name == "course_catalog" else content_collection
    )

    return client, catalog_collection, content_collection


# ---------------------------------------------------------------------------
# Vector store fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_search_results():
    """SearchResults with two mock documents."""
    return SearchResults(
        documents=[
            "This is the first chunk about Python functions.",
            "This is the second chunk about JavaScript async/await.",
        ],
        metadata=[
            {"course_title": "Python Basics", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "JavaScript Advanced", "lesson_number": 3, "chunk_index": 2},
        ],
        distances=[0.12, 0.34],
    )


@pytest.fixture
def empty_search_results():
    """Empty SearchResults (no error)."""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """SearchResults with an error message."""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Search error: ChromaDB connection refused",
    )


@pytest.fixture
def mock_vector_store(populated_search_results, mock_chroma_client):
    """VectorStore backed by a temp directory and mocked ChromaDB."""
    client, catalog_coll, content_coll = mock_chroma_client

    # Mock catalog query for course name resolution
    catalog_coll.query.return_value = {
        "documents": [["Python Basics"]],
        "metadatas": [[{"title": "Python Basics"}]],
        "distances": [[0.1]],
    }

    # Mock content query for search
    content_coll.query.return_value = {
        "documents": [populated_search_results.documents],
        "metadatas": [populated_search_results.metadata],
        "distances": [populated_search_results.distances],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("vector_store.chromadb.PersistentClient", return_value=client):
            from vector_store import VectorStore
            store = VectorStore(
                chroma_path=tmpdir,
                embedding_model="all-MiniLM-L6-v2",
                max_results=5,
            )
            yield store


# ---------------------------------------------------------------------------
# CourseSearchTool fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def course_search_tool(mock_vector_store):
    """CourseSearchTool with a mocked vector store."""
    return CourseSearchTool(mock_vector_store)


# ---------------------------------------------------------------------------
# ToolManager fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tool_manager(course_search_tool):
    """ToolManager pre-registered with CourseSearchTool."""
    tm = ToolManager()
    tm.register_tool(course_search_tool)
    return tm


# ---------------------------------------------------------------------------
# Mock Ollama client fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ollama_response_direct():
    """Ollama response with direct answer (no tool calls)."""
    response = MagicMock()
    response.message = MagicMock()
    response.message.content = "Python functions are defined using the `def` keyword."
    response.message.tool_calls = None
    return response


@pytest.fixture
def mock_ollama_response_with_tool_call():
    """Ollama response requesting a tool call."""
    response = MagicMock()
    response.message = MagicMock()
    response.message.content = None  # tool use requested, no direct content

    # Mock a single tool call
    tool_call = MagicMock()
    tool_call.function = MagicMock()
    tool_call.function.name = "search_course_content"
    tool_call.function.arguments = {
        "query": "Python functions",
        "course_name": "Python",
        "lesson_number": None,
    }
    response.message.tool_calls = [tool_call]
    return response


@pytest.fixture
def mock_ollama_final_response():
    """Final Ollama response after tool result is injected."""
    response = MagicMock()
    response.message = MagicMock()
    response.message.content = (
        "Based on the search results, Python functions are defined "
        "using the `def` keyword followed by the function name."
    )
    return response


# ---------------------------------------------------------------------------
# AIGenerator fixture (with mocked Ollama client)
# ---------------------------------------------------------------------------

@pytest.fixture
def ai_generator(mock_ollama_response_direct):
    """AIGenerator with a mocked Ollama client returning direct answer."""
    with patch("ai_generator.Client") as MockClient:
        MockClient.return_value.chat.return_value = mock_ollama_response_direct
        gen = AIGenerator(model="test-model", host="http://localhost:11434")
        yield gen, MockClient.return_value


# ---------------------------------------------------------------------------
# RAGSystem fixture (fully mocked)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ollama_for_rag(mock_ollama_response_with_tool_call, mock_ollama_final_response):
    """
    RAGSystem backed by mocked Ollama.
    First call returns tool call, second call returns synthesis.
    """
    mock_client = MagicMock()
    # First invocation: tool call
    mock_client.chat.side_effect = [
        mock_ollama_response_with_tool_call,
        mock_ollama_final_response,
    ]
    return mock_client


@pytest.fixture
def rag_system(mock_vector_store, mock_ollama_for_rag):
    """RAGSystem with mocked vector store and Ollama client."""
    with patch("rag_system.AIGenerator") as MockAIGen:
        mock_instance = MagicMock()
        mock_instance.generate_response.return_value = (
            "Based on the search results, Python functions are defined using `def`."
        )
        MockAIGen.return_value = mock_instance

        from config import Config
        config = Config()

        with patch("rag_system.VectorStore", return_value=mock_vector_store):
            with patch("rag_system.SessionManager") as MockSM:
                mock_sm_instance = MagicMock()
                mock_sm_instance.get_conversation_history.return_value = None
                MockSM.return_value = mock_sm_instance

                rag = RAGSystem(config)
                rag.ai_generator = mock_instance
                rag.session_manager = mock_sm_instance
                yield rag


# ---------------------------------------------------------------------------
# Sample course documents (in-memory)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_course_data():
    """Minimal course data used across tests."""
    return {
        "title": "Python Basics",
        "course_link": "https://example.com/python",
        "instructor": "Jane Doe",
        "lessons": [
            {
                "lesson_number": 1,
                "title": "Introduction",
                "lesson_link": "https://example.com/python/lesson1",
            },
            {
                "lesson_number": 2,
                "title": "Functions",
                "lesson_link": "https://example.com/python/lesson2",
            },
        ],
    }


# ---------------------------------------------------------------------------
# API Test Client fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_app():
    """Create a FastAPI test app without static file mounting."""
    import warnings
    warnings.filterwarnings("ignore", message="resource_tracker:.*")

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    app = FastAPI(title="Course Materials RAG System (Test)", root_path="")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # Store for mocked RAG system - will be injected by tests
    app.state.rag_system = None

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = app.state.rag_system.session_manager.create_session()

            answer, sources = app.state.rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = app.state.rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"status": "ok", "message": "RAG System API"}

    return app


@pytest.fixture
def test_client(test_app):
    """FastAPI test client with mocked RAG system."""
    from httpx import AsyncClient, ASGITransport

    async def _create_client(mock_rag_system=None):
        if mock_rag_system is not None:
            test_app.state.rag_system = mock_rag_system
        return AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test")

    return _create_client
