# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

All dependency management uses **uv** — never use pip or python directly.

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package>

# Remove a dependency
uv remove <package>

# Run any Python script
uv run python <script.py>

# Run the application (starts backend + serves frontend)
uv run ./run.sh

# Or manually (always use uv run, never python directly)
cd backend && uv run uvicorn app:app --reload --port 8000
```

Access at http://localhost:8000. Requires `ANTHROPIC_API_KEY` in `.env` file.
Note: `run.sh` handles its own uv invocation internally.

## Architecture

### RAG System Flow
The system uses **tool-based AI generation** where Claude decides when to search course materials:

1. User query → `RAGSystem.query()`
2. `AIGenerator.generate_response()` sends query + tool definitions to Claude
3. If Claude responds with `tool_use` stop reason, `_handle_tool_execution()` runs `CourseSearchTool`
4. `CourseSearchTool.execute()` calls `VectorStore.search()` which queries ChromaDB
5. Search results are returned to Claude, which synthesizes a final response

### Key Components
- **RAGSystem** (`rag_system.py`): Orchestrates document processing, search, AI generation, and sessions
- **AIGenerator** (`ai_generator.py`): Thin wrapper around Anthropic Claude API with tool execution
- **ToolManager/CourseSearchTool** (`search_tools.py`): Registers Claude tools and executes searches
- **VectorStore** (`vector_store.py`): ChromaDB wrapper with two collections:
  - `course_catalog`: Course metadata (title, instructor, lessons)
  - `course_content`: Chunked course material for semantic search
- **DocumentProcessor** (`document_processor.py`): Parses course docs, chunks text by sentences
- **SessionManager** (`session_manager.py`): In-memory conversation history (default: 2 exchanges)

### Document Format
Course files in `docs/` use this format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 1: [title]
Lesson Link: [url]
[content...]

Lesson 2: [title]
[content...]
```

### ChromaDB Storage
- Location: `./chroma_db/` (persistent)
- Course titles serve as document IDs in `course_catalog`
- Chunk IDs: `{course_title}_{chunk_index}`

### Configuration (`backend/config.py`)
- `CHUNK_SIZE`: 800 chars (text chunks)
- `CHUNK_OVERLAP`: 100 chars
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation exchanges
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
