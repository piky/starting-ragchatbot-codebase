#!/bin/bash
# Run ruff linter

cd "$(dirname "$0")/.."
uv run ruff check .