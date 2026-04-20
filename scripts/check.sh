#!/bin/bash
# Run all quality checks: format, lint, and test

set -e

cd "$(dirname "$0")/.."

echo "=== Running Black (format check) ==="
uv run black --check .

echo ""
echo "=== Running Ruff (lint check) ==="
uv run ruff check .

echo ""
echo "=== Running Tests ==="
uv run pytest backend/tests -v

echo ""
echo "=== All checks passed! ==="