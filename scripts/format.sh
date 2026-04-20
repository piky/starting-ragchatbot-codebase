#!/bin/bash
# Format all Python files with black

cd "$(dirname "$0")/.."
uv run black .