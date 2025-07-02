from __future__ import annotations

from typing import Dict, Any

from src.repo_indexer import analyse_repo_with_mcp, _clone_repo
from src.github_utils import fetch_github_profile

# Simplified tool implementations for demo


def search_code(pattern: str, file_pattern: str | None = None) -> Dict[str, Any]:
    """Mock search over indexed files; returns matches list."""
    # For a demo we return a stub response.
    # Integration with MCP search_code can be added later.
    return {"matches": [f"Found '{pattern}' in dummy_file.py:42"]}


def get_file_summary(file_path: str) -> Dict[str, Any]:
    # Stub file summary
    return {
        "file_path": file_path,
        "line_count": 120,
        "functions": ["def handler()", "def schema()"],
    }

TOOL_MAP = {
    "search_code": search_code,
    "get_file_summary": get_file_summary,
} 