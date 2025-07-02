from __future__ import annotations

import math
import re
from typing import Dict, Set, List, Tuple

from .models import JobPost, Candidate
from .github_utils import fetch_github_profile
from .repo_indexer import fetch_and_analyse_github, TECH_KEYWORDS
from .linkedin_utils import analyze_profile, linkedin_score as _ln_score


KEYWORD_SPLIT_RE = re.compile(r"[^a-zA-Z0-9_+#]")


def tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in KEYWORD_SPLIT_RE.split(text) if tok}


def resume_relevance(job: JobPost, candidate: Candidate) -> float:
    jd_tokens = tokenize(job.description)
    resume_tokens = tokenize(candidate.resume_text)
    overlap = jd_tokens & resume_tokens
    if not jd_tokens:
        return 0.0
    return len(overlap) / len(jd_tokens)


# Map tech categories for skill-transfer inference
_CATEGORY_MAP = {
    "ml_framework": {"pytorch", "tensorflow", "keras", "mxnet"},
    "cloud": {"aws", "gcp", "azure"},
    "vector_db": {"pinecone", "milvus", "weaviate", "redis"},
}


def _job_required_tools(job: JobPost) -> set[str]:
    jd_tokens = tokenize(job.description)
    return {tok for tok in jd_tokens if tok in _flatten_categories() or tok in TECH_KEYWORDS}


def _flatten_categories() -> set[str]:
    flat = set()
    for s in _CATEGORY_MAP.values():
        flat |= s
    return flat


def tech_match_score(job: JobPost, detected_tools: set[str]) -> tuple[float, list[str]]:
    required = _job_required_tools(job)
    if not required:
        return 0.0, []

    direct_match = required & detected_tools
    direct_ratio = len(direct_match) / len(required)

    # Skill transfer: same category but not direct
    transferable: list[str] = []
    for cat, group in _CATEGORY_MAP.items():
        if group & required and not group & detected_tools:
            # if candidate has any tech in same category different from req
            if group & detected_tools:
                transferable.append(cat)

    transfer_ratio = len(transferable) / len(_CATEGORY_MAP) if _CATEGORY_MAP else 0

    total = 0.7 * direct_ratio + 0.3 * transfer_ratio
    return total, transferable


def github_score(job: JobPost, candidate: Candidate) -> tuple[float, set[str], set[str], float]:
    if not candidate.github_handle:
        return 0.0, set(), set(), 0.0
    try:
        profile = fetch_github_profile(candidate.github_handle)
    except Exception:
        profile = {
            "repo_count": 0,
            "total_stars": 0,
            "followers": 0,
        }

    # MCP metrics (may fail quietly)
    try:
        mcp_metrics = fetch_and_analyse_github(candidate.github_handle)
        mcp_total = mcp_metrics.get("mcp_total", 0)
        detected_tools = set(mcp_metrics.get("detected_tools", []))
        detected_langs = set(mcp_metrics.get("detected_languages", []))
    except Exception:
        mcp_metrics = {"mcp_total": 0.0}
        mcp_total = 0.0
        detected_tools = set()
        detected_langs = set()

    # Heuristic scoring
    score = (
        profile["repo_count"] * 0.3
        + profile["total_stars"] * 0.05
        + profile["followers"] * 0.1
    )
    # Add MCP-based components
    tech_score, _ = tech_match_score(job, detected_tools)
    score += 50 * mcp_total + 30 * tech_score

    # Normalize to 0-1 after considering approximate upper bound (200)
    norm = min(score / 200.0, 1.0)
    return norm, detected_tools, detected_langs, tech_score


def overall_score(job: JobPost, candidate: Candidate) -> Dict[str, float]:
    resume_score = resume_relevance(job, candidate)
    gh_score, tools, langs, tech_subscore = github_score(job, candidate)

    if candidate.linkedin_url:
        ln_metrics = analyze_profile(candidate.linkedin_url)
        linkedin_score = _ln_score(ln_metrics)
    else:
        linkedin_score = 0.0

    total = 0.6 * resume_score + 0.3 * gh_score + 0.1 * linkedin_score

    return {
        "resume": resume_score,
        "github": gh_score,
        "linkedin": linkedin_score,
        "total": total,
        "tools": ", ".join(sorted(tools)),
        "languages": ", ".join(sorted(langs)),
        "tech_match": tech_subscore,
    } 