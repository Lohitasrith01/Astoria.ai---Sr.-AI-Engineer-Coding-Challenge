from __future__ import annotations

import os
import re
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

# Simplified list – extend as needed
BIG_TECH = {
    "google",
    "alphabet",
    "microsoft",
    "amazon",
    "apple",
    "meta",
    "facebook",
    "netflix",
    "openai",
    "nvidia",
}

TECH_KEYWORDS_POSTS = {
    "rag",
    "langchain",
    "llm",
    "transformer",
    "mistral",
    "fine-tune",
    "sagemaker",
    "bedrock",
    "mlops",
    "kubernetes",
    "graphql",
}

# --------------------------------------------------
# LinkedIn MCP Placeholder Implementation
# --------------------------------------------------

class LinkedInMCP:
    """
    Placeholder for a LinkedIn Machine Code Processing (MCP) or API integration.
    In production, this would connect to LinkedIn's API or a robust scraping/crawling backend.
    All methods here return mock/stub data for demonstration/testing purposes only.
    """
    def __init__(self):
        pass

    def analyze_profile(self, linkedin_url: str) -> Dict[str, float | int]:
        """
        Analyze a LinkedIn profile and return structured metrics.
        This is a stub. Replace with real API/MCP logic in production.
        """
        # Return mock data
        return {
            "exp_years": 5,
            "bigtech": False,
            "posts_90d": 2,
            "post_tech_ratio": 0.1,
            "publications": 0,
        }

    def linkedin_score(self, metrics: Dict[str, float | int]) -> float:
        """
        Convert raw metrics into a 0-1 score. This is a stub.
        """
        exp_score = min(int(metrics["exp_years"]) / 10, 1)
        bigtech_bonus = 0.2 if metrics["bigtech"] else 0.0
        post_score = min(int(metrics["posts_90d"]) / 20, 1) * 0.5 + float(metrics["post_tech_ratio"]) * 0.5
        pubs_score = min(int(metrics["publications"]) / 5, 1)
        score = 0.4 * exp_score + 0.2 * post_score + 0.2 * pubs_score + bigtech_bonus
        return min(score, 1.0)

# Singleton pattern for MCP instance
_linkedin_mcp = None

def get_linkedin_mcp() -> LinkedInMCP:
    """
    Get or create the global LinkedIn MCP instance.
    """
    global _linkedin_mcp
    if _linkedin_mcp is None:
        _linkedin_mcp = LinkedInMCP()
    return _linkedin_mcp

# --------------------------------------------------
# Deprecated: Old placeholder functions (for backward compatibility)
# --------------------------------------------------

def analyze_profile(linkedin_url: str) -> Dict[str, float | int]:
    """
    Deprecated. Use LinkedInMCP.analyze_profile instead.
    """
    return get_linkedin_mcp().analyze_profile(linkedin_url)

def linkedin_score(metrics: Dict[str, float | int]) -> float:
    """
    Deprecated. Use LinkedInMCP.linkedin_score instead.
    """
    return get_linkedin_mcp().linkedin_score(metrics)

# --------------------------------------------------
# NOTE: LinkedIn aggressively blocks scraping.  
# In production use a proper API or headless browser.  
# For the POC we assume the HTML is publicly accessible.
# --------------------------------------------------

def _fetch_html(url: str) -> str | None:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        }
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            return res.text
    except Exception:
        pass
    return None


def _extract_experience_years(soup: BeautifulSoup) -> int:
    exp_section = soup.find(text=re.compile(r"Experience"))
    if not exp_section:
        return 0
    # naive heuristic – count year spans like "2019 – 2023" or "Jan 2020 – Present"
    years: List[int] = []
    for match in re.finditer(r"(19|20)\d{2}", exp_section.parent.get_text(" ")):
        years.append(int(match.group()))
    if len(years) < 2:
        return 0
    years.sort()
    return max(years) - min(years)


def _worked_big_tech(soup: BeautifulSoup) -> bool:
    text = soup.get_text(" ").lower()
    return any(bt in text for bt in BIG_TECH)


def _post_activity(soup: BeautifulSoup) -> tuple[int, float]:
    """Return post_count_last_90_days, tech_keyword_ratio"""
    # simplified: look for 'posts' section
    posts = [t.get_text(" ") for t in soup.find_all("span") if " · Post" in t.get_text(" ")]
    recent_cutoff = datetime.now() - timedelta(days=90)
    recent_posts = posts  # placeholder – can't get timestamps w/out API
    if not recent_posts:
        return 0, 0.0
    joined_text = " ".join(recent_posts).lower()
    kw_hits = sum(1 for kw in TECH_KEYWORDS_POSTS if kw in joined_text)
    return len(recent_posts), kw_hits / len(TECH_KEYWORDS_POSTS)


def _publications(soup: BeautifulSoup) -> int:
    pubs_header = soup.find(text=re.compile(r"Publications"))
    if not pubs_header:
        return 0
    # Count list items following the header
    ul = pubs_header.find_next("ul")
    if not ul:
        return 0
    return len(ul.find_all("li"))


def analyze_profile(linkedin_url: str) -> Dict[str, float | int]:
    html = _fetch_html(linkedin_url)
    if not html:
        return {
            "exp_years": 0,
            "bigtech": False,
            "posts_90d": 0,
            "post_tech_ratio": 0.0,
            "publications": 0,
        }
    soup = BeautifulSoup(html, "lxml")
    exp_years = _extract_experience_years(soup)
    bigtech = _worked_big_tech(soup)
    posts_90d, post_tech_ratio = _post_activity(soup)
    pubs = _publications(soup)

    return {
        "exp_years": exp_years,
        "bigtech": bigtech,
        "posts_90d": posts_90d,
        "post_tech_ratio": post_tech_ratio,
        "publications": pubs,
    }


def linkedin_score(metrics: Dict[str, float | int]) -> float:
    """Convert raw metrics into 0-1 score."""
    exp_score = min(int(metrics["exp_years"]) / 10, 1)
    bigtech_bonus = 0.2 if metrics["bigtech"] else 0.0
    post_score = min(int(metrics["posts_90d"]) / 20, 1) * 0.5 + float(metrics["post_tech_ratio"]) * 0.5
    pubs_score = min(int(metrics["publications"]) / 5, 1)

    score = 0.4 * exp_score + 0.2 * post_score + 0.2 * pubs_score + bigtech_bonus
    return min(score, 1.0) 