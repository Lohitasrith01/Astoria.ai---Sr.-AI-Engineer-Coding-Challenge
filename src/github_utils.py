import os
import requests
from typing import Dict, Any


GITHUB_API_URL = "https://api.github.com"


def _github_headers() -> Dict[str, str]:
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_github_profile(handle: str) -> Dict[str, Any]:
    """Return basic user info and repo list summary for a GitHub handle.

    We keep the number of REST calls low to stay within free-tier limits (60/h unauth; 5k/h auth).
    """
    user_resp = requests.get(f"{GITHUB_API_URL}/users/{handle}", headers=_github_headers(), timeout=10)
    if user_resp.status_code != 200:
        raise ValueError(f"GitHub user '{handle}' not found or API error: {user_resp.text}")
    user_data = user_resp.json()

    repos_resp = requests.get(
        f"{GITHUB_API_URL}/users/{handle}/repos?per_page=100&type=owner&sort=pushed",
        headers=_github_headers(),
        timeout=10,
    )
    repos = repos_resp.json() if repos_resp.status_code == 200 else []

    aggregate = {
        "public_repos": user_data.get("public_repos", 0),
        "followers": user_data.get("followers", 0),
        "total_stars": sum(repo.get("stargazers_count", 0) for repo in repos),
        "repo_count": len(repos),
        "languages": list({repo.get("language") for repo in repos if repo.get("language")}),
    }
    return aggregate 