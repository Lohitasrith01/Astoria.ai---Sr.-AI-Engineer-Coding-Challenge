import os
import shutil
import subprocess
import tempfile
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Any

# Optional GitHub import
try:
    from github import Github  # type: ignore
    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False
    print("⚠️  PyGithub not found. GitHub analysis will be disabled.")

# Attempt to import internals of code_index_mcp for indexing convenience
try:
    from code_index_mcp.server import _index_project, supported_extensions, file_index  # type: ignore
except ImportError:  # pragma: no cover
    _index_project = None
    supported_extensions: List[str] = []
    file_index = {}


def _clone_repo(repo_clone_url: str, dest_dir: Path) -> Path:
    """Clone the given repository into dest_dir and return the repo path."""
    repo_name = repo_clone_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = dest_dir / repo_name
    if repo_path.exists():
        return repo_path  # already cloned
    subprocess.run(["git", "clone", "--depth", "1", repo_clone_url, str(repo_path)], check=True)
    return repo_path


def _basic_repo_metrics(repo_path: Path) -> Dict[str, int | Set[str]]:
    """Fallback metrics calculation without MCP."""
    languages = set()
    file_count = 0
    for root, _dirs, files in os.walk(repo_path):
        for fname in files:
            ext = Path(fname).suffix
            if ext:
                languages.add(ext.lstrip("."))
            file_count += 1
    return {"file_count": file_count, "languages": languages}


def analyse_repo_with_mcp(repo_path: Path) -> Dict[str, int | Set[str]]:
    if _index_project is None:
        return _basic_repo_metrics(repo_path)
    # Clear global file_index before each call
    file_index.clear()  # type: ignore
    try:
        _index_project(str(repo_path))
        langs: Set[str] = set()
        file_cnt = 0
        for rel_path in _walk_indexed_files(file_index):  # type: ignore
            file_cnt += 1
            ext = Path(rel_path).suffix.lstrip(".")
            if ext:
                langs.add(ext)
        return {"file_count": file_cnt, "languages": langs}
    except Exception:
        return _basic_repo_metrics(repo_path)


def _walk_indexed_files(tree: dict, prefix: str = "") -> List[str]:
    paths: List[str] = []
    for name, node in tree.items():
        if isinstance(node, dict) and node.get("type") == "file":
            paths.append(os.path.join(prefix, name))
        elif isinstance(node, dict):
            paths.extend(_walk_indexed_files(node, os.path.join(prefix, name)))
    return paths


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# ---- Tech keywords list (extensible) ----

TECH_KEYWORDS: Set[str] = {
    # Programming languages
    "python", "javascript", "typescript", "java", "c", "c++", "c#", "go", "rust", "scala", "kotlin", "ruby", "php", "swift", "dart", "r", "matlab",
    # ML / Data
    "pytorch", "tensorflow", "keras", "mxnet", "sklearn", "scikit-learn", "xgboost", "lightgbm", "pandas", "numpy", "scipy", "huggingface", "langchain",
    # RL / AI specific
    "q-learning", "qlearning", "dqn", "ppo", "dpo", "a3c", "ddpg", "sac", "td3", "trpo", "reinforcement learning", "rl", "mdp", "markov decision process",
    "openai gym", "gymnasium", "stable baselines", "ray rllib", "policy gradient", "value function", "actor critic", "monte carlo",
    # Cloud / DevOps
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ansible", "jenkins", "github actions", "circleci", "sagemaker", "bedrock",
    # Databases
    "postgres", "mysql", "mongodb", "redis", "dynamodb", "cassandra", "neo4j", "sqlite", "bigquery",
    # Web frameworks
    "react", "angular", "vue", "django", "flask", "fastapi", "express", "next.js", "nuxt",
}


def _extract_tools_from_text(text: str) -> Set[str]:
    text_lower = text.lower()
    found: Set[str] = set()
    for kw in TECH_KEYWORDS:
        # Handle both exact matches and word boundaries
        if len(kw.split()) > 1:  # Multi-word keywords
            if kw in text_lower:
                found.add(kw)
        else:  # Single word keywords
            import re
            # Use word boundaries to avoid false positives
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                found.add(kw)
    return found


def fetch_and_analyse_github(handle: str, max_repos: int = 20) -> Dict[str, Any]:
    """Analyze GitHub repos by reading ALL READMEs first, then let LLM decide if deeper analysis is needed."""
    if not HAS_GITHUB:
        print(f"⚠️  GitHub analysis disabled for {handle}")
        return {
            "total_files": 0,
            "languages_count": 0,
            "mcp_file_score": 0.0,
            "mcp_language_score": 0.0,
            "mcp_total": 0.0,
            "detected_languages": [],
            "detected_tools": [],
            "project_summaries": [],
        }
    
    gh = Github(GITHUB_TOKEN or None, per_page=100)
    user = gh.get_user(handle)
    # Get all public repos, sorted by activity/stars
    repos = sorted(user.get_repos(), key=lambda r: (r.stargazers_count, r.updated_at), reverse=True)[:max_repos]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        agg_langs: Set[str] = set()
        total_files = 0
        agg_tools: Set[str] = set()
        project_summaries = []
        
        print(f"📊 Analyzing {len(repos)} repositories for {handle}...")
        
        for i, repo in enumerate(repos):
            print(f"  {i+1}/{len(repos)}: {repo.name}")
            
            try:
                repo_path = _clone_repo(repo.clone_url, tmp_path)
                metrics = analyse_repo_with_mcp(repo_path)
                total_files += int(metrics["file_count"])
                agg_langs.update(metrics["languages"])
                
                # Comprehensive README analysis
                readme_files = [p for p in repo_path.iterdir() if p.name.lower().startswith("readme")]
                readme_content = ""
                repo_tools = set()
                
                for readme in readme_files:
                    try:
                        content = readme.read_text(encoding="utf-8", errors="ignore")
                        readme_content += content + "\n"
                        tools = _extract_tools_from_text(content)
                        repo_tools.update(tools)
                    except Exception:
                        continue
                
                # Create project summary
                project_summary = {
                    "repo_name": repo.name,
                    "stars": repo.stargazers_count,
                    "description": repo.description or "",
                    "readme_content": readme_content[:2000],  # First 2000 chars
                    "detected_tools": list(repo_tools),
                    "languages": list(metrics["languages"]),
                    "file_count": int(metrics["file_count"]),
                    "last_updated": repo.updated_at.isoformat() if repo.updated_at else "",
                    "topics": list(repo.get_topics()) if hasattr(repo, 'get_topics') else []
                }
                
                project_summaries.append(project_summary)
                agg_tools.update(repo_tools)
                
            except Exception as e:
                print(f"    ❌ Error analyzing {repo.name}: {e}")
                continue
        
        # Calculate scores
        file_score = min(total_files / 500.0, 1.0)
        lang_score = min(len(agg_langs) / 10.0, 1.0)
        
        return {
            "total_files": total_files,
            "languages_count": len(agg_langs),
            "mcp_file_score": file_score,
            "mcp_language_score": lang_score,
            "mcp_total": 0.7 * file_score + 0.3 * lang_score,
            "detected_languages": list(agg_langs),
            "detected_tools": list(agg_tools),
            "project_summaries": project_summaries,
            "repos_analyzed": len(project_summaries),
        }


def deep_analyze_repo(handle: str, repo_name: str) -> Dict[str, Any]:
    """Deep analysis of a specific repo including all code files, notebooks, etc."""
    if not HAS_GITHUB:
        return {"error": "GitHub analysis disabled"}
    
    gh = Github(GITHUB_TOKEN or None, per_page=100)
    user = gh.get_user(handle)
    
    try:
        repo = user.get_repo(repo_name)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            repo_path = _clone_repo(repo.clone_url, tmp_path)
            
            deep_tools = set()
            file_analysis = []
            
            # Walk all files for deep analysis
            for root, _dirs, files in os.walk(repo_path):
                for fname in files:
                    fpath = Path(root) / fname
                    ext = fpath.suffix.lower()
                    
                    try:
                        if ext == ".ipynb":
                            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                                nb = json.load(f)
                                nb_text = ""
                                for cell in nb.get("cells", []):
                                    if cell.get("cell_type") in ("code", "markdown"):
                                        nb_text += "\n".join(cell.get("source", [])) + "\n"
                                tools = _extract_tools_from_text(nb_text)
                                deep_tools.update(tools)
                                if tools:
                                    file_analysis.append({
                                        "file": str(fpath.relative_to(repo_path)),
                                        "type": "jupyter_notebook",
                                        "tools": list(tools),
                                        "content_preview": nb_text[:500]
                                    })
                        
                        elif ext == ".py":
                            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                                py_text = f.read()
                                tools = _extract_tools_from_text(py_text)
                                deep_tools.update(tools)
                                if tools:
                                    file_analysis.append({
                                        "file": str(fpath.relative_to(repo_path)),
                                        "type": "python_code",
                                        "tools": list(tools),
                                        "content_preview": py_text[:500]
                                    })
                        
                        elif ext == ".rmd":
                            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                                rmd_text = f.read()
                                code_chunks = re.findall(r'```[a-zA-Z]*\n(.*?)```', rmd_text, re.DOTALL)
                                all_text = rmd_text + "\n".join(code_chunks)
                                tools = _extract_tools_from_text(all_text)
                                deep_tools.update(tools)
                                if tools:
                                    file_analysis.append({
                                        "file": str(fpath.relative_to(repo_path)),
                                        "type": "r_markdown",
                                        "tools": list(tools),
                                        "content_preview": all_text[:500]
                                    })
                    except Exception:
                        continue
            
            return {
                "repo_name": repo_name,
                "deep_tools": list(deep_tools),
                "file_analysis": file_analysis,
                "files_analyzed": len(file_analysis)
            }
            
    except Exception as e:
        return {"error": f"Failed to analyze {repo_name}: {e}"} 