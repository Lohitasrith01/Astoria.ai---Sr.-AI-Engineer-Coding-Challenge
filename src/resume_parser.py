from __future__ import annotations

from pathlib import Path
from typing import Dict

# Optional imports with fallbacks
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("⚠️  pdfplumber not found. PDF parsing will be limited.")

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("⚠️  BeautifulSoup not found. HTML parsing will be limited.")

try:
    from docx import Document  # type: ignore
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("⚠️  python-docx not found. DOCX parsing will be limited.")

# Removed LangChain dependencies - we don't need chunking for LLM scoring!
# from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter


class ResumeParserError(Exception):
    pass


def load_resume_text(path_str: str | Path) -> str:
    """
    Load complete resume text for LLM processing.
    No chunking needed - modern LLMs can handle full resumes (128K+ tokens).
    """
    path = Path(path_str)
    if not path.exists():
        raise ResumeParserError(f"Resume file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix == ".docx":
        return _parse_docx(path)
    elif suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif suffix in {".html", ".htm"}:
        return _parse_html(path)
    else:
        raise ResumeParserError(f"Unsupported resume format: {suffix}")


def _parse_pdf(pdf_path: Path) -> str:
    """Parse PDF with fallback if pdfplumber is not available."""
    if not HAS_PDFPLUMBER:
        raise ResumeParserError("pdfplumber not installed. Run: pip install pdfplumber")
    
    text_parts: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
    except Exception as e:
        raise ResumeParserError(f"Failed to parse PDF: {e}")
    return "\n".join(text_parts)


def _parse_docx(docx_path: Path) -> str:
    """Parse DOCX with fallback if python-docx is not available."""
    if not HAS_DOCX:
        raise ResumeParserError("python-docx not installed. Run: pip install python-docx")
    
    try:
        doc = Document(str(docx_path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        raise ResumeParserError(f"Failed to parse DOCX: {e}")


def _parse_html(html_path: Path) -> str:
    """Parse HTML with fallback if BeautifulSoup is not available."""
    if not HAS_BS4:
        raise ResumeParserError("beautifulsoup4 not installed. Run: pip install beautifulsoup4")
    
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    return soup.get_text(" \n")


# Optional: Chunk text ONLY for sentence transformer pre-filtering
def chunk_for_embeddings(text: str, max_tokens: int = 512) -> list[str]:
    """
    Chunk text only when needed for sentence transformers (ATS pre-filter).
    Most of the time, you won't need this - send full text to LLM instead.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_tokens = len(word) // 4 + 1  # rough token estimate
        if current_length + word_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_tokens
        else:
            current_chunk.append(word)
            current_length += word_tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


# Simple skill extraction for quick heuristics (optional)
import re
from collections import Counter

SKILL_REGEX = re.compile(r"\b(python|java(script)?|typescript|c\+\+|c#|go|rust|react|django|flask|fastapi|aws|gcp|azure|docker|kubernetes|tensorflow|pytorch|mlops|graphql|sql|postgres|mysql|mongodb)\b",
                        re.IGNORECASE)

def extract_skills_heuristic(text: str) -> Dict[str, str | int | list[str]]:
    """
    Quick skill extraction - use this for fast filtering, 
    let LLM do the deep analysis on full text.
    """
    # Experience years heuristic
    years = [int(y) for y in re.findall(r"(19|20)\d{2}", text)]
    exp_years = 0
    if len(years) >= 2:
        exp_years = max(years) - min(years)
        exp_years = max(exp_years, 0)

    # Skills frequency
    skills_found = [m.group(0).lower() for m in SKILL_REGEX.finditer(text)]
    top_skills = [s for s, _c in Counter(skills_found).most_common(15)]

    return {
        "experience_years": exp_years,
        "skills": top_skills,
    }


# Backward compatibility - keeping old function names
load_text_from_file = load_resume_text  # alias
analyse_text = extract_skills_heuristic  # alias 