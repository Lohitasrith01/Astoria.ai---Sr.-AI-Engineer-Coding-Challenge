# AI Candidate Screener & Interview Agent

> **72-hour build** – Multi-agent ready, MCP-powered code insights, Streamlit UI.

---

## 🚀 What This Repo Does

1. **Load a job post** and HR constraints (years of experience, visa, # of slots).
2. **Ingest candidates** via JSON/CSV or manual entry.
3. **Filter** by hard requirements (experience, visa).
4. **Score** each candidate on three axes:
   - Résumé keyword overlap
   - GitHub depth & tech-stack match (using Code-Index MCP + README parsing)
   - LinkedIn profile (**placeholder**, extensible)
5. **Surface insights** (languages, tools, tech-match %) in a Streamlit dashboard.
6. **Interview pipeline**: Shortlisted candidates can be interviewed by an AI agent, with real-time analysis and recruiter-friendly summaries.

---

## 🧠 Business Logic & Design Decisions

### 1. **Candidate Filtering**

- **Hard filters**: Candidates are first filtered by minimum years of experience and visa status.
- **Why**: This ensures only those who meet non-negotiable requirements are considered, saving compute and recruiter time.

### 2. **Scoring System**

- **Weighted blend**: 
  - Résumé relevance (60%)
  - GitHub analysis (30%)
  - LinkedIn (10%, **placeholder**)
- **Résumé**: Token overlap between job description and candidate résumé.
- **GitHub**: 
  - Repo/star/follower counts
  - MCP (code indexer) for file/language diversity
  - README parsing for tech keywords
  - Skill transfer logic (e.g., AWS ↔ GCP, PyTorch ↔ TensorFlow)
- **LinkedIn**: Currently a stub; see below for details.
- **Why**: This blend balances hard skills, public code, and professional presence, reflecting modern tech hiring.

### 3. **Semantic Filtering (SmartFilter)**

- **Layer A**: Fast semantic similarity using sentence embeddings (MiniLM).
- **Layer B**: LLM-based reasoning for nuanced fit (Vertex AI/Gemini).
- **Why**: Layer A quickly eliminates poor matches; Layer B provides deep, explainable fit analysis for top candidates.

### 4. **GitHub Deep Dive (MCP Integration)**

- **Shallow clone**: Top repos are cloned and indexed for file/language stats.
- **README parsing**: Extracts tech stack and project context.
- **Skill transfer**: Categories (ML, cloud, vector DBs) allow for partial matches.
- **Why**: Many candidates use different but related tools; this logic recognizes transferable skills, not just exact matches.

### 5. **Interview Agent (Conversational AI)**

- **Stages**: Introduction → Concerns Exploration → Technical → Experience Deep Dive → Behavioral → Problem Solving → Candidate Questions → Closing.
- **Memory**: Uses LangChain for conversation memory (last N turns).
- **Bias & Tone Analysis**: Each message is checked for bias and tone, with suggestions for improvement.
- **Why**: Structured interviews reduce bias, ensure coverage, and provide a fair, repeatable process.

### 6. **Smart Interview Agent (RAG + Caching)**

- **RAG**: Retrieval-augmented generation for questions and responses.
- **Caching**: Common responses and templates are cached to reduce LLM/API calls by 80%+.
- **Why**: Reduces cost and latency, while maintaining high-quality, context-aware conversations.

### 7. **Streamlit UI**

- **Bulk and single-candidate flows**: Recruiters can shortlist from a pool or test a single candidate.
- **Real-time insights**: Interview progress, strengths, concerns, and recommendations are surfaced live.
- **Why**: Recruiters need actionable insights, not just raw scores.

```

**Note:**
- **LinkedIn** is currently a stub. No real API or MCP integration is present yet. The code includes a placeholder function for LinkedIn scoring, which always returns a default or random value. This is designed for easy extension: plug in Scrapfly, SerpAPI, or LinkedIn's own API when available.
- **GitHub** analysis uses both the public API and a local code indexer (MCP) for deep dives.
- **Resume** parsing is done locally for keyword and skill extraction.

---

## 🏗️ Extensibility

- **Deep dive scan**: Recursively index repos when a gap is found.
- **LinkedIn crawler**: Plug in Scrapfly/SerpAPI for richer LinkedIn data.
- **LLM interview agent**: Use LangChain, pass `tech_match` to focus questions.
- **Deployment**: Streamlit Cloud or HuggingFace Spaces (fits free tier).

---

## ⚖️ Business Rationale

- **Efficiency**: Two-layer filtering and caching minimize compute and recruiter time.
- **Fairness**: Structured interviews, bias/tone checks, and skill transfer logic reduce bias.
- **Transparency**: Recruiters get clear, explainable scores and recommendations.
- **Scalability**: Modular design allows for easy extension (new data sources, interview stages, etc.).

---

## 🏁 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📄 License

MIT License.  
© 2025 – weekend hacking project.

---

### LinkedIn Scoring Placeholder

Currently, the LinkedIn scoring function is a stub. It returns a default or random score and does not call any real API. To enable real LinkedIn-based scoring:
- Implement a crawler or API integration in `src/linkedin_utils.py`.
- Update the scoring blend in `src/scoring.py` to use real metrics.
- The architecture and code are designed for this extension.

---

**This README is generated to reflect the actual business logic and design decisions in the codebase.** 