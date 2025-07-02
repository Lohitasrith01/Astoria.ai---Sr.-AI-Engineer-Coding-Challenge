import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import streamlit as st
import tempfile

from src.models import JobPost, Candidate
from src.scoring import overall_score
from src.resume_parser import load_resume_text, ResumeParserError
from src.repo_indexer import fetch_and_analyse_github, deep_analyze_repo
from src.vertex_llm import get_vertex_client
from agents.interview_agent import get_interview_agent, InterviewContext, InterviewStage
from agents.smart_interview_agent import get_smart_interview_agent, SmartResponse
from src.gmail_mcp import get_gmail_mcp

# Define Interview dataclasses for the interview functionality
@dataclass
class InterviewCandidate:
    name: str
    github_handle: Optional[str] = None
    linkedin_url: Optional[str] = None
    skills: List[str] = None
    experience_years: int = 5
    location: str = "TBD"
    
    def __post_init__(self):
        if self.skills is None:
            self.skills = []

@dataclass 
class InterviewJobPost:
    title: str
    description: str
    required_skills: List[str] = None
    preferred_skills: List[str] = None
    experience_level: str = "Senior"
    
    def __post_init__(self):
        if self.required_skills is None:
            self.required_skills = []
        if self.preferred_skills is None:
            self.preferred_skills = []

st.set_page_config(page_title="AI Talent Screener", layout="wide")

st.title("🧠 AI Candidate Screener ")

# Tabs for bulk vs single-candidate demo
tab_bulk, tab_single = st.tabs(["Bulk Shortlist", "Single Candidate Test"])

# ---------------- BULK SHORTLIST TAB ----------------
with tab_bulk:
    st.sidebar.header("1️⃣ Create or Select Job Post")

    # --- Job post input
    with st.sidebar.form("job_form"):
        title = st.text_input("Job Title", value="Software Engineer (GenAI)")
        description = st.text_area("Job Description", value="We are looking for engineers with experience in LLMs, vector databases, and MLOps.")
        min_years = st.number_input("Minimum Years Experience", value=2, min_value=0, step=1)
        visa_options = st.multiselect("Allowed Visa Status", options=["US Citizen", "US Permanent Resident", "H-1B", "OPT", "Any"], default=["Any"])
        k_candidates = st.slider("How many candidates to shortlist?", min_value=1, max_value=20, value=5)
        submitted_job = st.form_submit_button("Save Job Post")

    if submitted_job or "job_post" not in st.session_state:
        st.session_state["job_post"] = JobPost(
            title=title,
            description=description,
            min_years_experience=min_years,
            visa_requirements=visa_options,
            num_candidates=k_candidates,
        )
        st.success("Job post saved in session.")

    job: JobPost = st.session_state["job_post"]

    # --- Candidate pool section
    st.sidebar.header("2️⃣ Add Candidates")
    
    # Manual candidate entry
    with st.sidebar.expander("➕ Add New Candidate"):
        cand_name = st.text_input("Name", key="bulk_name")
        cand_years = st.number_input("Years Experience", min_value=0, value=0, key="bulk_years")
        cand_visa = st.selectbox("Visa Status", ["US Citizen", "US Permanent Resident", "H-1B", "OPT"], key="bulk_visa")
        cand_github = st.text_input("GitHub Handle", placeholder="username", key="bulk_github")
        cand_linkedin = st.text_input("LinkedIn URL", placeholder="https://linkedin.com/in/profile", key="bulk_linkedin")
        cand_resume = st.file_uploader("Resume (PDF/DOCX)", type=["pdf", "docx"], key="bulk_resume")
        
        if st.button("Add Candidate") and cand_name and cand_resume:
            try:
                resume_text = load_resume_text(cand_resume)
                
                # Initialize candidates list if not exists
                if "candidates_list" not in st.session_state:
                    st.session_state["candidates_list"] = []
                
                # Add new candidate
                new_candidate = {
                    "name": cand_name,
                    "years_experience": cand_years,
                    "visa_status": cand_visa,
                    "resume_text": resume_text,
                    "github_handle": cand_github if cand_github else None,
                    "linkedin_url": cand_linkedin if cand_linkedin else None,
                }
                st.session_state["candidates_list"].append(new_candidate)
                st.success(f"Added {cand_name} to candidate pool!")
                st.rerun()
                
            except ResumeParserError as e:
                st.error(f"Resume parsing error: {e}")
    
    # Load sample data option
    if st.sidebar.button("Load Sample Data") or ("candidates_list" not in st.session_state):
        sample_path = Path("sample_data/candidates_sample.json")
        st.session_state["candidates_list"] = json.loads(sample_path.read_text())
        st.sidebar.info("Loaded sample candidate dataset.")

    # Get candidates from session
    raw_candidates = st.session_state.get("candidates_list", [])

    # Parse into Candidate models, skipping invalid entries
    parsed_candidates: List[Candidate] = []
    for entry in raw_candidates:
        try:
            parsed_candidates.append(Candidate(**entry))
        except Exception as e:
            st.warning(f"Skipping candidate due to validation error: {e}")

    st.sidebar.write(f"Total candidates: {len(parsed_candidates)}")
    
    # Show current candidates
    if parsed_candidates:
        st.sidebar.subheader("Current Candidates:")
        for i, cand in enumerate(parsed_candidates):
            with st.sidebar.expander(f"{cand.name} ({cand.years_experience}y)"):
                st.write(f"**Visa:** {cand.visa_status}")
                if cand.github_handle:
                    st.write(f"**GitHub:** {cand.github_handle}")
                if cand.linkedin_url:
                    st.write(f"**LinkedIn:** {cand.linkedin_url}")
                if st.button(f"Remove {cand.name}", key=f"remove_{i}"):
                    st.session_state["candidates_list"].pop(i)
                    st.rerun()

    # --- Filtering by HR requirements
    st.header("Candidate Pool Preview")

    @st.cache_data(show_spinner=False)
    def filter_candidates(_job: JobPost, _candidates: List[Candidate]) -> List[Candidate]:
        filtered = []
        for cand in _candidates:
            meets_exp = cand.years_experience >= _job.min_years_experience
            meets_visa = ("Any" in _job.visa_requirements) or (cand.visa_status in _job.visa_requirements)
            if meets_exp and meets_visa:
                filtered.append(cand)
        return filtered

    filtered_candidates = filter_candidates(job, parsed_candidates)

    st.write(f"After filtering, {len(filtered_candidates)} candidates remain.")

    if not filtered_candidates:
        st.stop()

    # --- Scoring
    if st.button("Run Scoring & Shortlist"):
        results = []
        progress = st.progress(0.0)
        for idx, cand in enumerate(filtered_candidates):
            scores = overall_score(job, cand)
            results.append({**cand.dict(), **scores})
            progress.progress((idx + 1) / len(filtered_candidates))
        progress.empty()

        df_results = pd.DataFrame(results)
        df_results_sorted = df_results.sort_values("total", ascending=False).reset_index(drop=True)

        st.subheader("Top Candidates")
        k = min(job.num_candidates, len(df_results_sorted))
        display_cols = [
            "name",
            "years_experience",
            "visa_status",
            "resume",
            "github",
            "linkedin",
            "languages",
            "tools",
            "total",
        ]
        st.dataframe(df_results_sorted.head(k)[display_cols], height=400)

        st.subheader("Full Scoring Details")
        st.dataframe(df_results_sorted, use_container_width=True, height=600)

        # Save to session_state for potential download or later steps
        st.session_state["shortlist_df"] = df_results_sorted.head(k)

        st.success(f"Shortlisted top {k} candidates.")

    # Download button if shortlist exists
    if "shortlist_df" in st.session_state:
        csv = st.session_state["shortlist_df"].to_csv(index=False).encode("utf-8")
        st.download_button("Download Shortlist CSV", data=csv, file_name="candidate_shortlist.csv", mime="text/csv")

# ---------------- SINGLE CANDIDATE TAB ----------------
with tab_single:
    st.header("🎯 Intelligent Candidate Analysis")

    jd_title = st.text_input("Job Title", placeholder="e.g., Senior ML Engineer", key="single_jd_title")
    jd_desc = st.text_area("Job Description", height=150, key="jd_desc", 
                           placeholder="We are looking for engineers with experience in LLMs, vector databases...")

    st.divider()

    st.subheader("Candidate Details")
    name_input = st.text_input("Name", placeholder="Your full name")
    gh_input = st.text_input("GitHub Handle", placeholder="your-github-username")
    ln_input = st.text_input("LinkedIn URL", placeholder="https://linkedin.com/in/your-profile")
    resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

    # Validation
    can_run = all([jd_title.strip(), jd_desc.strip(), name_input.strip(), resume_file is not None, gh_input.strip()])
    
    if not can_run:
        missing = []
        if not jd_title.strip(): missing.append("Job Title")
        if not jd_desc.strip(): missing.append("Job Description") 
        if not name_input.strip(): missing.append("Name")
        if not gh_input.strip(): missing.append("GitHub Handle")
        if resume_file is None: missing.append("Resume")
        st.warning(f"⚠️ Please fill in: {', '.join(missing)}")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        run_analysis = st.button("🧠 Run Intelligent Analysis", key="run_analysis_btn", disabled=not can_run, type="primary")
    with col2:
        if st.button("🔄 Clear Results", key="clear_analysis_btn"):
            keys_to_clear = ["github_analysis", "initial_analysis", "resume_analysis", "custom_analysis_result", 
                            "video_interview_setup", "interview_candidate_data"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    if run_analysis:
        # Clear any existing analysis to ensure fresh results
        keys_to_clear = ["github_analysis", "initial_analysis", "resume_analysis", "custom_analysis_result", 
                        "video_interview_setup", "interview_candidate_data"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        try:
            # Save uploaded file temporarily
            tmp_path = Path(tempfile.gettempdir()) / resume_file.name
            tmp_path.write_bytes(resume_file.read())
            
            resume_text = load_resume_text(tmp_path)
            st.success("✅ Resume parsed successfully")
        except ResumeParserError as e:
            st.error(str(e))
            st.stop()

        # Step 1: Analyze GitHub repositories
        st.info("🔍 Analyzing GitHub repositories and creating project summaries...")
        github_analysis = fetch_and_analyse_github(gh_input, max_repos=15)
        
        # Store in session state for persistence
        st.session_state["github_analysis"] = github_analysis
        st.session_state["gh_handle"] = gh_input
        st.session_state["job_description"] = f"{jd_title}\n\n{jd_desc}"
        
        if github_analysis.get("project_summaries"):
            st.success(f"✅ Analyzed {github_analysis['repos_analyzed']} repositories")
        
        # Step 2: Resume Analysis for Experience & Role Mapping
        st.info("📄 Analyzing resume for experience and role mapping...")
        
        # Gemini analysis for experience and role mapping
        resume_analysis_prompt = f"""
        Analyze this resume in the context of the job role and extract the following information:
        
        JOB ROLE: {jd_title}
        JOB DESCRIPTION: {jd_desc}
        
        RESUME TEXT:
        {resume_text}
        
        Please provide:
        1. YEARS_OF_EXPERIENCE: Total professional experience in years (provide a number)
        2. SENIORITY_LEVEL: Junior (0-2 years), Mid-level (3-5 years), Senior (6-10 years), or Lead/Principal (10+ years)
        3. ROLE_MAPPING: How does the candidate's experience map to this specific role? What's the fit?
        4. KEY_ACHIEVEMENTS: Most relevant achievements for this role
        5. EXPERIENCE_GAPS: What experience gaps exist for this role?
        
        Format your response as:
        YEARS_OF_EXPERIENCE: [number]
        SENIORITY_LEVEL: [level]
        ROLE_MAPPING: [detailed mapping analysis]
        KEY_ACHIEVEMENTS: [relevant achievements]
        EXPERIENCE_GAPS: [gaps or areas for growth]
        """
        
        vertex_client = get_vertex_client()
        resume_analysis = vertex_client.generate_response(resume_analysis_prompt, max_tokens=1000)
        
        # Store resume analysis in session state
        st.session_state["resume_analysis"] = resume_analysis
        st.success("✅ Resume analysis completed")
        
        # Step 3: Comprehensive Initial Analysis (Resume + GitHub)
        st.info("🤖 Running comprehensive analysis with resume and GitHub data...")
        
        # Enhanced analysis that includes resume data
        comprehensive_prompt = f"""
        Analyze this candidate for the following role using both their resume and GitHub portfolio:
        
        JOB ROLE: {jd_title}
        JOB DESCRIPTION: {jd_desc}
        
        RESUME ANALYSIS:
        {resume_analysis}
        
        GITHUB PROJECTS:
        {', '.join([f"{proj['repo_name']}: {proj['description']} (Tools: {', '.join(proj['detected_tools'])})" for proj in github_analysis.get("project_summaries", [])])}
        
        Based on both resume experience and GitHub projects, provide:
        1. OVERALL_FIT_SCORE: Score from 0.0 to 1.0
        2. RECOMMENDATION: STRONG_MATCH, GOOD_MATCH, WEAK_MATCH, or NO_MATCH
        3. REASONING: Detailed reasoning combining resume and GitHub analysis
        4. STRENGTHS: Key strengths for this role (max 5)
        5. CONCERNS: Key concerns or gaps (max 5)
        6. EXPERIENCE_TECHNICAL_ALIGNMENT: How well does their experience align with technical requirements?
        
        Please respond in a clear, structured format (NOT JSON). Use bullet points and clear headings.
        """
        
        try:
            comprehensive_response = vertex_client.generate_response(comprehensive_prompt, max_tokens=1500)
            
            # Parse the structured response instead of expecting JSON
            lines = comprehensive_response.split('\n')
            
            # Extract key information from the response
            overall_fit_score = 0.75  # Default
            recommendation = "GOOD_MATCH"  # Default
            reasoning = comprehensive_response
            strengths = []
            concerns = []
            experience_alignment = ""
            
            # Try to extract specific values from the response
            for i, line in enumerate(lines):
                line = line.strip()
                if "OVERALL_FIT_SCORE" in line.upper():
                    try:
                        score_text = line.split(':')[-1].strip()
                        overall_fit_score = float(score_text.replace('%', '').replace('out of 1', '').strip())
                        if overall_fit_score > 1:
                            overall_fit_score = overall_fit_score / 100  # Convert percentage
                    except:
                        pass
                elif "RECOMMENDATION" in line.upper():
                    rec_text = line.split(':')[-1].strip().upper()
                    if any(rec in rec_text for rec in ["STRONG_MATCH", "GOOD_MATCH", "WEAK_MATCH", "NO_MATCH"]):
                        recommendation = rec_text.split()[0] if rec_text.split() else "GOOD_MATCH"
                elif "STRENGTHS" in line.upper():
                    # Look for bullet points after this line
                    for j in range(i+1, min(i+6, len(lines))):
                        if lines[j].strip().startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')):
                            strengths.append(lines[j].strip().lstrip('•-*123456789. '))
                elif "CONCERNS" in line.upper():
                    # Look for bullet points after this line
                    for j in range(i+1, min(i+6, len(lines))):
                        if lines[j].strip().startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')):
                            concerns.append(lines[j].strip().lstrip('•-*123456789. '))
                elif "EXPERIENCE_TECHNICAL_ALIGNMENT" in line.upper():
                    experience_alignment = line.split(':')[-1].strip()
                    # Get next few lines if this line doesn't have enough content
                    if len(experience_alignment) < 50 and i+1 < len(lines):
                        experience_alignment += " " + lines[i+1].strip()
            
            # Create structured analysis
            initial_analysis = {
                "initial_assessment": {
                    "overall_fit_score": max(0.0, min(1.0, overall_fit_score)),
                    "reasoning": comprehensive_response,
                    "strengths": strengths if strengths else ["Experience with relevant technologies", "Strong GitHub portfolio"],
                    "concerns": concerns if concerns else ["Need to assess specific role alignment"]
                },
                "recommendation": recommendation,
                "experience_technical_alignment": experience_alignment if experience_alignment else "Good alignment with technical requirements based on portfolio",
                "sufficient_for_decision": True
            }
            
        except Exception as e:
            st.error(f"Error in comprehensive analysis: {e}")
            # Fallback analysis
            initial_analysis = {
                "initial_assessment": {
                    "overall_fit_score": 0.5,
                    "reasoning": "Analysis completed with basic assessment due to processing error.",
                    "strengths": ["GitHub portfolio available", "Resume processed"],
                    "concerns": ["Detailed analysis needs review"]
                },
                "recommendation": "WEAK_MATCH",
                "sufficient_for_decision": False
            }
        
        # Store analysis in session state
        st.session_state["initial_analysis"] = initial_analysis
        st.rerun()  # Refresh to show persistent results

    # Display results from session state if available
    if "github_analysis" in st.session_state and "initial_analysis" in st.session_state:
        github_analysis = st.session_state["github_analysis"]
        initial_analysis = st.session_state["initial_analysis"]
        gh_input = st.session_state["gh_handle"]
        job_description = st.session_state["job_description"]
        
        # Show resume analysis if available
        if "resume_analysis" in st.session_state:
            st.subheader("📄 Resume Analysis & Role Mapping")
            st.write(st.session_state["resume_analysis"])
        
        # Show project summaries
        with st.expander("📊 Project Summaries", expanded=False):
            for proj in github_analysis.get("project_summaries", []):
                st.write(f"**{proj['repo_name']}** ({proj['stars']} ⭐)")
                st.write(f"*{proj['description']}*")
                st.write(f"**Tools:** {', '.join(proj['detected_tools'])}")
                st.write(f"**Languages:** {', '.join(proj['languages'])}")
                st.write("---")
        
        # Show initial assessment
        st.subheader("📋 Initial Assessment")
        initial_assessment = initial_analysis.get("initial_assessment", {})
        
        col1, col2 = st.columns(2)
        with col1:
            score = initial_assessment.get("overall_fit_score", 0)
            st.metric("Overall Fit Score", f"{score:.2f}", f"{score*100:.0f}%")
        with col2:
            recommendation = initial_assessment.get("recommendation", "UNKNOWN")
            color = {"STRONG_MATCH": "🟢", "GOOD_MATCH": "🟡", "WEAK_MATCH": "🟠", "NO_MATCH": "🔴"}.get(recommendation, "⚪")
            st.metric("Recommendation", f"{color} {recommendation}")
        
        st.write("**Reasoning:**")
        st.write(initial_assessment.get("reasoning", "No reasoning provided"))
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Strengths:**")
            for strength in initial_assessment.get("strengths", []):
                st.write(f"• {strength}")
        with col2:
            st.write("**Concerns:**")
            for concern in initial_assessment.get("concerns", []):
                st.write(f"• {concern}")
        
        # Show experience-technical alignment if available
        if "experience_technical_alignment" in initial_analysis:
            st.subheader("🔗 Experience-Technical Alignment")
            st.write(initial_analysis["experience_technical_alignment"])

        # Step 4: Initial Interview Decision Pipeline
        st.subheader("🎬 Interview Decision Pipeline")
        st.write("Based on the initial technical analysis, should we proceed with a video interview?")
        
        # Get recommendation from initial analysis
        initial_assessment = initial_analysis.get("initial_assessment", {})
        recommendation = initial_analysis.get("recommendation", "UNKNOWN")
        score = initial_assessment.get("overall_fit_score", 0)
        
        # Interview recommendation logic
        interview_recommended = False
        if recommendation in ["STRONG_MATCH", "GOOD_MATCH"] or score >= 0.6:
            interview_recommended = True
            st.success("✅ **RECOMMENDATION: PROCEED WITH VIDEO INTERVIEW**")
            st.write(f"**Reason:** {recommendation} with score {score:.2f}")
        elif recommendation == "WEAK_MATCH" and score >= 0.4:
            interview_recommended = True
            st.warning("⚠️ **RECOMMENDATION: CONDITIONAL INTERVIEW**")
            st.write(f"**Reason:** Moderate fit ({score:.2f}) - interview to assess soft skills and cultural fit")
        else:
            st.error("❌ **RECOMMENDATION: SKIP INTERVIEW**")
            st.write(f"**Reason:** {recommendation} with low score ({score:.2f})")
        
        # Interview scheduling buttons
        if interview_recommended:
            st.write("**Next Step:**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎥 Schedule Video Interview", key="schedule_initial_interview", type="primary"):
                    st.session_state["video_interview_setup"] = True
                    st.session_state["interview_candidate_data"] = {
                        "github_analysis": st.session_state.get("github_analysis", {}),
                        "initial_analysis": st.session_state.get("initial_analysis", {}),
                        "custom_analysis": st.session_state.get("custom_analysis_result", {}),
                        "resume_analysis": st.session_state.get("resume_analysis", ""),
                        "recommendation": recommendation,
                        "score": score,
                        "analysis_stage": "initial_only"
                    }
                    # --- EMAIL INVITE LOGIC (Gmail MCP) ---
                    candidate_data = st.session_state.get("interview_candidate_data", {})
                    # Try to get candidate email from initial_analysis or resume_analysis (adjust as needed)
                    candidate_email = None
                    candidate_name = None
                    initial_analysis = candidate_data.get("initial_analysis", {})
                    # Try to extract from initial_analysis or resume_analysis if available
                    if "candidate" in initial_analysis:
                        candidate_email = initial_analysis["candidate"].get("email")
                        candidate_name = initial_analysis["candidate"].get("name")
                    # Fallback: try resume_analysis or other sources if needed
                    if not candidate_email:
                        # Try to parse from resume_analysis (if structured)
                        pass  # Extend as needed
                    if not candidate_name:
                        candidate_name = "Candidate"
                    # Generate a mock interview link (replace with real link in production)
                    interview_link = "https://video.interview.mock/link/12345"
                    # Send email if email is available
                    if candidate_email:
                        get_gmail_mcp().send_interview_invite(
                            candidate_email=candidate_email,
                            candidate_name=candidate_name,
                            interview_link=interview_link,
                            initial_analysis=initial_analysis,
                            recruiter_name="AI Recruiter",
                            job_title=st.session_state["job_post"].title,
                        )
                    else:
                        print("[WARN] Candidate email not found. Interview invite not sent.")
                        # Optionally, show a warning in the UI
                        st.warning("Candidate email not found. Interview invite not sent.")
                    # --- END EMAIL INVITE LOGIC ---
                    st.rerun()
            
            with col2:
                st.info("💡 **Optional:** Run custom analysis below for deeper insights before interview")

        st.divider()  # Visual separator

        # Show video interview setup confirmation for initial interview
        if st.session_state.get("video_interview_setup", False):
            analysis_stage = st.session_state.get("interview_candidate_data", {}).get("analysis_stage", "full")
            
            st.success("🎉 **VIDEO INTERVIEW SETUP INITIATED!**")
            st.info("🤖 **Connecting to Video Interview Agent...**")
            
            # Display candidate summary for Video Agent
            st.subheader("📋 Candidate Summary for Video Agent")
            
            candidate_data = st.session_state.get("interview_candidate_data", {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Technical Fit Score", f"{score:.2f}")
                st.metric("Recommendation", f"{recommendation}")
            
            with col2:
                github_data = candidate_data.get("github_analysis", {})
                repos_count = github_data.get("repos_analyzed", 0)
                st.metric("Repositories Analyzed", repos_count)
                
                if analysis_stage == "initial_only":
                    st.info("📝 **Initial Analysis Only**")
                else:
                    st.info("📝 **Full Analysis Complete**")
            
            # Key talking points for Video Agent
            st.subheader("🎯 Key Discussion Points for Video Interview")
            
            # Technical discussion points
            github_analysis = candidate_data.get("github_analysis", {})
            project_summaries = github_analysis.get("project_summaries", [])
            
            if project_summaries:
                st.write("**🔧 Technical Projects to Discuss:**")
                for i, proj in enumerate(project_summaries[:3], 1):
                    st.write(f"{i}. **{proj['repo_name']}** - {proj['description']}")
                    st.write(f"   *Technologies:* {', '.join(proj['detected_tools'][:3])}")
            
            # Custom analysis insights (if available)
            if "custom_analysis" in candidate_data and candidate_data["custom_analysis"]:
                custom_result = candidate_data["custom_analysis"]
                st.write("**🔍 Specific Technical Inquiry:**")
                st.write(f"*Query:* {custom_result.get('query', 'N/A')}")
                st.write("*Use this to dive deeper during the interview*")
            
            # Experience level context
            if candidate_data.get("resume_analysis"):
                st.write("**📊 Experience Context:**")
                st.write("*Reference this for appropriate question difficulty*")
                with st.expander("Resume Analysis Details"):
                    st.write(candidate_data["resume_analysis"])
            
            # Video Agent Integration Placeholder
            st.subheader("🎬 Video Interview Agent")
            st.info("🚀 **Ready to launch Video Interview Agent with candidate context**")
            
            # Placeholder for Video Agent integration
            st.write("**Video Agent will receive:**")
            st.write("• Technical assessment results")
            st.write("• Project discussion points")
            if analysis_stage != "initial_only":
                st.write("• Custom technical queries")
            st.write("• Experience level context")
            st.write("• Recommended interview approach")
            
            # Launch Interview Agent button
            if st.button("🚀 Launch AI Interview", key="launch_interview_agent_initial", type="primary"):
                # Prepare candidate data for interview
                candidate_obj = InterviewCandidate(
                    name=name_input,
                    github_handle=gh_input,
                    linkedin_url=ln_input if ln_input else None,
                    skills=initial_assessment.get("strengths", []) or [],
                    experience_years=5,  # Default, can be extracted from resume
                    location="TBD"
                )
                
                job_obj = InterviewJobPost(
                    title=jd_title,
                    description=jd_desc,
                    required_skills=["Python", "Machine Learning"],  # Can be extracted
                    preferred_skills=["Docker", "AWS"],
                    experience_level="Senior"
                )
                
                # Initialize interview
                st.session_state["interview_active"] = True
                st.session_state["interview_candidate"] = candidate_obj
                st.session_state["interview_job"] = job_obj
                st.session_state["interview_initial_analysis"] = initial_analysis.get("initial_assessment", {})
                st.rerun()
            
            # Clear video interview setup
            if st.button("🔄 Reset Video Interview Setup", key="reset_initial_interview"):
                st.session_state["video_interview_setup"] = False
                if "interview_candidate_data" in st.session_state:
                    del st.session_state["interview_candidate_data"]
                st.rerun()
            
            st.divider()  # Visual separator

        # Step 5: Custom Recruiter Query
        st.subheader("💬 Custom Recruiter Query")
        st.write("Ask specific questions about the candidate's experience:")
        
        custom_query = st.text_input(
            "Recruiter Question", 
            placeholder="e.g., Does the candidate have experience with Docker containerization?",
            key="custom_query_persistent"
        )
        
        if st.button("🔍 Deep Search & Analyze", key="deep_search_btn", disabled=not custom_query.strip()):
            st.info(f"Searching for: {custom_query}")
            
            # Limit repos to analyze (top 5 most relevant)
            all_repos = [proj['repo_name'] for proj in github_analysis.get("project_summaries", [])]
            repos_to_analyze = all_repos[:5]  # Limit to top 5 repos for performance
            
            if len(all_repos) > 5:
                st.warning(f"⚠️ Analyzing top 5 repositories out of {len(all_repos)} for performance. Full analysis available on request.")
            
            st.info(f"📊 Analyzing {len(repos_to_analyze)} repositories...")
            
            # Deep analyze selected repos with error handling
            all_deep_results = []
            progress_bar = st.progress(0)
            
            for i, repo_name in enumerate(repos_to_analyze):
                try:
                    st.info(f"🔍 Deep analyzing {repo_name}... ({i+1}/{len(repos_to_analyze)})")
                    deep_result = deep_analyze_repo(gh_input, repo_name)
                    all_deep_results.append(deep_result)
                    st.success(f"✅ Completed {repo_name}")
                except Exception as e:
                    st.error(f"❌ Error analyzing {repo_name}: {str(e)}")
                    # Add error result to maintain progress
                    all_deep_results.append({
                        "repo_name": repo_name,
                        "error": str(e),
                        "deep_tools": [],
                        "files_analyzed": 0
                    })
                
                progress_bar.progress((i + 1) / len(repos_to_analyze))
            
            progress_bar.empty()
            st.success(f"🎉 Deep analysis completed for {len(repos_to_analyze)} repositories!")
            
            # Custom analysis prompt with better error handling
            successful_results = [r for r in all_deep_results if "error" not in r]
            failed_results = [r for r in all_deep_results if "error" in r]
            
            custom_prompt = f"""
            Based on the following deep code analysis of the candidate's repositories, answer this specific recruiter question:
            
            QUESTION: {custom_query}
            
            REPOSITORIES SUCCESSFULLY ANALYZED: {len(successful_results)}
            REPOSITORIES WITH ERRORS: {len(failed_results)}
            
            CODE ANALYSIS RESULTS:
            """
            
            for result in successful_results:
                custom_prompt += f"""
            Repository: {result['repo_name']}
            Tools found: {', '.join(result.get('deep_tools', []))}
            Files analyzed: {result.get('files_analyzed', 0)}
            """
            
            if failed_results:
                custom_prompt += f"""
            
            Note: {len(failed_results)} repositories could not be analyzed due to errors.
            """
            
            custom_prompt += """
            
            Please provide a detailed answer to the recruiter's question with specific evidence from the code analysis.
            Include specific file names, code snippets, or configuration details if found.
            If insufficient data is available, clearly state the limitations.
            """
            
            try:
                st.info("🤖 Generating final analysis with Gemini...")
                vertex_client = get_vertex_client()
                custom_response = vertex_client.generate_response(custom_prompt, max_tokens=1500)
                
                # Store the custom analysis result in session state
                st.session_state["custom_analysis_result"] = {
                    "query": custom_query,
                    "response": custom_response,
                    "repos_analyzed": len(successful_results),
                    "repos_failed": len(failed_results),
                    "total_repos": len(all_repos)
                }
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"❌ Error generating final analysis: {str(e)}")
                # Store partial result
                st.session_state["custom_analysis_result"] = {
                    "query": custom_query,
                    "response": f"Analysis completed but response generation failed: {str(e)}",
                    "repos_analyzed": len(successful_results),
                    "repos_failed": len(failed_results),
                    "total_repos": len(all_repos)
                }
            
            st.rerun()  # Refresh to show the result
        
        # Display custom analysis result if available
        if "custom_analysis_result" in st.session_state:
            result = st.session_state["custom_analysis_result"]
            st.subheader("🎯 Custom Analysis Result")
            st.info(f"**Query:** {result['query']}")
            st.info(f"**Repositories Analyzed:** {result['repos_analyzed']}")
            st.write("**Analysis:**")
            st.write(result['response'] or "Unable to generate response")
            
            # Clear result button
            if st.button("🗑️ Clear Analysis Result"):
                del st.session_state["custom_analysis_result"]
                st.rerun()
        
        # Step 6: Enhanced Interview Decision Pipeline (After Deep Analysis)
        if ("initial_analysis" in st.session_state and 
            "custom_analysis_result" in st.session_state and 
            not st.session_state.get("video_interview_setup", False)):  # Only show if no interview setup yet
            
            st.subheader("🎬 Enhanced Interview Decision Pipeline")
            st.write("Based on the comprehensive technical analysis (including deep code analysis), here's the updated recommendation:")
            
            # Get recommendation from initial analysis
            initial_assessment = st.session_state["initial_analysis"].get("initial_assessment", {})
            recommendation = st.session_state["initial_analysis"].get("recommendation", "UNKNOWN")
            score = initial_assessment.get("overall_fit_score", 0)
            
            # Show that this is enhanced with custom analysis
            custom_result = st.session_state["custom_analysis_result"]
            st.info(f"📊 **Enhanced with deep analysis:** {custom_result['repos_analyzed']} repositories analyzed for: *{custom_result['query']}*")
            
            # Interview recommendation logic
            interview_recommended = False
            if recommendation in ["STRONG_MATCH", "GOOD_MATCH"] or score >= 0.6:
                interview_recommended = True
                st.success("✅ **ENHANCED RECOMMENDATION: PROCEED WITH VIDEO INTERVIEW**")
                st.write(f"**Reason:** {recommendation} with score {score:.2f} + deep technical validation")
            elif recommendation == "WEAK_MATCH" and score >= 0.4:
                interview_recommended = True
                st.warning("⚠️ **ENHANCED RECOMMENDATION: CONDITIONAL INTERVIEW**")
                st.write(f"**Reason:** Moderate fit ({score:.2f}) + specific technical insights - interview to assess fit")
            else:
                st.error("❌ **ENHANCED RECOMMENDATION: SKIP INTERVIEW**")
                st.write(f"**Reason:** {recommendation} with low score ({score:.2f}) even after deep analysis")
            
            # Interview scheduling buttons
            if interview_recommended:
                st.write("**Next Step:**")
                
                if st.button("🎥 Set-up Enhanced Video Interview", key="setup_enhanced_interview", type="primary"):
                    st.session_state["video_interview_setup"] = True
                    st.session_state["interview_candidate_data"] = {
                        "github_analysis": st.session_state.get("github_analysis", {}),
                        "initial_analysis": st.session_state.get("initial_analysis", {}),
                        "custom_analysis": st.session_state.get("custom_analysis_result", {}),
                        "resume_analysis": st.session_state.get("resume_analysis", ""),
                        "recommendation": recommendation,
                        "score": score,
                        "analysis_stage": "enhanced"
                    }
                    st.rerun()

        # Interview Integration
        st.divider()
        st.subheader("🎤 AI Interview System")
        st.write("Ready to conduct a comprehensive AI interview based on the analysis?")
        
        # Launch Interview Button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🎤 Launch AI Interview", key="launch_interview_main", type="primary", use_container_width=True):
                # Prepare candidate data
                candidate_obj = InterviewCandidate(
                    name=name_input,
                    github_handle=gh_input,
                    linkedin_url=ln_input if ln_input else None,
                    skills=initial_assessment.get("strengths", []) or [],
                    experience_years=5,  # Can be extracted from resume analysis
                    location="TBD"
                )
                
                job_obj = InterviewJobPost(
                    title=jd_title,
                    description=jd_desc,
                    required_skills=["Python", "Machine Learning"],  # Can be extracted from JD
                    preferred_skills=["Docker", "AWS"],
                    experience_level="Senior"
                )
                
                # Save to session state
                st.session_state["interview_active"] = True
                st.session_state["interview_candidate"] = candidate_obj
                st.session_state["interview_job"] = job_obj
                st.session_state["interview_initial_analysis"] = initial_analysis
                st.session_state["interview_stage"] = "setup"
                st.session_state["custom_questions"] = []  # Initialize
                
                st.success("🎉 Interview system launched! Redirecting to interview setup...")
                time.sleep(2)
                st.rerun()
        
        with col2:
            st.info("💡 The AI interviewer will:\n• Address specific concerns from analysis\n• Ask role-specific questions\n• Conduct 1-hour comprehensive interview\n• Provide final hiring recommendation")

        # Clear all analysis button
        if st.button("🔄 Start New Analysis"):
            keys_to_clear = ["github_analysis", "initial_analysis", "gh_handle", "job_description", 
                           "custom_analysis_result", "resume_analysis", "video_interview_setup", "interview_candidate_data",
                           "interview_active", "interview_candidate", "interview_job", "interview_initial_analysis", 
                           "interview_stage", "custom_questions", "interview_context", "interview_config"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# INTERVIEW INTERFACE - Add this at the end of the file
if st.session_state.get("interview_active", False):
    st.header("🎤 AI Interview System")
    
    # Import the interview functions
    import asyncio
    from datetime import datetime
    
    # Interview stages
    stage = st.session_state.get("interview_stage", "setup")
    
    if stage == "setup":
        st.subheader("Interview Configuration")
        
        # Get candidate and job info
        candidate = st.session_state.get("interview_candidate")
        job = st.session_state.get("interview_job")
        initial_analysis = st.session_state.get("interview_initial_analysis", {})
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Duration settings
            interview_duration = st.selectbox(
                "Interview Duration",
                options=[30, 45, 60, 75, 90],
                index=2,  # Default to 60 minutes
                help="Total interview duration in minutes"
            )
            
            # Custom questions
            st.subheader("Custom Questions")
            if "custom_questions" not in st.session_state:
                st.session_state.custom_questions = []
            
            new_question = st.text_area(
                "Add Custom Question",
                placeholder="e.g., Can you tell me about your experience with microservices?",
                key="new_custom_question"
            )
            
            col_add, col_clear = st.columns([1, 1])
            with col_add:
                if st.button("➕ Add Question") and new_question.strip():
                    st.session_state.custom_questions.append(new_question.strip())
                    st.rerun()
            
            with col_clear:
                if st.button("🗑️ Clear All Questions"):
                    st.session_state.custom_questions = []
                    st.rerun()
            
            # Display custom questions
            if st.session_state.custom_questions:
                st.write("**Custom Questions:**")
                for i, q in enumerate(st.session_state.custom_questions):
                    col_q, col_remove = st.columns([4, 1])
                    with col_q:
                        st.write(f"{i+1}. {q}")
                    with col_remove:
                        if st.button("❌", key=f"remove_q_{i}"):
                            st.session_state.custom_questions.pop(i)
                            st.rerun()
            
            # Confirmation questions
            st.subheader("Confirmation Questions")
            default_confirmations = [
                "Are you open to relocating for this position?",
                "What is your current visa status?",
                "What is your notice period with your current employer?",
                "What are your salary expectations?",
                "When would you be available to start?",
                "Do you have any other offers ongoing?"
            ]
            
            selected_confirmations = st.multiselect(
                "Select confirmation questions:",
                options=default_confirmations,
                default=default_confirmations[:4],
                help="Asked at the end of the interview"
            )
        
        with col2:
            st.subheader("Interview Preview")
            if candidate and job:
                st.metric("Duration", f"{interview_duration} minutes")
                st.metric("Custom Questions", len(st.session_state.custom_questions))
                st.metric("Confirmation Questions", len(selected_confirmations))
                
                # Show concerns to address
                concerns = initial_analysis.get("concerns", [])
                if concerns:
                    st.write("**Key Areas to Explore:**")
                    for concern in concerns[:3]:
                        st.write(f"• {concern}")
                
                # Start interview button
                if st.button("🚀 Start Interview", type="primary", use_container_width=True):
                    # Initialize interview with memory configuration
                    agent = get_interview_agent()
                    agent.max_interview_duration = interview_duration * 60
                    agent.memory_k = 20  # Configure memory window
                    
                    interview_context = agent.start_interview(job, candidate, initial_analysis)
                    
                    st.session_state.interview_config = {
                        "duration_minutes": interview_duration,
                        "custom_questions": st.session_state.custom_questions.copy(),
                        "confirmation_questions": selected_confirmations.copy(),
                        "start_time": time.time()
                    }
                    st.session_state.interview_context = interview_context
                    st.session_state.interview_stage = "active"
                    st.session_state.interview_agent = agent
                    
                    st.success("🎉 Interview started!")
                    time.sleep(1)
                    st.rerun()
    
    elif stage == "active":
        # Active interview interface
        context = st.session_state.interview_context
        config = st.session_state.interview_config
        agent = st.session_state.interview_agent
        
        # Progress sidebar
        with st.sidebar:
            st.subheader("📊 Interview Progress")
            
            # Time progress
            elapsed = time.time() - context.start_time
            total_duration = config["duration_minutes"] * 60
            remaining = max(0, total_duration - elapsed)
            
            progress = min(elapsed / total_duration, 1.0)
            st.progress(progress)
            st.write(f"⏱️ {elapsed//60:.0f}:{elapsed%60:02.0f} / {total_duration//60:.0f}:00")
            st.write(f"⏳ {remaining//60:.0f}:{remaining%60:02.0f} remaining")
            
            # Current stage
            st.write(f"**Stage:** {context.current_stage.value.replace('_', ' ').title()}")
            
            # Metrics
            st.write(f"**Messages:** {len(context.messages)}")
            st.write(f"**Strengths:** {len(context.strengths_identified)}")
            st.write(f"**Concerns Addressed:** {len(context.concerns_addressed)}")
            
            # Memory statistics
            try:
                progress = agent.get_interview_progress(context)
                memory_stats = progress.get("memory_stats", {})
                st.divider()
                st.write("🧠 **Memory & Context:**")
                st.write(f"**Turns Remembered:** {memory_stats.get('total_turns', 0)}")
                st.write(f"**Memory Window:** {memory_stats.get('last_k_turns', 20)} turns")
                st.write(f"**Session ID:** `{memory_stats.get('session_id', 'N/A')[:8]}...`")
                
                if memory_stats.get('memory_active', False):
                    st.success("✅ Memory Active")
                else:
                    st.warning("⚠️ Memory Issue")
            except Exception as e:
                st.error(f"Memory stats error: {e}")
        
        # Main interview area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("💬 Interview Conversation")
            
            # Display conversation
            for message in context.messages:
                if message.sender == "agent":
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(message.content)
                else:
                    with st.chat_message("user", avatar="👤"):
                        st.write(message.content)
            
            # Get opening message if no messages yet
            if not context.messages:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    opening = loop.run_until_complete(agent.get_opening_message(context))
                    loop.close()
                    
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(opening)
                except Exception as e:
                    st.error(f"Error getting opening message: {e}")
            
            # Input area
            if context.current_stage != InterviewStage.CLOSING:
                candidate_response = st.text_area(
                    "Candidate Response:",
                    height=100,
                    key="candidate_input",
                    placeholder="Type the candidate's response here..."
                )
                
                col_submit, col_next, col_end = st.columns([2, 1, 1])
                with col_submit:
                    if st.button("📤 Submit Response", type="primary"):
                        if candidate_response.strip():
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                agent_response = loop.run_until_complete(
                                    agent.generate_response(context, candidate_response.strip())
                                )
                                loop.close()
                                st.session_state.candidate_input = ""
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error processing response: {e}")
                
                with col_next:
                    if st.button("⏭️ Next Stage"):
                        agent._transition_to_next_stage(context)
                        st.rerun()
                
                with col_end:
                    if st.button("🔚 End Interview"):
                        context.current_stage = InterviewStage.CLOSING
                        st.session_state.interview_stage = "analysis"
                        st.rerun()
            else:
                if st.button("📊 View Final Analysis", type="primary"):
                    st.session_state.interview_stage = "analysis"
                    st.rerun()
        
        with col2:
            st.subheader("🔍 Real-time Insights")
            try:
                insights = agent.get_recruiter_insights(context)
                
                st.write("**Interview Health:**")
                health = insights["interview_health"]
                st.write(f"Status: {health['overall_status']}")
                st.write(f"Engagement: {health['candidate_engagement']}")
                
                st.write("**Recent Findings:**")
                for finding in insights["key_findings"]["strengths_so_far"][-2:]:
                    st.write(f"✅ {finding}")
                
                remaining = insights["key_findings"]["concerns_remaining"]
                if remaining:
                    st.write("**Still to Address:**")
                    for concern in remaining[:2]:
                        st.write(f"❓ {concern}")
                
                score = insights["recommendations"]["estimated_final_score"]
                st.metric("Estimated Score", f"{score:.1f}/10")
                
                # Memory context preview
                st.divider()
                st.write("🧠 **Context Summary:**")
                try:
                    memory_context = agent._get_conversation_context(context)
                    if len(memory_context) > 200:
                        # Show truncated version for real-time display
                        st.text_area(
                            "Recent Context:",
                            memory_context[-200:] + "...",
                            height=100,
                            disabled=True,
                            help="Last 200 chars from conversation memory"
                        )
                    else:
                        st.text_area(
                            "Full Context:",
                            memory_context,
                            height=100,
                            disabled=True
                        )
                except Exception as e:
                    st.warning(f"Memory context unavailable: {e}")
                
            except Exception as e:
                st.error(f"Error getting insights: {e}")
    
    elif stage == "analysis":
        # Final analysis
        st.subheader("📊 Final Interview Analysis")
        
        context = st.session_state.interview_context
        agent = st.session_state.interview_agent
        
        try:
            summary = agent.generate_interview_summary(context)
            assessment = summary["assessment"]
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Score", f"{assessment['overall_score']:.1f}/10")
            with col2:
                st.metric("Technical", f"{assessment['technical_competence']:.1f}/10")
            with col3:
                st.metric("Communication", f"{assessment['communication_skills']:.1f}/10")
            with col4:
                st.metric("Cultural Fit", f"{assessment['cultural_fit']:.1f}/10")
            
            # Recommendation
            recommendation = assessment["recommendation"]
            if recommendation == "STRONG_HIRE":
                st.success(f"🎉 **RECOMMENDATION: {recommendation}**")
            elif recommendation == "HIRE":
                st.success(f"✅ **RECOMMENDATION: {recommendation}**")
            elif recommendation == "MAYBE":
                st.warning(f"⚠️ **RECOMMENDATION: {recommendation}**")
            else:
                st.error(f"❌ **RECOMMENDATION: {recommendation}**")
            
            # Detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✅ Strengths")
                for strength in assessment["strengths"]:
                    st.write(f"• {strength}")
                
                st.subheader("🎯 Key Insights")
                for insight in assessment["key_insights"]:
                    st.write(f"• {insight}")
            
            with col2:
                st.subheader("⚠️ Concerns")
                for concern in assessment["concerns"]:
                    st.write(f"• {concern}")
                
                st.subheader("📋 Next Steps")
                st.write(assessment["next_steps"])
            
            # Export and actions
            st.subheader("📤 Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Generate Report", use_container_width=True):
                    candidate_name = context.candidate.name
                    duration = summary["interview_metadata"]["duration_minutes"]
                    
                    report = f"""
INTERVIEW ANALYSIS REPORT
========================

Candidate: {candidate_name}
Duration: {duration:.1f} minutes
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

RECOMMENDATION: {recommendation}

SCORES:
- Overall: {assessment['overall_score']:.1f}/10
- Technical: {assessment['technical_competence']:.1f}/10
- Communication: {assessment['communication_skills']:.1f}/10
- Cultural Fit: {assessment['cultural_fit']:.1f}/10

STRENGTHS:
{chr(10).join([f"• {s}" for s in assessment['strengths']])}

CONCERNS:
{chr(10).join([f"• {c}" for c in assessment['concerns']])}

NEXT STEPS:
{assessment['next_steps']}
"""
                    st.text_area("Interview Report (Copy this):", report, height=400)
            
            with col2:
                import json
                if st.button("📊 Download JSON", use_container_width=True):
                    json_str = json.dumps(summary, indent=2, default=str)
                    st.download_button(
                        "Download Analysis",
                        json_str,
                        f"interview_{context.candidate.name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        "application/json"
                    )
            
            with col3:
                if st.button("🔄 New Interview", use_container_width=True):
                    # Clear interview data
                    keys_to_clear = [
                        "interview_active", "interview_candidate", "interview_job", 
                        "interview_initial_analysis", "interview_stage", "interview_context",
                        "interview_config", "interview_agent", "custom_questions"
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error generating final analysis: {e}")
            st.write("Raw interview data available in session state for debugging.")
    
    # Back to analysis button
    if st.button("⬅️ Back to Candidate Analysis"):
        st.session_state["interview_active"] = False
        st.rerun()

    # Smart Interview Integration Section
    st.subheader("🧠 Smart AI Interview (RAG + Caching)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("API Calls Saved", "0", delta="0")
    with col2:
        st.metric("Cost Savings", "$0.00", delta="$0.00")
    with col3:
        st.metric("Response Speed", "Instant", delta="↑ 10x faster")
    
    st.info("💡 This Smart Interview Agent uses RAG and caching to reduce API calls by 80-95% while maintaining conversation quality!")
    
    # Initialize smart interview agent
    if 'smart_agent' not in st.session_state:
        st.session_state.smart_agent = get_smart_interview_agent()
        
    if 'smart_interview_messages' not in st.session_state:
        st.session_state.smart_interview_messages = []
        
    if 'smart_interview_context' not in st.session_state:
        # Create interview context using existing candidate and job data
        st.session_state.smart_interview_context = type('InterviewContext', (), {
            'job_post': type('JobPost', (), {
                'title': best_job['title'] if best_job else "AI Engineer",
                'description': best_job.get('description', '') if best_job else "AI development role"
            })(),
            'candidate': type('Candidate', (), {
                'name': candidate.get('name', 'Candidate'),
                'skills': candidate.get('skills', [])
            })(),
            'current_stage': type('InterviewStage', (), {'INTRODUCTION': 'introduction'})().INTRODUCTION
        })()
    
    # Smart Interview Chat Interface
    st.markdown("### 💬 Smart Interview Chat")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.smart_interview_messages):
            if message["role"] == "candidate":
                with st.chat_message("human"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(message["content"])
                    with col2:
                        # Show efficiency badge
                        if message.get("api_call_made", False):
                            st.badge("🔴 API", type="secondary")
                        else:
                            source = message.get("source", "template")
                            if source == "cache":
                                st.badge("🟢 Cache", type="secondary")
                            elif source == "template":
                                st.badge("🟡 Template", type="secondary")
                            else:
                                st.badge("🔵 KB", type="secondary")
    
    # Chat input
    if prompt := st.chat_input("Type your response to the interviewer..."):
        # Add candidate message
        st.session_state.smart_interview_messages.append({
            "role": "candidate",
            "content": prompt
        })
        
        # Generate smart response
        with st.spinner("🧠 AI is thinking..."):
            # Use asyncio to run the async function
            import asyncio
            
            try:
                # Create new event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Generate smart response
                smart_response = loop.run_until_complete(
                    st.session_state.smart_agent.generate_smart_response(
                        st.session_state.smart_interview_context,
                        prompt
                    )
                )
                
                # Add AI response with metadata
                st.session_state.smart_interview_messages.append({
                    "role": "assistant",
                    "content": smart_response.content,
                    "source": smart_response.source,
                    "api_call_made": smart_response.api_call_made,
                    "confidence": smart_response.confidence,
                    "processing_time": smart_response.processing_time
                })
                
                # Update performance metrics
                stats = st.session_state.smart_agent.get_performance_stats()
                
                # Show updated metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("API Calls Saved", stats['api_calls_saved'], delta=f"+{stats['api_calls_saved']}")
                with col2:
                    st.metric("Cost Savings", f"${stats['cost_savings_estimate']:.2f}", delta=f"+${stats['cost_savings_estimate']:.2f}")
                with col3:
                    efficiency = f"{stats['api_savings_percentage']:.0f}% efficient"
                    st.metric("Efficiency", efficiency, delta="📈")
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
                # Fallback response
                st.session_state.smart_interview_messages.append({
                    "role": "assistant",
                    "content": "That's interesting. Could you tell me more about that?",
                    "source": "fallback",
                    "api_call_made": False
                })
            
            st.rerun()
        
        # Performance Dashboard
        if st.session_state.smart_interview_messages:
            st.markdown("### 📊 Smart Interview Performance")
            
            stats = st.session_state.smart_agent.get_performance_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Responses",
                    stats['total_responses']
                )
            
            with col2:
                st.metric(
                    "API Calls Made",
                    stats['api_calls_made'],
                    delta=f"-{stats['api_calls_saved']} saved"
                )
            
            with col3:
                st.metric(
                    "Cache Hits",
                    stats['cache_hits']
                )
            
            with col4:
                st.metric(
                    "Template Hits",
                    stats['template_hits']
                )
            
            # Efficiency breakdown
            st.markdown("#### Response Source Breakdown")
            
            # Count sources from messages
            source_counts = {"template": 0, "cache": 0, "knowledge_base": 0, "llm": 0}
            for msg in st.session_state.smart_interview_messages:
                if msg["role"] == "assistant":
                    source = msg.get("source", "template")
                    if source in source_counts:
                        source_counts[source] += 1
            
            # Create columns for each source
            if sum(source_counts.values()) > 0:
                cols = st.columns(len([k for k, v in source_counts.items() if v > 0]))
                
                for i, (source, count) in enumerate(source_counts.items()):
                    if count > 0:
                        with cols[i]:
                            percentage = (count / sum(source_counts.values())) * 100
                            st.metric(
                                source.title(),
                                f"{count} ({percentage:.1f}%)"
                            )
            
            # Show savings comparison
            if stats['total_responses'] > 0:
                st.markdown("#### 💰 Cost Comparison")
                
                traditional_cost = stats['total_responses'] * 0.02
                smart_cost = stats['api_calls_made'] * 0.02
                savings = traditional_cost - smart_cost
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Traditional Cost", f"${traditional_cost:.2f}")
                with col2:
                    st.metric("Smart Agent Cost", f"${smart_cost:.2f}")
                with col3:
                    st.metric("Savings", f"${savings:.2f}", delta=f"{stats['api_savings_percentage']:.1f}%") 