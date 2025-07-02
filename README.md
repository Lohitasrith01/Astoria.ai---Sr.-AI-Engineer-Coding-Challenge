# AI-Powered Video Interview Platform

## 🚀 Overview

This project is a proof-of-concept (POC) for a next-generation, AI-powered video interview platform. It features a personalized AI Interview Agent that conducts dynamic, intelligent interviews tailored to each candidate's background and the company's hiring needs. The system is designed for efficiency, fairness, and transparency, with robust human-in-the-loop controls and hallucination mitigation.

---

## 🏗️ System Architecture

- **Frontend:**  
  - React Recruiter Dashboard (dark theme)
  - React Candidate Dashboard (dark theme)
  - WebRTC for real-time video interviews

- **Backend:**  
  - FastAPI server orchestrating all workflows
  - WebRTC signaling and interview WebSocket endpoints
  - Orchestration engine for ATS, scoring, and video interview management
  - Redis for session/context storage
  - AWS S3 for video and resume storage
  - LangChain (with Gemini LLM) for interview agent, RAG, and memory
  - Mediapipe (video analysis), OpenSmile (audio analysis)
  - SmartFilter (ATS/semantic filtering), Scoring Engine

- **External Services:**  
  - Gemini (LLM)
  - Gmail API (notifications)
  - Calendly API (scheduling)

---

## 🧩 Inputs

- **Candidate Data:** Resume (PDF/DOCX), LinkedIn, GitHub, skills, experience, etc.
- **Job Data:** Title, description, required/preferred skills, experience level.
- **Recruiter Custom Questions:** Recruiters can add custom questions before launching an interview.
- **Video/Audio Streams:** Captured in real time during the interview.

---

## 🧠 AI Functionalities

- **Resume Parsing & ATS Filtering:**  
  - Uses SmartFilter (SentenceTransformers) for fast, semantic candidate-job matching.
- **Role-Specific Question Generation:**  
  - LangChain agent generates questions tailored to the candidate's background and the job requirements.
- **RAG (Retrieval-Augmented Generation):**  
  - The agent is grounded in a curated set of FAQs, role-specific questions, and expected answers, minimizing hallucinations.
- **Real-Time Video/Audio Analysis:**  
  - Mediapipe for facial/gesture analysis; OpenSmile for speech/tone features.
- **Scoring & Reporting:**  
  - Each interview generates a structured report: overall score, technical/communication/cultural fit, bias/tone analysis, strengths, concerns, and recommendations.
- **Human-in-the-Loop:**  
  - Recruiters review all reports and transcripts, can flag issues, and override recommendations.
- **Notifications & Scheduling:**  
  - Gmail API for email notifications; Calendly API for scheduling follow-up calls.

---

## 🛡️ Hallucination Mitigation

- **RAG System:**  
  - The agent's answers are grounded in a retrieval-augmented system with FAQs and role-specific Q&A, reducing the risk of hallucinated or irrelevant responses.
- **Human-in-the-Loop:**  
  - All interview outputs are reviewed by a human recruiter, who can flag or correct any inconsistencies, hallucinations, or bias.
- **Logging & Feedback:**  
  - All reports and recruiter feedback are logged for continuous improvement.

---

## 🗣️ Question Generation & Adaptivity

- The AI agent adapts questions in real time based on:
  - Candidate's resume, skills, and experience
  - Job requirements and recruiter custom questions
  - Previous answers and detected strengths/concerns
- The agent maintains a supportive, unbiased, and conversational tone, with real-time tone/bias analysis.

---

## 🖥️ User Experience

### **Recruiter Workflow**
1. Upload job and candidate data (or import from LinkedIn/GitHub).
2. Review candidate pool, see match status, and expand for detailed reports.
3. Add custom questions and launch video interviews (one or many in parallel).
4. Review structured reports and full transcripts after each interview.
5. Reject, schedule follow-up, or proceed to next steps—all from a modern dashboard.

### **Candidate Workflow**
1. Receive an invite and join the interview via a secure link.
2. Participate in a real-time video interview with the AI agent.
3. Receive supportive feedback and a thank-you message at the end.

---

## 📊 Evaluation & Metrics

- **Metrics Captured:**
  - Overall score, technical competence, communication, cultural fit, problem solving, bias risk, strengths, concerns, recommendation, and more.
- **Evaluation Process:**
  - Recruiters review all reports for consistency, depth, and hallucination.
  - RAG and human-in-the-loop ensure high-quality, reliable outputs.
- **Continuous Improvement:**
  - All recruiter feedback and flagged issues are logged for future model and system improvements.

---

## 🔒 Security & Traceability

- All actions are session-based and logged (Redis).
- Video and resume data are securely stored in S3.
- All communications and scheduling actions are deduplicated and auditable.

---

## 🛠️ Extensibility

- Easily add new LLMs, data sources, or analytics endpoints.
- Ready for integration with additional HR tools or ATS systems.
- Modular design for rapid feature development.

---

## 🏁 How to Run

1. **Backend:**  
   - `pip install -r requirements.txt`
   - `python -m video_interview_backend.main`
2. **Frontend:**  
   - `cd recruiter-dashboard` / `cd candidate-dashboard`
   - `npm install && npm start`
3. **Configure environment variables** for Redis, S3, Gmail, Calendly, and Gemini as needed.

---

## 📚 Solution Narrative

This system demonstrates a scalable, human-centered approach to AI-powered interviews. By combining real-time video, advanced AI, RAG grounding, and human oversight, it delivers a fair, efficient, and transparent hiring experience for both recruiters and candidates.

---

