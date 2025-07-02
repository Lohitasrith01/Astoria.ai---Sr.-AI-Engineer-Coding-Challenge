"""
Complete Interview Agent Integration for Streamlit
Includes:
- Question generation based on role and candidate profile
- 1-hour interview duration
- Custom recruiter questions
- Confirmation questions (visa, relocation, notice period)
- Real-time progress tracking
- Final analysis for recruiters
"""

import streamlit as st
import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict

from src.models import JobPost, Candidate
from agents.interview_agent import get_interview_agent, InterviewContext, InterviewStage


class InterviewUI:
    """Streamlit UI for conducting AI interviews"""
    
    def __init__(self):
        self.agent = get_interview_agent()
        self.session_key = "interview_session"
    
    def setup_interview_page(self):
        """Setup the interview configuration page"""
        st.header("🎤 AI Interview Setup")
        
        # Interview Configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Interview Configuration")
            
            # Duration settings
            interview_duration = st.selectbox(
                "Interview Duration",
                options=[30, 45, 60, 75, 90],
                index=2,  # Default to 60 minutes
                help="Total interview duration in minutes"
            )
            
            # Question customization
            st.subheader("Custom Questions")
            st.write("Add specific questions you want the interviewer to ask:")
            
            # Initialize custom questions in session state
            if "custom_questions" not in st.session_state:
                st.session_state.custom_questions = []
            
            # Add new custom question
            new_question = st.text_area(
                "Add Custom Question",
                placeholder="e.g., Can you tell me about your experience with microservices architecture?",
                key="new_custom_question"
            )
            
            col_add, col_clear = st.columns([1, 1])
            with col_add:
                if st.button("➕ Add Question") and new_question.strip():
                    st.session_state.custom_questions.append(new_question.strip())
                    st.session_state.new_custom_question = ""  # Clear the input
                    st.rerun()
            
            with col_clear:
                if st.button("🗑️ Clear All Questions"):
                    st.session_state.custom_questions = []
                    st.rerun()
            
            # Display current custom questions
            if st.session_state.custom_questions:
                st.write("**Custom Questions Added:**")
                for i, q in enumerate(st.session_state.custom_questions):
                    col_q, col_remove = st.columns([4, 1])
                    with col_q:
                        st.write(f"{i+1}. {q}")
                    with col_remove:
                        if st.button("❌", key=f"remove_q_{i}"):
                            st.session_state.custom_questions.pop(i)
                            st.rerun()
            
            # Confirmation questions settings
            st.subheader("Confirmation Questions")
            st.write("Standard questions asked at the end of the interview:")
            
            default_confirmations = [
                "Are you open to relocating for this position?",
                "What is your current visa status?",
                "What is your notice period with your current employer?",
                "What are your salary expectations for this role?",
                "When would you be available to start?",
                "Do you have any other offers or interview processes ongoing?"
            ]
            
            selected_confirmations = st.multiselect(
                "Select confirmation questions to include:",
                options=default_confirmations,
                default=default_confirmations[:4],  # Select first 4 by default
                help="These questions will be asked at the end of the interview"
            )
            
            # Custom confirmation question
            custom_confirmation = st.text_input(
                "Add Custom Confirmation Question",
                placeholder="e.g., Are you willing to work in a hybrid environment?"
            )
            
            if st.button("➕ Add Confirmation Question") and custom_confirmation.strip():
                selected_confirmations.append(custom_confirmation.strip())
                st.rerun()
        
        with col2:
            # Interview Preview
            st.subheader("Interview Preview")
            
            # Get candidate and job info from session state
            candidate = st.session_state.get("interview_candidate")
            job = st.session_state.get("interview_job")
            initial_analysis = st.session_state.get("interview_initial_analysis", {})
            
            if candidate and job:
                st.metric("Duration", f"{interview_duration} minutes")
                st.metric("Custom Questions", len(st.session_state.custom_questions))
                st.metric("Confirmation Questions", len(selected_confirmations))
                
                # Show key concerns from analysis
                concerns = initial_analysis.get("concerns", [])
                if concerns:
                    st.write("**Key Areas to Explore:**")
                    for concern in concerns[:3]:
                        st.write(f"• {concern}")
                
                # Generate interview questions preview
                if st.button("🔍 Preview Questions"):
                    st.session_state.preview_questions = self._generate_interview_questions(
                        job, candidate, initial_analysis, st.session_state.custom_questions
                    )
                    st.rerun()
                
                # Show question preview
                if "preview_questions" in st.session_state:
                    with st.expander("📋 Interview Questions Preview", expanded=True):
                        questions = st.session_state.preview_questions
                        for stage, qs in questions.items():
                            st.write(f"**{stage.replace('_', ' ').title()}:**")
                            for i, q in enumerate(qs, 1):
                                st.write(f"  {i}. {q}")
                            st.write("---")
                
                # Start interview button
                st.subheader("🚀 Launch Interview")
                if st.button("🎤 Start AI Interview", type="primary", use_container_width=True):
                    # Save interview configuration
                    st.session_state.interview_config = {
                        "duration_minutes": interview_duration,
                        "custom_questions": st.session_state.custom_questions.copy(),
                        "confirmation_questions": selected_confirmations.copy(),
                        "start_time": time.time()
                    }
                    
                    # Initialize interview context
                    interview_context = self.agent.start_interview(
                        job, candidate, initial_analysis
                    )
                    
                    # Modify agent settings for longer interview
                    self.agent.max_interview_duration = interview_duration * 60  # Convert to seconds
                    
                    st.session_state.interview_context = interview_context
                    st.session_state.interview_stage = "active"
                    st.session_state.current_message = ""
                    
                    st.success("🎉 Interview started! Redirecting to interview interface...")
                    time.sleep(1)
                    st.rerun()
            else:
                st.error("⚠️ Missing candidate or job information. Please complete the analysis first.")
    
    def _generate_interview_questions(self, job: JobPost, candidate: Candidate, 
                                    initial_analysis: Dict, custom_questions: List[str]) -> Dict[str, List[str]]:
        """Generate interview questions for preview"""
        questions = {
            "introduction": [
                f"Hi {candidate.name}! Welcome to your interview for the {job.title} position. Could you start by telling me about yourself and what excites you about this role?"
            ],
            "concerns_exploration": [],
            "technical_screening": [
                f"Tell me about your experience with the key technologies mentioned in the job description.",
                "What's been your most challenging technical project recently?",
                "How do you stay updated with new technologies in your field?"
            ],
            "experience_deep_dive": [
                "Can you walk me through one of your most impactful projects?",
                "Tell me about a time when you had to learn a new technology quickly.",
                "How do you approach debugging complex technical issues?"
            ],
            "behavioral": [
                "Describe a situation where you had to work with a difficult team member.",
                "Tell me about a time when you had to make a decision with incomplete information.",
                "How do you prioritize your work when facing multiple deadlines?"
            ],
            "problem_solving": [
                "I'd like to present you with a technical scenario relevant to this role...",
                "How would you approach scaling a system that's experiencing performance issues?"
            ],
            "custom_questions": custom_questions,
            "confirmation": [
                "Now I'd like to ask a few practical questions about the role..."
            ]
        }
        
        # Add concern-specific questions
        concerns = initial_analysis.get("concerns", [])
        for concern in concerns:
            if "experience" in concern.lower():
                questions["concerns_exploration"].append(f"I'd love to hear about your experience related to {concern.lower()}.")
            elif any(tech in concern.lower() for tech in ["python", "react", "docker", "aws", "kubernetes"]):
                questions["concerns_exploration"].append(f"Tell me about any projects where you've worked with {concern.split()[-1]}.")
            else:
                questions["concerns_exploration"].append(f"Can you share your thoughts on {concern.lower()}?")
        
        return questions
    
    def run_interview_interface(self):
        """Main interview interface"""
        st.header("🎤 AI Interview in Progress")
        
        context = st.session_state.interview_context
        config = st.session_state.interview_config
        
        # Interview progress sidebar
        with st.sidebar:
            self._show_interview_progress(context, config)
        
        # Main interview area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            self._show_interview_conversation(context)
        
        with col2:
            self._show_real_time_insights(context)
        
        # Interview controls
        self._show_interview_controls(context, config)
    
    def _show_interview_progress(self, context: InterviewContext, config: Dict):
        """Show interview progress in sidebar"""
        st.subheader("📊 Interview Progress")
        
        # Time progress
        elapsed = time.time() - context.start_time
        total_duration = config["duration_minutes"] * 60
        remaining = max(0, total_duration - elapsed)
        
        progress = min(elapsed / total_duration, 1.0)
        st.progress(progress)
        st.write(f"⏱️ {elapsed//60:.0f}:{elapsed%60:02.0f} / {total_duration//60:.0f}:00")
        st.write(f"⏳ {remaining//60:.0f}:{remaining%60:02.0f} remaining")
        
        # Stage progress
        st.write("**Current Stage:**")
        st.write(f"🎯 {context.current_stage.value.replace('_', ' ').title()}")
        
        # Completion status
        stages_completed = len([s for s in context.stage_progress.values() if s.get('completed', False)])
        total_stages = len(context.stage_progress)
        st.write(f"**Stages:** {stages_completed}/{total_stages} completed")
        
        # Key metrics
        st.write("**Interview Metrics:**")
        st.write(f"• Messages: {len(context.messages)}")
        st.write(f"• Strengths found: {len(context.strengths_identified)}")
        st.write(f"• Concerns addressed: {len(context.concerns_addressed)}")
        
        # Real-time recommendation
        insights = self.agent.get_recruiter_insights(context)
        recommendation = insights["current_assessment"]["preliminary_recommendation"]
        st.write(f"**Current Assessment:** {recommendation}")
    
    def _show_interview_conversation(self, context: InterviewContext):
        """Show the interview conversation"""
        st.subheader("💬 Interview Conversation")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display conversation history
            for message in context.messages:
                if message.sender == "agent":
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(message.content)
                        
                        # Show tone analysis if available
                        if message.tone_analysis and hasattr(message.tone_analysis, 'professionalism_score'):
                            if message.tone_analysis.professionalism_score < 0.7:
                                st.caption("⚠️ Tone could be improved")
                else:
                    with st.chat_message("user", avatar="👤"):
                        st.write(message.content)
                        
                        # Show confidence indicators
                        if message.tone_analysis and hasattr(message.tone_analysis, 'confidence_level'):
                            confidence = message.tone_analysis.confidence_level
                            if confidence < 0.4:
                                st.caption("💡 Consider providing encouragement")
                            elif confidence > 0.8:
                                st.caption("✨ High confidence response")
        
        # Input for next response
        if context.current_stage != InterviewStage.CLOSING:
            candidate_response = st.text_area(
                "Candidate Response:",
                height=100,
                key="candidate_input",
                placeholder="Type the candidate's response here..."
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if st.button("📤 Submit Response", type="primary"):
                    if candidate_response.strip():
                        # Process the response asynchronously
                        self._process_candidate_response(context, candidate_response.strip())
                        st.session_state.candidate_input = ""  # Clear input
                        st.rerun()
                    else:
                        st.error("Please enter a response")
            
            with col2:
                if st.button("⏭️ Next Stage"):
                    self.agent._transition_to_next_stage(context)
                    st.rerun()
            
            with col3:
                if st.button("🔚 End Interview"):
                    context.current_stage = InterviewStage.CLOSING
                    st.session_state.interview_stage = "ending"
                    st.rerun()
        else:
            st.success("✅ Interview completed!")
            if st.button("📊 View Final Analysis", type="primary"):
                st.session_state.interview_stage = "analysis"
                st.rerun()
    
    def _show_real_time_insights(self, context: InterviewContext):
        """Show real-time insights for recruiters"""
        st.subheader("🔍 Real-time Insights")
        
        insights = self.agent.get_recruiter_insights(context)
        
        # Interview health
        health = insights["interview_health"]
        st.write("**Interview Health:**")
        st.write(f"Status: {health['overall_status']}")
        st.write(f"Engagement: {health['candidate_engagement']}")
        st.write(f"Bias Risk: {health['bias_risk']}")
        
        # Key findings
        st.write("**Recent Findings:**")
        for finding in insights["key_findings"]["strengths_so_far"][-3:]:
            st.write(f"✅ {finding}")
        
        # Remaining concerns
        remaining = insights["key_findings"]["concerns_remaining"]
        if remaining:
            st.write("**Still to Address:**")
            for concern in remaining[:2]:
                st.write(f"❓ {concern}")
        
        # Recommendations
        st.write("**Recommendations:**")
        for rec in insights["recommendations"]["focus_areas"][:2]:
            st.write(f"💡 {rec}")
        
        # Estimated score
        score = insights["recommendations"]["estimated_final_score"]
        st.metric("Estimated Final Score", f"{score:.1f}/10")
    
    def _process_candidate_response(self, context: InterviewContext, response: str):
        """Process candidate response asynchronously"""
        try:
            # Use asyncio to handle the async method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            agent_response = loop.run_until_complete(
                self.agent.generate_response(context, response)
            )
            loop.close()
            
            # Update session state
            st.session_state.interview_context = context
            
        except Exception as e:
            st.error(f"Error processing response: {e}")
    
    def show_final_analysis(self):
        """Show final interview analysis"""
        st.header("📊 Final Interview Analysis")
        
        context = st.session_state.interview_context
        config = st.session_state.interview_config
        
        # Generate comprehensive summary
        summary = self.agent.generate_interview_summary(context)
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        assessment = summary["assessment"]
        
        with col1:
            st.metric("Overall Score", f"{assessment['overall_score']:.1f}/10")
        with col2:
            st.metric("Technical Competence", f"{assessment['technical_competence']:.1f}/10")
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
        
        # Interview quality metrics
        st.subheader("📈 Interview Quality Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Interview Quality", f"{assessment.get('interview_quality_score', 8.0):.1f}/10")
            st.metric("Bias Risk Assessment", f"{assessment.get('bias_risk_assessment', 1.0):.1f}/10")
        
        with col2:
            duration = summary["interview_metadata"]["duration_minutes"]
            st.metric("Duration", f"{duration:.1f} min")
            st.metric("Total Messages", summary["interview_metadata"]["total_messages"])
        
        with col3:
            confidence = assessment.get("confidence_in_assessment", 8.0)
            st.metric("Assessment Confidence", f"{confidence:.1f}/10")
            st.metric("Stages Completed", f"{summary['interview_metadata']['stages_completed']}/7")
        
        # Concerns from initial analysis
        if "concerns_from_initial_analysis" in assessment:
            st.subheader("🔍 Initial Concerns Analysis")
            st.write("How well were the initial concerns addressed:")
            for concern_analysis in assessment["concerns_from_initial_analysis"]:
                st.write(f"• {concern_analysis}")
        
        # Export options
        st.subheader("📤 Export Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Copy Summary", use_container_width=True):
                summary_text = self._generate_summary_text(summary, assessment)
                st.text_area("Summary (Copy this text):", summary_text, height=300)
        
        with col2:
            if st.button("📊 Download JSON", use_container_width=True):
                import json
                json_str = json.dumps(summary, indent=2, default=str)
                st.download_button(
                    "Download JSON Report",
                    json_str,
                    f"interview_analysis_{context.candidate.name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    "application/json"
                )
        
        with col3:
            if st.button("🔄 New Interview", use_container_width=True):
                # Clear interview session
                keys_to_clear = [
                    "interview_context", "interview_config", "interview_stage",
                    "interview_active", "candidate_input", "preview_questions"
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    def _generate_summary_text(self, summary: Dict, assessment: Dict) -> str:
        """Generate a text summary for copying"""
        candidate_name = summary.get("interview_metadata", {}).get("candidate_name", "Candidate")
        duration = summary.get("interview_metadata", {}).get("duration_minutes", 0)
        
        text = f"""
INTERVIEW ANALYSIS SUMMARY
==========================

Candidate: {candidate_name}
Duration: {duration:.1f} minutes
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

SCORES:
- Overall: {assessment['overall_score']:.1f}/10
- Technical: {assessment['technical_competence']:.1f}/10
- Communication: {assessment['communication_skills']:.1f}/10
- Cultural Fit: {assessment['cultural_fit']:.1f}/10

RECOMMENDATION: {assessment['recommendation']}

STRENGTHS:
{chr(10).join([f"• {s}" for s in assessment['strengths']])}

CONCERNS:
{chr(10).join([f"• {c}" for c in assessment['concerns']])}

KEY INSIGHTS:
{chr(10).join([f"• {i}" for i in assessment['key_insights']])}

NEXT STEPS:
{assessment['next_steps']}

ASSESSMENT CONFIDENCE: {assessment.get('confidence_in_assessment', 8.0):.1f}/10
"""
        return text


# Main function to integrate with existing app
def add_interview_interface():
    """Add interview interface to the existing Streamlit app"""
    
    # Check if interview is active
    if st.session_state.get("interview_active", False):
        interview_ui = InterviewUI()
        
        # Determine interview stage
        stage = st.session_state.get("interview_stage", "setup")
        
        if stage == "setup":
            interview_ui.setup_interview_page()
        elif stage == "active":
            interview_ui.run_interview_interface()
        elif stage == "analysis":
            interview_ui.show_final_analysis()
        else:
            st.error("Unknown interview stage")
    
    return False  # Not active


# Function to launch interview from existing analysis
def launch_interview_button(candidate_data: Dict, job_data: Dict, initial_analysis: Dict):
    """Add a launch interview button to existing analysis sections"""
    
    if st.button("🎤 Launch AI Interview", key="launch_interview_btn", type="primary"):
        # Prepare data
        candidate = Candidate(
            name=candidate_data.get("name", "Candidate"),
            github_handle=candidate_data.get("github_handle", ""),
            linkedin_url=candidate_data.get("linkedin_url"),
            skills=initial_analysis.get("strengths", []),
            experience_years=candidate_data.get("years_experience", 5),
            location=candidate_data.get("location", "TBD")
        )
        
        job = JobPost(
            title=job_data.get("title", "Position"),
            description=job_data.get("description", ""),
            required_skills=job_data.get("required_skills", []),
            preferred_skills=job_data.get("preferred_skills", []),
            experience_level=job_data.get("experience_level", "Mid-level")
        )
        
        # Save to session state
        st.session_state["interview_active"] = True
        st.session_state["interview_candidate"] = candidate
        st.session_state["interview_job"] = job
        st.session_state["interview_initial_analysis"] = initial_analysis
        st.session_state["interview_stage"] = "setup"
        
        st.success("🎉 Interview setup initiated! Redirecting...")
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    # Test the interview interface
    st.set_page_config(page_title="AI Interview System", layout="wide")
    
    if not st.session_state.get("interview_active", False):
        st.title("🎤 AI Interview System - Demo")
        
        # Demo button
        if st.button("Start Demo Interview"):
            demo_candidate = Candidate(
                name="John Doe",
                github_handle="johndoe",
                skills=["Python", "JavaScript", "React"],
                experience_years=5
            )
            
            demo_job = JobPost(
                title="Senior Software Engineer",
                description="Looking for experienced developers",
                required_skills=["Python", "React", "PostgreSQL"],
                experience_level="Senior"
            )
            
            demo_analysis = {
                "overall_fit_score": 0.8,
                "concerns": ["No PostgreSQL experience", "Limited backend experience"],
                "strengths": ["Strong frontend skills", "Good GitHub portfolio"],
                "recommendation": "GOOD_MATCH"
            }
            
            st.session_state["interview_active"] = True
            st.session_state["interview_candidate"] = demo_candidate
            st.session_state["interview_job"] = demo_job
            st.session_state["interview_initial_analysis"] = demo_analysis
            st.session_state["interview_stage"] = "setup"
            st.rerun()
    else:
        add_interview_interface() 