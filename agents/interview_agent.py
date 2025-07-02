import asyncio
import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.memory import BaseMemory

from src.models import JobPost, Candidate
from src.vertex_llm import get_vertex_client


class InterviewStage(Enum):
    INTRODUCTION = "introduction"
    CONCERNS_EXPLORATION = "concerns_exploration"  # New stage for addressing gaps
    TECHNICAL_SCREENING = "technical_screening"
    EXPERIENCE_DEEP_DIVE = "experience_deep_dive"
    BEHAVIORAL = "behavioral"
    PROBLEM_SOLVING = "problem_solving"
    QUESTIONS_FROM_CANDIDATE = "questions_from_candidate"
    CLOSING = "closing"


@dataclass
class ToneAnalysis:
    """Results of tone analysis for a message"""
    confidence_level: float = 0.0
    enthusiasm_level: float = 0.0
    clarity_score: float = 0.0
    professionalism_score: float = 0.0
    stress_indicators: List[str] = field(default_factory=list)
    positive_indicators: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)


@dataclass
class BiasAnalysis:
    """Results of bias analysis for questions/responses"""
    bias_detected: bool = False
    bias_types: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    suggestions: List[str] = field(default_factory=list)
    approved_content: str = ""


@dataclass
class InterviewMessage:
    """Represents a single message in the interview conversation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    sender: str = ""  # "agent" or "candidate"
    content: str = ""
    stage: InterviewStage = InterviewStage.INTRODUCTION
    metadata: Dict[str, Any] = field(default_factory=dict)
    tone_analysis: Optional[ToneAnalysis] = None
    bias_analysis: Optional[BiasAnalysis] = None


@dataclass
class InterviewContext:
    """Maintains context and state throughout the interview"""
    job_post: JobPost
    candidate: Candidate
    initial_analysis: Dict[str, Any] = field(default_factory=dict)  # From profile analysis
    messages: List[InterviewMessage] = field(default_factory=list)
    current_stage: InterviewStage = InterviewStage.INTRODUCTION
    stage_progress: Dict[InterviewStage, Dict[str, Any]] = field(default_factory=dict)
    technical_topics_covered: List[str] = field(default_factory=list)
    behavioral_topics_covered: List[str] = field(default_factory=list)
    concerns_addressed: List[str] = field(default_factory=list)  # Concerns from analysis that were explored
    gaps_explored: List[str] = field(default_factory=list)  # Skills gaps that were discussed
    red_flags: List[str] = field(default_factory=list)
    strengths_identified: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    interview_duration: float = 0.0
    start_time: float = field(default_factory=time.time)
    bias_incidents: List[Dict[str, Any]] = field(default_factory=list)  # Track bias issues
    tone_trends: Dict[str, List[float]] = field(default_factory=dict)  # Track tone over time
    # LangChain Memory Integration
    conversation_memory: Optional[ConversationBufferWindowMemory] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_k: int = 20  # Number of conversation turns to remember


class InterviewAgent:
    """AI Agent that conducts personalized, unbiased interviews with tone analysis and conversation memory"""
    
    def __init__(self, memory_k: int = 20):
        self.llm_client = get_vertex_client()
        self.max_interview_duration = 45 * 60  # 45 minutes in seconds
        self.memory_k = memory_k  # Number of conversation turns to remember
        
        # Define stage transitions and typical durations (updated with new stage)
        self.stage_flow = {
            InterviewStage.INTRODUCTION: InterviewStage.CONCERNS_EXPLORATION,
            InterviewStage.CONCERNS_EXPLORATION: InterviewStage.TECHNICAL_SCREENING,
            InterviewStage.TECHNICAL_SCREENING: InterviewStage.EXPERIENCE_DEEP_DIVE,
            InterviewStage.EXPERIENCE_DEEP_DIVE: InterviewStage.BEHAVIORAL,
            InterviewStage.BEHAVIORAL: InterviewStage.PROBLEM_SOLVING,
            InterviewStage.PROBLEM_SOLVING: InterviewStage.QUESTIONS_FROM_CANDIDATE,
            InterviewStage.QUESTIONS_FROM_CANDIDATE: InterviewStage.CLOSING,
        }
        
        self.stage_durations = {
            InterviewStage.INTRODUCTION: 3 * 60,  # 3 minutes
            InterviewStage.CONCERNS_EXPLORATION: 8 * 60,  # 8 minutes - new stage
            InterviewStage.TECHNICAL_SCREENING: 10 * 60,  # 10 minutes (reduced)
            InterviewStage.EXPERIENCE_DEEP_DIVE: 8 * 60,  # 8 minutes (reduced)
            InterviewStage.BEHAVIORAL: 8 * 60,  # 8 minutes
            InterviewStage.PROBLEM_SOLVING: 8 * 60,  # 8 minutes (reduced)
            InterviewStage.QUESTIONS_FROM_CANDIDATE: 2 * 60,  # 2 minutes
        }
    
    async def analyze_tone(self, message: str, sender: str) -> ToneAnalysis:
        """Analyze the tone and sentiment of a message"""
        tone_prompt = f"""
Analyze the tone and communication quality of this {sender} message in an interview context:

MESSAGE: "{message}"

Provide analysis in JSON format:
{{
    "confidence_level": <float 0.0-1.0, how confident/certain the speaker sounds>,
    "enthusiasm_level": <float 0.0-1.0, level of enthusiasm and energy>,
    "clarity_score": <float 0.0-1.0, how clear and articulate the communication is>,
    "professionalism_score": <float 0.0-1.0, level of professionalism>,
    "stress_indicators": ["list of any stress or nervousness indicators"],
    "positive_indicators": ["list of positive communication elements"],
    "concerns": ["list of communication concerns if any"]
}}
"""
        
        response = self.llm_client.generate_response(tone_prompt, max_tokens=800, temperature=0.2)
        
        if response:
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
                    return ToneAnalysis(**analysis_data)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Fallback analysis
        return ToneAnalysis(
            confidence_level=0.5,
            enthusiasm_level=0.5,
            clarity_score=0.5,
            professionalism_score=0.8,
            concerns=["Unable to analyze tone"]
        )

    def _initialize_conversation_memory(self, context: InterviewContext) -> None:
        """Initialize LangChain conversation memory for the interview session"""
        if context.conversation_memory is None:
            context.conversation_memory = ConversationBufferWindowMemory(
                k=context.memory_k,
                return_messages=True,
                memory_key="chat_history",
                input_key="human_input",
                output_key="ai_response"
            )
            
            # Add initial context to memory
            initial_message = f"""Interview started for {context.candidate.name} applying for {context.job_post.title}.
Initial Analysis Summary: {json.dumps(context.initial_analysis, indent=2) if context.initial_analysis else 'None'}"""
            
            context.conversation_memory.save_context(
                {"human_input": "Starting interview session"},
                {"ai_response": initial_message}
            )

    def _add_to_memory(self, context: InterviewContext, human_input: str, ai_response: str) -> None:
        """Add a conversation turn to the memory"""
        if context.conversation_memory:
            context.conversation_memory.save_context(
                {"human_input": human_input},
                {"ai_response": ai_response}
            )

    def _get_conversation_context(self, context: InterviewContext) -> str:
        """Get the conversation history from memory as formatted text"""
        if not context.conversation_memory:
            return "No conversation history available."
        
        try:
            memory_variables = context.conversation_memory.load_memory_variables({})
            chat_history = memory_variables.get("chat_history", [])
            
            if not chat_history:
                return "No conversation history available."
            
            # Format the conversation history
            formatted_history = []
            for message in chat_history:
                if hasattr(message, 'content'):
                    role = "Human" if isinstance(message, HumanMessage) else "AI"
                    formatted_history.append(f"{role}: {message.content}")
            
            return "\n".join(formatted_history[-10:])  # Last 10 messages for context
        except Exception as e:
            return f"Error retrieving conversation history: {str(e)}"

    def _get_memory_summary(self, context: InterviewContext) -> Dict[str, Any]:
        """Get a summary of the conversation memory"""
        if not context.conversation_memory:
            return {"total_turns": 0, "memory_length": 0, "session_id": context.session_id}
        
        try:
            memory_variables = context.conversation_memory.load_memory_variables({})
            chat_history = memory_variables.get("chat_history", [])
            
            return {
                "total_turns": len(chat_history) // 2,  # Each turn = human + AI message
                "memory_length": len(chat_history),
                "session_id": context.session_id,
                "last_k_turns": context.memory_k,
                "memory_active": True
            }
        except Exception as e:
            return {
                "total_turns": 0, 
                "memory_length": 0, 
                "session_id": context.session_id,
                "error": str(e),
                "memory_active": False
            }

    async def check_for_bias(self, content: str, context: InterviewContext) -> BiasAnalysis:
        """Check interview content for potential bias and suggest improvements"""
        bias_prompt = f"""
You are a bias detection expert. Analyze this interview question/response for potential bias.

CONTENT TO ANALYZE: "{content}"

INTERVIEW CONTEXT:
- Position: {context.job_post.title}
- Candidate: {context.candidate.name}
- Stage: {context.current_stage.value}

Check for these bias types:
1. Gender bias (assumptions based on gender)
2. Age bias (assumptions about age/experience)
3. Cultural bias (assumptions about background/culture)
4. Educational bias (assumptions about education/credentials)
5. Appearance bias (comments about physical attributes)
6. Socioeconomic bias (assumptions about background)
7. Disability bias (assumptions about capabilities)
8. Unconscious language bias (loaded or problematic terms)

Provide analysis in JSON format:
{{
    "bias_detected": <true/false>,
    "bias_types": ["list of bias types found"],
    "risk_score": <float 0.0-1.0, overall bias risk>,
    "suggestions": ["list of suggestions to improve fairness"],
    "approved_content": "<revised version if bias detected, original if clean>"
}}
"""
        
        response = self.llm_client.generate_response(bias_prompt, max_tokens=1000, temperature=0.1)
        
        if response:
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
                    return BiasAnalysis(**analysis_data)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Fallback - assume content is okay
        return BiasAnalysis(
            bias_detected=False,
            risk_score=0.0,
            approved_content=content,
            suggestions=["Analysis failed - manual review recommended"]
        )
    
    def _create_system_prompt(self, context: InterviewContext) -> str:
        """Create the system prompt based on current interview context"""
        
        # Calculate time remaining
        elapsed_time = time.time() - context.start_time
        remaining_time = max(0, self.max_interview_duration - elapsed_time)
        
        # Build context summary with initial analysis
        initial_analysis_summary = ""
        if context.initial_analysis:
            initial_analysis_summary = f"""
INITIAL CANDIDATE ANALYSIS:
Overall Fit Score: {context.initial_analysis.get('overall_fit_score', 'N/A')}
Key Concerns: {', '.join(context.initial_analysis.get('concerns', []))}
Strengths: {', '.join(context.initial_analysis.get('strengths', []))}
Transferable Skills: {', '.join(context.initial_analysis.get('transferable_skills', []))}
Recommendation: {context.initial_analysis.get('recommendation', 'N/A')}

AREAS TO EXPLORE BASED ON ANALYSIS:
- Address the concerns: {', '.join(context.initial_analysis.get('concerns', []))}
- Validate the strengths: {', '.join(context.initial_analysis.get('strengths', []))}
- Explore transferable skills: {', '.join(context.initial_analysis.get('transferable_skills', []))}
"""
        
        context_summary = f"""
INTERVIEW CONTEXT:
- Position: {context.job_post.title}
- Candidate: {context.candidate.name}
- Current Stage: {context.current_stage.value}
- Time Remaining: {remaining_time // 60:.0f} minutes
- Topics Covered: {', '.join(context.technical_topics_covered + context.behavioral_topics_covered)}
- Concerns Addressed: {', '.join(context.concerns_addressed)}
- Gaps Explored: {', '.join(context.gaps_explored)}
- Strengths Identified: {', '.join(context.strengths_identified)}
- Red Flags: {', '.join(context.red_flags)}

{initial_analysis_summary}

JOB REQUIREMENTS:
{context.job_post.description}
Required Skills: {', '.join(context.job_post.required_skills or [])}
Preferred Skills: {', '.join(context.job_post.preferred_skills or [])}
Experience Level: {context.job_post.experience_level}

CANDIDATE PROFILE:
Name: {context.candidate.name}
Experience: {context.candidate.experience_years} years
Skills: {', '.join(context.candidate.skills or [])}
Location: {context.candidate.location}
"""
        
        # Add conversation history from LangChain memory
        conversation_context = self._get_conversation_context(context)
        if conversation_context and conversation_context != "No conversation history available.":
            context_summary += f"\nCONVERSATION HISTORY:\n{conversation_context}\n"
        
        # Also add recent message tone analysis if available
        recent_messages = context.messages[-3:] if len(context.messages) > 3 else context.messages
        if recent_messages:
            tone_summary = ""
            for msg in recent_messages:
                if msg.tone_analysis:
                    tone_summary += f"{msg.sender}: Confidence={msg.tone_analysis.confidence_level:.1f}, Clarity={msg.tone_analysis.clarity_score:.1f}\n"
            if tone_summary:
                context_summary += f"\nRECENT TONE ANALYSIS:\n{tone_summary}"
        
        # Stage-specific instructions
        stage_instructions = self._get_stage_instructions(context.current_stage, context)
        
        system_prompt = f"""You are an expert AI interview agent conducting a professional, friendly, and unbiased technical interview. 

{context_summary}

CURRENT STAGE INSTRUCTIONS:
{stage_instructions}

CORE PRINCIPLES:
1. FRIENDLINESS: Maintain a warm, encouraging, and supportive tone throughout
2. CONVERSATIONAL: Use natural, flowing conversation rather than rigid Q&A
3. UNBIASED: Ask fair, job-relevant questions regardless of candidate background
4. SMART: Adapt questions based on responses and dig deeper into interesting areas
5. SUPPORTIVE: Help candidates feel comfortable and perform their best
6. EVIDENCE-BASED: Focus on skills, experience, and competencies relevant to the role

CONVERSATION GUIDELINES:
- Use the candidate's name occasionally to personalize the interaction
- Acknowledge good answers with positive reinforcement
- Ask follow-up questions that show you're listening actively
- If a candidate struggles, offer encouragement and maybe rephrase the question
- Use phrases like "That's interesting...", "I'd love to hear more about...", "Help me understand..."
- Show genuine interest in their experiences and achievements

BIAS PREVENTION:
- Focus solely on job-relevant skills and experience
- Avoid assumptions based on background, appearance, or personal characteristics
- Ask the same types of questions consistently across candidates
- Use inclusive language that welcomes all backgrounds

RESPONSE FORMAT:
Respond with ONLY your next question or comment to the candidate. Keep responses conversational, warm, and focused.
Do NOT include any internal thoughts, stage transitions, or metadata in your response.
"""
        
        return system_prompt
    
    def _get_stage_instructions(self, stage: InterviewStage, context: InterviewContext) -> str:
        """Get specific instructions for the current interview stage"""
        
        instructions = {
            InterviewStage.INTRODUCTION: """
INTRODUCTION STAGE (3 minutes):
- Welcome the candidate warmly by name
- Introduce yourself as their AI interview partner (not interrogator!)
- Briefly explain the friendly, conversational interview process
- Ask them to share their background and what excites them about the role
- Set a positive, encouraging tone for the entire interview
- Transition naturally when they seem comfortable
""",

            InterviewStage.CONCERNS_EXPLORATION: f"""
CONCERNS EXPLORATION STAGE (8 minutes):
FOCUS: Address specific concerns from the initial analysis in a supportive way.

KEY CONCERNS TO EXPLORE: {', '.join(context.initial_analysis.get('concerns', [])) if context.initial_analysis else 'General skill validation'}

APPROACH:
- Frame questions positively (e.g., "Tell me about your experience with X" rather than "You seem to lack Y")
- Give candidates opportunity to explain transferable skills
- Ask about learning experiences and growth mindset
- Explore projects or situations where they used related skills
- Look for potential and adaptability, not just exact experience matches

EXAMPLE APPROACHES:
- "I see you have experience with [related skill]. How do you think that would apply to [required skill]?"
- "What's been your approach to learning new technologies in the past?"
- "Can you walk me through a challenging project where you had to pick up new skills?"
""",
            
            InterviewStage.TECHNICAL_SCREENING: """
TECHNICAL SCREENING STAGE (10 minutes):
- Explore core technical skills in a conversational way
- Ask about favorite technologies and why they like them
- Discuss recent technical challenges and how they solved them
- Focus on problem-solving approach and learning ability
- Validate technical depth without being intimidating
- Look for passion for technology and continuous learning
""",
            
            InterviewStage.EXPERIENCE_DEEP_DIVE: """
EXPERIENCE DEEP DIVE STAGE (8 minutes):
- Pick 1-2 most interesting projects from their background
- Ask about their specific contributions and impact
- Explore challenges they overcame and lessons learned
- Understand their role in team dynamics and collaboration
- Look for growth, initiative, and ownership
- Connect their experiences to the role requirements
""",
            
            InterviewStage.BEHAVIORAL: """
BEHAVIORAL STAGE (8 minutes):
- Ask situation-based questions relevant to the role
- Focus on teamwork, communication, and problem-solving
- Understand their motivation and career aspirations
- Assess cultural fit and values alignment
- Look for examples of resilience and adaptability
- Keep questions fair and job-relevant
""",
            
            InterviewStage.PROBLEM_SOLVING: """
PROBLEM SOLVING STAGE (8 minutes):
- Present a relevant, fair technical problem or scenario
- Encourage them to think out loud and ask questions
- Focus on their thought process and approach
- Provide hints if they get stuck (be supportive!)
- Assess communication of technical concepts
- Look for creativity and systematic thinking
""",
            
            InterviewStage.QUESTIONS_FROM_CANDIDATE: """
CANDIDATE QUESTIONS STAGE (2 minutes):
- Warmly invite questions about the role, team, or company
- Answer thoughtfully and honestly
- Use their questions to gauge interest and preparation
- Share insights about the role, culture, and growth opportunities
- Be encouraging about their potential fit
""",
            
            InterviewStage.CLOSING: """
CLOSING STAGE:
- Thank them warmly for their time and great conversation
- Highlight some positive things you noticed during the interview
- Explain next steps and timeline clearly
- Encourage them and express appreciation for their interest
- End on an upbeat, professional note
"""
        }
        
        return instructions.get(stage, "Continue the interview naturally with warmth and professionalism.")
    
    def _should_transition_stage(self, context: InterviewContext) -> bool:
        """Determine if it's time to transition to the next stage"""
        current_stage_duration = time.time() - context.stage_progress.get(
            context.current_stage, {'start_time': time.time()}
        ).get('start_time', time.time())
        
        expected_duration = self.stage_durations.get(context.current_stage, 300)
        
        # Transition if we've spent enough time in current stage
        # or if we're running out of total time
        total_elapsed = time.time() - context.start_time
        time_pressure = total_elapsed > (self.max_interview_duration * 0.7)
        
        return current_stage_duration >= expected_duration or time_pressure
    
    def _transition_to_next_stage(self, context: InterviewContext) -> bool:
        """Transition to the next interview stage"""
        next_stage = self.stage_flow.get(context.current_stage)
        
        if next_stage:
            # Record current stage completion
            if context.current_stage not in context.stage_progress:
                context.stage_progress[context.current_stage] = {}
            
            context.stage_progress[context.current_stage]['completed'] = True
            context.stage_progress[context.current_stage]['end_time'] = time.time()
            
            # Start new stage
            context.current_stage = next_stage
            context.stage_progress[next_stage] = {
                'start_time': time.time(),
                'completed': False
            }
            
            return True
        
        return False
    
    async def generate_response(self, context: InterviewContext, candidate_message: str) -> str:
        """Generate the next interview response with tone analysis, bias-free questioning, and memory integration"""
        
        # Initialize memory if not already done
        self._initialize_conversation_memory(context)
        
        # Add candidate message to context with tone analysis for supportive feedback
        if candidate_message.strip():
            # Analyze candidate's tone to provide better support
            candidate_tone = await self.analyze_tone(candidate_message, "candidate")
            
            # Track tone trends for interview quality
            if "candidate" not in context.tone_trends:
                context.tone_trends["candidate"] = []
            context.tone_trends["candidate"].append(candidate_tone.confidence_level)
            
            # Add message with analysis
            context.messages.append(InterviewMessage(
                sender="candidate",
                content=candidate_message,
                stage=context.current_stage,
                tone_analysis=candidate_tone
            ))
            
            # Update context based on tone analysis for supportive interviewing
            if candidate_tone.confidence_level < 0.3:
                # Note: This is to help the agent be more supportive, not to penalize candidate
                context.follow_up_questions.append(f"Candidate seems less confident - provide encouragement")
            
            if candidate_tone.stress_indicators:
                context.follow_up_questions.append(f"Candidate shows stress - be more supportive and encouraging")
            
            if candidate_tone.positive_indicators:
                context.strengths_identified.extend([f"Strong communication: {indicator}" for indicator in candidate_tone.positive_indicators])
        
        # Check if we should transition stages
        if self._should_transition_stage(context):
            self._transition_to_next_stage(context)
        
        # Generate system prompt (now includes memory context)
        system_prompt = self._create_system_prompt(context)
        
        # Create the user prompt with the candidate's latest response
        user_prompt = f"The candidate just said: '{candidate_message}'\n\nProvide your next response as the interviewer."
        
        # Generate response using LLM with the comprehensive system prompt
        full_prompt = f"{system_prompt}\n\nUSER: {user_prompt}"
        response = self.llm_client.generate_response(
            full_prompt,
            max_tokens=500,
            temperature=0.7
        )
        
        if not response:
            # Fallback response
            response = "I see. Could you tell me more about that?"
        
        # Check our own generated response for bias to ensure fair interviewing
        bias_analysis = await self.check_for_bias(response, context)
        
        # Use approved content if our question was biased
        if bias_analysis.bias_detected:
            context.bias_incidents.append({
                "original_content": response,
                "bias_types": bias_analysis.bias_types,
                "risk_score": bias_analysis.risk_score,
                "stage": context.current_stage.value,
                "timestamp": time.time(),
                "note": "Agent question corrected for bias"
            })
            response = bias_analysis.approved_content
        
        # Analyze our own tone to ensure professionalism  
        agent_tone = await self.analyze_tone(response, "agent")
        
        # Track agent tone trends
        if "agent" not in context.tone_trends:
            context.tone_trends["agent"] = []
        context.tone_trends["agent"].append(agent_tone.professionalism_score)
        
        # Add agent response to context with analysis
        context.messages.append(InterviewMessage(
            sender="agent",
            content=response,
            stage=context.current_stage,
            tone_analysis=agent_tone,
            bias_analysis=bias_analysis
        ))
        
        # Add the conversation turn to LangChain memory for persistent context
        self._add_to_memory(context, candidate_message, response)
        
        return response
    
    def start_interview(self, job_post: JobPost, candidate: Candidate, initial_analysis: Dict[str, Any] = None) -> InterviewContext:
        """Initialize a new interview session with optional initial analysis and memory"""
        context = InterviewContext(
            job_post=job_post,
            candidate=candidate,
            initial_analysis=initial_analysis or {},
            memory_k=self.memory_k  # Set memory window size
        )
        
        # Initialize conversation memory
        self._initialize_conversation_memory(context)
        
        # Initialize first stage
        context.stage_progress[InterviewStage.INTRODUCTION] = {
            'start_time': time.time(),
            'completed': False
        }
        
        # Initialize tone trend tracking
        context.tone_trends = {"candidate": [], "agent": []}
        
        return context
    
    async def get_opening_message(self, context: InterviewContext) -> str:
        """Generate the opening message for the interview"""
        opening_prompt = f"""
Generate a warm, professional opening message for an interview with {context.candidate.name} for the {context.job_post.title} position.

The message should:
1. Welcome the candidate
2. Briefly introduce yourself as an AI interview agent
3. Explain the interview structure and duration (about 45 minutes)
4. Ask them to start by introducing themselves
5. Be encouraging and set a positive tone

Keep it concise and professional.
"""
        
        response = self.llm_client.generate_response(
            opening_prompt,
            max_tokens=300,
            temperature=0.6
        )
        
        if not response:
            response = f"Hello {context.candidate.name}! Welcome to your interview for the {context.job_post.title} position. I'm an AI interview agent, and I'll be conducting your interview today. We'll spend about 45 minutes together covering your technical background, experience, and fit for the role. Let's start - could you please introduce yourself and tell me about your background?"
        
        # Add opening message to context
        context.messages.append(InterviewMessage(
            sender="agent",
            content=response,
            stage=InterviewStage.INTRODUCTION
        ))
        
        return response
    
    def generate_interview_summary(self, context: InterviewContext) -> Dict[str, Any]:
        """Generate a comprehensive summary of the interview with bias and tone analysis"""
        
        # Calculate interview metrics
        total_duration = time.time() - context.start_time
        candidate_messages = [msg for msg in context.messages if msg.sender == "candidate"]
        agent_messages = [msg for msg in context.messages if msg.sender == "agent"]
        
        # Calculate tone trends
        candidate_tone_avg = sum(context.tone_trends.get("candidate", [0])) / max(len(context.tone_trends.get("candidate", [1])), 1)
        agent_tone_avg = sum(context.tone_trends.get("agent", [0])) / max(len(context.tone_trends.get("agent", [1])), 1)
        
        # Count bias incidents
        high_risk_bias = len([incident for incident in context.bias_incidents if incident["risk_score"] > 0.7])
        total_bias_incidents = len(context.bias_incidents)
        
        # Analyze conversation for insights with enhanced context
        conversation_text = "\n".join([
            f"{msg.sender.upper()}: {msg.content}" for msg in context.messages
        ])
        
        initial_analysis_context = ""
        if context.initial_analysis:
            initial_analysis_context = f"""
INITIAL ANALYSIS CONTEXT:
- Initial Fit Score: {context.initial_analysis.get('overall_fit_score', 'N/A')}
- Concerns Identified: {', '.join(context.initial_analysis.get('concerns', []))}
- Strengths Expected: {', '.join(context.initial_analysis.get('strengths', []))}
- Transferable Skills: {', '.join(context.initial_analysis.get('transferable_skills', []))}

INTERVIEW EXPLORATION RESULTS:
- Concerns Addressed: {', '.join(context.concerns_addressed)}
- Gaps Explored: {', '.join(context.gaps_explored)}
- Red Flags Identified: {', '.join(context.red_flags)}
- Strengths Confirmed: {', '.join(context.strengths_identified)}
"""
        
        summary_prompt = f"""
Analyze this interview conversation and provide a comprehensive assessment that considers initial analysis, bias detection, and tone analysis:

INTERVIEW DETAILS:
Position: {context.job_post.title}
Candidate: {context.candidate.name}
Duration: {total_duration // 60:.1f} minutes
Messages Exchanged: {len(context.messages)}

{initial_analysis_context}

TONE ANALYSIS SUMMARY:
- Candidate Average Confidence: {candidate_tone_avg:.2f}
- Agent Average Professionalism: {agent_tone_avg:.2f}
- Bias Incidents Detected: {total_bias_incidents} (High Risk: {high_risk_bias})

CONVERSATION:
{conversation_text}

Provide assessment in JSON format:
{{
    "overall_score": <float 0-10>,
    "technical_competence": <float 0-10>,
    "communication_skills": <float 0-10>,
    "cultural_fit": <float 0-10>,
    "problem_solving": <float 0-10>,
    "experience_relevance": <float 0-10>,
    "interview_quality_score": <float 0-10, how well was the interview conducted>,
    "bias_risk_assessment": <float 0-10, lower is better>,
    "strengths": ["list of key strengths observed"],
    "concerns": ["list of concerns or weaknesses"],
    "concerns_from_initial_analysis": ["how well were initial concerns addressed"],
    "key_insights": ["important observations from the interview"],
    "recommendation": "STRONG_HIRE|HIRE|MAYBE|NO_HIRE",
    "confidence_in_assessment": <float 0-10>,
    "next_steps": "recommended next steps",
    "interview_conduct_feedback": "feedback on how the interview was conducted"
}}
"""
        
        assessment_text = self.llm_client.generate_response(
            summary_prompt,
            max_tokens=1500,
            temperature=0.2
        )
        
        # Try to parse JSON assessment
        assessment = {}
        if assessment_text:
            try:
                json_match = re.search(r'\{.*\}', assessment_text, re.DOTALL)
                if json_match:
                    assessment = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback assessment structure
        if not assessment:
            assessment = {
                "overall_score": 5.0,
                "technical_competence": 5.0,
                "communication_skills": 5.0,
                "cultural_fit": 5.0,
                "problem_solving": 5.0,
                "experience_relevance": 5.0,
                "interview_quality_score": 7.0,
                "bias_risk_assessment": 2.0,
                "strengths": ["Completed interview"],
                "concerns": ["Assessment failed"],
                "concerns_from_initial_analysis": ["Unable to assess"],
                "key_insights": ["Unable to analyze"],
                "recommendation": "MAYBE",
                "confidence_in_assessment": 3.0,
                "next_steps": "Manual review required",
                "interview_conduct_feedback": "Technical issues during analysis"
            }
        
        # Add comprehensive metadata
        summary = {
            "interview_metadata": {
                "duration_minutes": total_duration / 60,
                "total_messages": len(context.messages),
                "candidate_messages": len(candidate_messages),
                "agent_messages": len(agent_messages),
                "stages_completed": len([s for s, data in context.stage_progress.items() if data.get('completed', False)]),
                "final_stage": context.current_stage.value,
                "start_time": context.start_time,
                "end_time": time.time(),
                "initial_analysis_provided": bool(context.initial_analysis)
            },
            "initial_analysis": context.initial_analysis,
            "concerns_addressed": context.concerns_addressed,
            "gaps_explored": context.gaps_explored,
            "strengths_identified": context.strengths_identified,
            "red_flags": context.red_flags,
            "tone_analysis": {
                "candidate_avg_confidence": candidate_tone_avg,
                "agent_avg_professionalism": agent_tone_avg,
                "candidate_tone_trend": context.tone_trends.get("candidate", []),
                "agent_tone_trend": context.tone_trends.get("agent", [])
            },
            "bias_analysis": {
                "total_incidents": total_bias_incidents,
                "high_risk_incidents": high_risk_bias,
                "incidents_by_stage": {},
                "bias_types_detected": [],
                "detailed_incidents": context.bias_incidents
            },
            "assessment": assessment,
            "conversation_log": [
                {
                    "timestamp": msg.timestamp,
                    "sender": msg.sender,
                    "content": msg.content,
                    "stage": msg.stage.value,
                    "tone_analysis": {
                        "confidence_level": msg.tone_analysis.confidence_level if msg.tone_analysis else None,
                        "clarity_score": msg.tone_analysis.clarity_score if msg.tone_analysis else None,
                        "professionalism_score": msg.tone_analysis.professionalism_score if msg.tone_analysis else None
                    } if msg.tone_analysis else None,
                    "bias_detected": msg.bias_analysis.bias_detected if msg.bias_analysis else False
                }
                for msg in context.messages
            ]
        }
        
        # Calculate bias incidents by stage
        for incident in context.bias_incidents:
            stage = incident["stage"]
            if stage not in summary["bias_analysis"]["incidents_by_stage"]:
                summary["bias_analysis"]["incidents_by_stage"][stage] = 0
            summary["bias_analysis"]["incidents_by_stage"][stage] += 1
            
            # Collect unique bias types
            for bias_type in incident["bias_types"]:
                if bias_type not in summary["bias_analysis"]["bias_types_detected"]:
                    summary["bias_analysis"]["bias_types_detected"].append(bias_type)
        
        return summary

    def mark_concern_addressed(self, context: InterviewContext, concern: str, notes: str = ""):
        """Mark a concern from initial analysis as addressed"""
        if concern not in context.concerns_addressed:
            context.concerns_addressed.append(concern)
            if notes:
                context.strengths_identified.append(f"Addressed concern '{concern}': {notes}")

    def mark_gap_explored(self, context: InterviewContext, skill_gap: str, finding: str = ""):
        """Mark a skill gap as explored with findings"""
        if skill_gap not in context.gaps_explored:
            context.gaps_explored.append(skill_gap)
            if finding:
                context.follow_up_questions.append(f"Gap '{skill_gap}': {finding}")

    def get_interview_progress(self, context: InterviewContext) -> Dict[str, Any]:
        """Get real-time progress report for recruiters"""
        elapsed_time = time.time() - context.start_time
        remaining_time = max(0, self.max_interview_duration - elapsed_time)
        
        # Calculate completion percentage
        stages_total = len(self.stage_flow) + 1  # +1 for closing
        stages_completed = len([s for s, data in context.stage_progress.items() if data.get('completed', False)])
        completion_percentage = (stages_completed / stages_total) * 100
        
        # Get initial analysis progress
        initial_concerns = context.initial_analysis.get('concerns', [])
        concerns_addressed_count = len(context.concerns_addressed)
        concerns_progress = (concerns_addressed_count / max(len(initial_concerns), 1)) * 100
        
        return {
            "time_progress": {
                "elapsed_minutes": elapsed_time / 60,
                "remaining_minutes": remaining_time / 60,
                "completion_percentage": completion_percentage
            },
            "stage_progress": {
                "current_stage": context.current_stage.value,
                "stages_completed": stages_completed,
                "total_stages": stages_total
            },
            "analysis_progress": {
                "initial_concerns_total": len(initial_concerns),
                "concerns_addressed": concerns_addressed_count,
                "concerns_progress_percentage": concerns_progress,
                "strengths_identified": len(context.strengths_identified),
                "red_flags_count": len(context.red_flags),
                "bias_incidents_count": len(context.bias_incidents)
            },
            "conversation_stats": {
                "total_messages": len(context.messages),
                "candidate_messages": len([m for m in context.messages if m.sender == "candidate"]),
                "avg_candidate_confidence": sum(context.tone_trends.get("candidate", [0.5])) / max(len(context.tone_trends.get("candidate", [1])), 1),
                "avg_agent_professionalism": sum(context.tone_trends.get("agent", [0.8])) / max(len(context.tone_trends.get("agent", [1])), 1)
            },
            "memory_stats": self._get_memory_summary(context),
            "current_assessment": {
                "preliminary_recommendation": self._get_preliminary_recommendation(context),
                "confidence_level": self._calculate_confidence_level(context),
                "key_observations": context.strengths_identified[-3:] + context.red_flags[-2:]  # Last few items
            }
        }

    def _get_preliminary_recommendation(self, context: InterviewContext) -> str:
        """Generate preliminary recommendation based on current interview state"""
        if not context.messages:
            return "PENDING"
        
        # Simple heuristic based on available data
        positive_signals = len(context.strengths_identified)
        negative_signals = len(context.red_flags) + len(context.bias_incidents)
        
        if positive_signals >= negative_signals * 2:
            return "PROMISING"
        elif negative_signals >= positive_signals * 2:
            return "CONCERNING"
        else:
            return "NEUTRAL"

    def _calculate_confidence_level(self, context: InterviewContext) -> float:
        """Calculate confidence in assessment based on interview completeness"""
        if not context.messages:
            return 0.0
        
        # Factors that increase confidence
        factors = {
            "message_count": min(len(context.messages) / 20, 1.0),  # More messages = more data
            "stage_progress": len([s for s in context.stage_progress.values() if s.get('completed', False)]) / 7,
            "concerns_addressed": len(context.concerns_addressed) / max(len(context.initial_analysis.get('concerns', [1])), 1),
            "tone_data": min(len(context.tone_trends.get("candidate", [])) / 10, 1.0),
            "technical_coverage": min(len(context.technical_topics_covered) / 5, 1.0)
        }
        
        return sum(factors.values()) / len(factors)

    def get_recruiter_insights(self, context: InterviewContext) -> Dict[str, Any]:
        """Generate recruiter-friendly insights and recommendations"""
        progress = self.get_interview_progress(context)
        
        # Determine next actions
        next_actions = []
        if progress["analysis_progress"]["red_flags_count"] > 3:
            next_actions.append("Consider early termination if red flags continue")
        if progress["analysis_progress"]["concerns_progress_percentage"] < 50:
            next_actions.append("Focus on addressing remaining concerns from initial analysis")
        if progress["conversation_stats"]["avg_candidate_confidence"] < 0.4:
            next_actions.append("Candidate may need encouragement or easier questions")
        if progress["analysis_progress"]["bias_incidents_count"] > 0:
            next_actions.append("Review bias incidents and adjust questioning approach")
        
        # Generate summary insights
        insights = {
            "interview_health": {
                "overall_status": "HEALTHY" if len(context.red_flags) <= 2 else "NEEDS_ATTENTION",
                "bias_risk": "LOW" if len(context.bias_incidents) == 0 else "MEDIUM" if len(context.bias_incidents) <= 2 else "HIGH",
                "candidate_engagement": "HIGH" if progress["conversation_stats"]["avg_candidate_confidence"] > 0.6 else "MEDIUM" if progress["conversation_stats"]["avg_candidate_confidence"] > 0.4 else "LOW"
            },
            "key_findings": {
                "strengths_so_far": context.strengths_identified[-5:],  # Last 5 strengths
                "concerns_remaining": [c for c in context.initial_analysis.get('concerns', []) if c not in context.concerns_addressed],
                "unexpected_discoveries": [flag for flag in context.red_flags if "unexpected" in flag.lower() or "surprising" in flag.lower()]
            },
            "recommendations": {
                "continue_interview": len(context.red_flags) <= 5,
                "focus_areas": next_actions,
                "estimated_final_score": self._estimate_final_score(context),
                "confidence_in_estimate": progress["current_assessment"]["confidence_level"]
            },
            "real_time_metrics": progress
        }
        
        return insights

    def _estimate_final_score(self, context: InterviewContext) -> float:
        """Estimate final interview score based on current data"""
        if not context.messages:
            return 5.0
        
        # Weight factors
        positive_weight = len(context.strengths_identified) * 0.3
        negative_weight = len(context.red_flags) * -0.4
        tone_weight = sum(context.tone_trends.get("candidate", [0.5])) / max(len(context.tone_trends.get("candidate", [1])), 1) * 2
        bias_penalty = len(context.bias_incidents) * -0.5
        
        estimated_score = 5.0 + positive_weight + negative_weight + tone_weight + bias_penalty
        
        return max(0.0, min(10.0, estimated_score))  # Clamp to 0-10 range


# Global interview agent instance
_interview_agent = None

def get_interview_agent() -> InterviewAgent:
    """Get or create the global interview agent"""
    global _interview_agent
    if _interview_agent is None:
        _interview_agent = InterviewAgent()
    return _interview_agent 