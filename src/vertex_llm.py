import os
import json
import time
import random
from typing import Dict, Any, Optional

try:
    from vertexai.generative_models import GenerativeModel
    import vertexai
    HAS_VERTEX_AI = True
except ImportError:
    HAS_VERTEX_AI = False
    print("⚠️  google-cloud-aiplatform not found. Install with: pip install google-cloud-aiplatform")


class VertexAIClient:
    """Client for interacting with Gemini Flash via Google Vertex AI SDK with rate limiting and retry logic"""
    
    def __init__(self):
        if not HAS_VERTEX_AI:
            raise ImportError("google-cloud-aiplatform is required")
            
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        self.region = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        self.model_id = 'gemini-2.0-flash-001'
        self.max_output_tokens = 1024
        self.temperature = 0.7
        
        # Rate limiting and retry configuration
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        self.last_request_time = 0
        self.max_retries = 3
        self.base_delay = 2.0  # Base delay for exponential backoff
        
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.region)
        self.model = GenerativeModel(self.model_id)
    
    def _wait_for_rate_limit(self):
        """Ensure minimum interval between requests to avoid rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def generate_response(self, prompt: str, max_tokens: int = None, temperature: float = None) -> Optional[str]:
        """Generate response using Gemini Flash via SDK with retry logic and rate limiting"""
        
        # Rate limiting
        self._wait_for_rate_limit()
        
        generation_config = {
            "max_output_tokens": max_tokens or self.max_output_tokens,
            "temperature": temperature or self.temperature,
            "top_p": 0.9,
        }
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                if response and response.text:
                    return response.text.strip()
                else:
                    print("No response text received")
                    return None
                    
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limiting (429 errors)
                if "429" in error_str or "Resource exhausted" in error_str:
                    if attempt < self.max_retries - 1:  # Don't sleep on last attempt
                        # Exponential backoff with jitter
                        delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}). Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"Rate limit exceeded after {self.max_retries} attempts: {e}")
                        return None
                
                # Handle other errors
                elif "500" in error_str or "Internal error" in error_str:
                    if attempt < self.max_retries - 1:
                        delay = 1.0 + random.uniform(0, 1)  # Shorter delay for server errors
                        print(f"Server error (attempt {attempt + 1}/{self.max_retries}). Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"Server error after {self.max_retries} attempts: {e}")
                        return None
                
                # For other errors, don't retry
                else:
                    print(f"Non-retryable error generating response: {e}")
                    return None
        
        # If we get here, all retries failed
        print(f"Failed to generate response after {self.max_retries} attempts")
        return None

    def analyze_candidate_fit(self, job_description: str, candidate_profile: str) -> Dict[str, Any]:
        """Analyze how well a candidate fits a job using LLM reasoning"""
        prompt = f"""
You are an expert technical interviewer and recruiter. Analyze the following candidate profile against the job requirements. Use chain-of-thought reasoning to assess both direct and transferable skills. If the candidate has related experience (e.g., Q-learning for PPO, PyTorch for TensorFlow, AWS for GCP), explain how these skills transfer and what learning curve might be expected. If the information is insufficient, state what deeper search or additional data would be needed, and simulate a deeper reasoning process based on what is typical for such profiles.

Provide your analysis in the following JSON format:
{{
  "overall_fit_score": <float between 0.0 and 1.0>,
  "reasoning": "<multi-paragraph, recruiter-friendly summary explaining the fit, skill transfer, and any caveats. Be specific about which experiences or projects are relevant.>",
  "strengths": ["<list of candidate strengths relevant to the role>"],
  "concerns": ["<list of potential concerns or gaps>"],
  "transferable_skills": ["<skills that could transfer even if not exact match>"],
  "recommendation": "<STRONG_MATCH|GOOD_MATCH|WEAK_MATCH|NO_MATCH>"
}}

### Example 1:
JOB DESCRIPTION:
Looking for an RL engineer with PPO or DPO experience, ideally using OpenAI Gym.

CANDIDATE PROFILE:
- Implemented Q-learning and MDP in Python
- Used Gymnasium (OpenAI Gym fork) for RL environments
- No direct PPO/DPO, but strong RL fundamentals

EXPECTED OUTPUT:
{{
  "overall_fit_score": 0.8,
  "reasoning": "The candidate has implemented core RL algorithms (Q-learning, MDP) and used Gymnasium, which is directly relevant to the job's requirements. While they have not implemented PPO or DPO specifically, their experience with RL fundamentals and Gym environments suggests a strong ability to learn and apply these algorithms. The learning curve for PPO/DPO would be moderate, but the candidate's background indicates high potential for success.",
  "strengths": ["RL fundamentals", "Python", "Gymnasium"],
  "concerns": ["No direct PPO/DPO experience"],
  "transferable_skills": ["Q-learning", "MDP", "Gymnasium"],
  "recommendation": "GOOD_MATCH"
}}

### Example 2:
JOB DESCRIPTION:
Seeking a cloud engineer with GCP experience.

CANDIDATE PROFILE:
- 2 years AWS DevOps
- Deployed ML models on Azure
- No GCP, but strong cloud background

EXPECTED OUTPUT:
{{
  "overall_fit_score": 0.7,
  "reasoning": "The candidate has significant experience with AWS and Azure, which are highly transferable to GCP. While there will be some learning curve for GCP-specific tools, their cloud engineering background and DevOps skills are directly relevant. The candidate is likely to ramp up quickly.",
  "strengths": ["AWS", "Azure", "DevOps"],
  "concerns": ["No direct GCP experience"],
  "transferable_skills": ["Cloud engineering", "DevOps"],
  "recommendation": "GOOD_MATCH"
}}

### Example 3:
JOB DESCRIPTION:
Looking for a data scientist with experience in PyTorch and NLP.

CANDIDATE PROFILE:
- 3 years TensorFlow, Keras
- Built NLP models for sentiment analysis
- No PyTorch

EXPECTED OUTPUT:
{{
  "overall_fit_score": 0.6,
  "reasoning": "The candidate has strong experience in deep learning and NLP, but their work has been exclusively in TensorFlow and Keras. While the transition to PyTorch is feasible, it will require some ramp-up time. Their NLP experience is a strong asset.",
  "strengths": ["Deep learning", "NLP", "TensorFlow", "Keras"],
  "concerns": ["No PyTorch experience"],
  "transferable_skills": ["NLP", "Deep learning"],
  "recommendation": "WEAK_MATCH"
}}

Now, analyze the following:

JOB DESCRIPTION:
{job_description}

CANDIDATE PROFILE:
{candidate_profile}
"""
        response = self.generate_response(prompt, max_tokens=1800, temperature=0.2)
        
        if response:
            try:
                # Try to parse JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback if JSON parsing fails
        return {
            "overall_fit_score": 0.5,
            "reasoning": "Unable to analyze candidate fit",
            "strengths": [],
            "concerns": ["Analysis failed"],
            "transferable_skills": [],
            "recommendation": "WEAK_MATCH"
        }

    def analyze_projects_and_decide_deep_dive(self, job_description: str, project_summaries: list) -> Dict[str, Any]:
        """Analyze project summaries and decide if deeper analysis is needed for specific repos"""
        projects_text = ""
        for proj in project_summaries:
            projects_text += f"""
Repository: {proj['repo_name']} ({proj['stars']} stars)
Description: {proj['description']}
Tools detected: {', '.join(proj['detected_tools'])}
Languages: {', '.join(proj['languages'])}
Topics: {', '.join(proj['topics'])}
README excerpt: {proj['readme_content'][:300]}...
---
"""
        
        prompt = f"""
You are an expert technical recruiter analyzing a candidate's GitHub portfolio. Based on the job requirements and the candidate's project summaries below, provide:

1. An initial assessment of the candidate's fit
2. Identify which specific repositories (if any) need deeper code-level analysis
3. Explain your reasoning for each recommendation

JOB DESCRIPTION:
{job_description}

CANDIDATE'S PROJECT SUMMARIES:
{projects_text}

Provide your analysis in JSON format:
{{
  "initial_assessment": {{
    "overall_fit_score": <float 0.0-1.0>,
    "reasoning": "<detailed multi-paragraph assessment>",
    "strengths": ["<list of strengths>"],
    "concerns": ["<list of concerns>"]
  }},
  "deep_dive_recommendations": [
    {{
      "repo_name": "<repository name>",
      "priority": "<HIGH|MEDIUM|LOW>",
      "reason": "<why this repo needs deeper analysis>",
      "specific_focus": "<what to look for in the code>"
    }}
  ],
  "sufficient_for_decision": <true/false>,
  "recommendation": "<STRONG_MATCH|GOOD_MATCH|WEAK_MATCH|NO_MATCH>"
}}
"""
        
        response = self.generate_response(prompt, max_tokens=2000, temperature=0.2)
        
        if response:
            try:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return {
            "initial_assessment": {
                "overall_fit_score": 0.5,
                "reasoning": "Unable to analyze projects",
                "strengths": [],
                "concerns": ["Analysis failed"]
            },
            "deep_dive_recommendations": [],
            "sufficient_for_decision": True,
            "recommendation": "WEAK_MATCH"
        }

    def analyze_deep_dive_results(self, job_description: str, initial_assessment: dict, deep_dive_results: list) -> Dict[str, Any]:
        """Final analysis after deep diving into specific repositories"""
        deep_dive_text = ""
        for result in deep_dive_results:
            if "error" not in result:
                deep_dive_text += f"""
Repository: {result['repo_name']}
Deep analysis found: {', '.join(result['deep_tools'])}
Files analyzed: {result['files_analyzed']}
Key findings: {result['file_analysis'][:3]}  # First 3 files
---
"""
        
        prompt = f"""
You are an expert technical recruiter. You previously provided an initial assessment of a candidate based on their project summaries. Now you have deeper code-level analysis of specific repositories. Provide a final, comprehensive assessment.

JOB DESCRIPTION:
{job_description}

INITIAL ASSESSMENT:
{json.dumps(initial_assessment, indent=2)}

DEEP DIVE RESULTS:
{deep_dive_text}

Provide your final analysis in JSON format:
{{
  "final_fit_score": <float 0.0-1.0>,
  "reasoning": "<comprehensive multi-paragraph assessment incorporating both initial and deep findings>",
  "strengths": ["<updated list of strengths>"],
  "concerns": ["<updated list of concerns>"],
  "transferable_skills": ["<skills that could transfer>"],
  "evidence_found": ["<specific evidence from code analysis>"],
  "recommendation": "<STRONG_MATCH|GOOD_MATCH|WEAK_MATCH|NO_MATCH>",
  "confidence_level": "<HIGH|MEDIUM|LOW>"
}}
"""
        
        response = self.generate_response(prompt, max_tokens=2500, temperature=0.2)
        
        if response:
            try:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return {
            "final_fit_score": 0.5,
            "reasoning": "Unable to complete final analysis",
            "strengths": [],
            "concerns": ["Analysis failed"],
            "transferable_skills": [],
            "evidence_found": [],
            "recommendation": "WEAK_MATCH",
            "confidence_level": "LOW"
        }


# Global client instance
_vertex_client = None

def get_vertex_client() -> VertexAIClient:
    """Get or create the global Vertex AI client"""
    global _vertex_client
    if _vertex_client is None:
        _vertex_client = VertexAIClient()
    return _vertex_client 