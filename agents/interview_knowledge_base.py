#!/usr/bin/env python3
"""
RAG-based Interview Knowledge Base
Reduces API calls by using pre-generated questions and cached responses
"""

import json
import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import time

# Simple vector similarity (can be upgraded to proper embeddings later)
def simple_similarity(text1: str, text2: str) -> float:
    """Basic similarity scoring using word overlap"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

@dataclass
class QuestionTemplate:
    """A reusable interview question template"""
    id: str
    stage: str
    category: str
    template: str
    variables: List[str] = field(default_factory=list)
    follow_ups: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    difficulty_level: str = "medium"  # easy, medium, hard

@dataclass
class ResponseTemplate:
    """A cached response template"""
    trigger_keywords: List[str]
    response_template: str
    variables: List[str] = field(default_factory=list)
    confidence_score: float = 0.8

@dataclass
class CachedResponse:
    """A fully cached response"""
    input_hash: str
    response: str
    timestamp: float
    usage_count: int = 0

class InterviewKnowledgeBase:
    """RAG-based knowledge system for efficient interviews"""
    
    def __init__(self):
        self.question_templates: Dict[str, List[QuestionTemplate]] = {}
        self.response_templates: List[ResponseTemplate] = []
        self.cached_responses: Dict[str, CachedResponse] = {}
        self.role_specific_questions: Dict[str, List[QuestionTemplate]] = {}
        
        # Initialize with default templates
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default question and response templates"""
        
        # Introduction Stage Questions
        intro_questions = [
            QuestionTemplate(
                id="intro_welcome",
                stage="introduction",
                category="welcome",
                template="Hello {candidate_name}! Welcome to your interview for the {job_title} position. I'm excited to learn more about your background. Could you start by telling me a bit about yourself and what interests you about this role?",
                variables=["candidate_name", "job_title"]
            ),
            QuestionTemplate(
                id="intro_background",
                stage="introduction", 
                category="background",
                template="That's great! Could you walk me through your professional journey and how you got into {field}?",
                variables=["field"],
                follow_ups=["What's been the most exciting project you've worked on?"]
            )
        ]
        
        # Store by stage
        self.question_templates = {
            "introduction": intro_questions
        }
        
        # Response Templates for Common Situations
        self.response_templates = [
            ResponseTemplate(
                trigger_keywords=["thank you", "thanks", "appreciate"],
                response_template="You're welcome! {encouragement}"
            ),
            ResponseTemplate(
                trigger_keywords=["nervous", "anxious", "worried"],
                response_template="That's completely normal! Take your time, and remember this is just a conversation. {reassurance}"
            )
                 ]
    
    def get_question_for_stage(self, stage: str, context: Dict[str, Any]) -> Optional[QuestionTemplate]:
        """Get an appropriate question for the given stage and context"""
        
        # Get questions for this stage
        stage_questions = self.question_templates.get(stage, [])
        
        if not stage_questions:
            return None
        
        # For now, return a random question from the stage
        import random
        return random.choice(stage_questions)
    
    def fill_question_template(self, template: QuestionTemplate, context: Dict[str, Any]) -> str:
        """Fill a question template with context variables"""
        question = template.template
        
        # Replace variables with context values
        for var in template.variables:
            if var in context:
                placeholder = "{" + var + "}"
                question = question.replace(placeholder, str(context[var]))
        
        return question
    
    def get_cached_response(self, input_text: str, context: Dict[str, Any]) -> Optional[str]:
        """Get a cached response if available"""
        
        # Create hash of input and relevant context
        context_str = json.dumps(sorted(context.items()), sort_keys=True)
        input_hash = hashlib.md5(f"{input_text}_{context_str}".encode()).hexdigest()
        
        if input_hash in self.cached_responses:
            cached = self.cached_responses[input_hash]
            cached.usage_count += 1
            return cached.response
        
        return None
    
    def cache_response(self, input_text: str, response: str, context: Dict[str, Any]):
        """Cache a response for future use"""
        
        context_str = json.dumps(sorted(context.items()), sort_keys=True)
        input_hash = hashlib.md5(f"{input_text}_{context_str}".encode()).hexdigest()
        
        self.cached_responses[input_hash] = CachedResponse(
            input_hash=input_hash,
            response=response,
            timestamp=time.time()
        )
    
    def get_template_response(self, input_text: str, context: Dict[str, Any]) -> Optional[str]:
        """Get a response using templates if applicable"""
        
        input_lower = input_text.lower()
        
        # Find matching response template
        best_match = None
        best_score = 0.0
        
        for template in self.response_templates:
            score = 0.0
            for keyword in template.trigger_keywords:
                if keyword in input_lower:
                    score += 1.0
            
            # Normalize by number of keywords
            score = score / len(template.trigger_keywords) if template.trigger_keywords else 0.0
            
            if score > best_score and score > 0.3:  # Minimum threshold
                best_match = template
                best_score = score
        
        if best_match:
            response = best_match.response_template
            
            # Fill in variables if needed
            if "{encouragement}" in response:
                encouragements = ["You're doing great!", "Keep going!", "That shows good thinking."]
                response = response.replace("{encouragement}", random.choice(encouragements))
            
            if "{reassurance}" in response:
                reassurances = ["There are no wrong answers here.", "We're just having a conversation.", "Take your time to think it through."]
                response = response.replace("{reassurance}", random.choice(reassurances))
            
            return response
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base usage statistics"""
        return {
            "total_question_templates": sum(len(questions) for questions in self.question_templates.values()),
            "total_response_templates": len(self.response_templates),
            "cached_responses": len(self.cached_responses),
            "cache_hit_rate": sum(c.usage_count for c in self.cached_responses.values())
        }

# Global knowledge base instance
_knowledge_base = None

def get_knowledge_base() -> InterviewKnowledgeBase:
    """Get or create the global knowledge base"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = InterviewKnowledgeBase()
    return _knowledge_base 