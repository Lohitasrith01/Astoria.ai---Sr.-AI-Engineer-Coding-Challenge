import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
import concurrent.futures
from .models import JobPost, Candidate
from .vertex_llm import get_vertex_client


class SmartFilter:
    """Two-layer semantic filtering system for candidates"""
    
    def __init__(self):
        # Load lightweight embedding model for Layer A
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vertex_client = get_vertex_client()
        
        # Thresholds for filtering
        self.embedding_threshold = 0.3  # Minimum similarity for Layer A
        self.llm_threshold = 0.6  # Minimum LLM score for Layer B
        
    def _create_candidate_text(self, candidate: Candidate) -> str:
        """Create a comprehensive text representation of candidate"""
        parts = []
        
        # Basic info
        if candidate.name:
            parts.append(f"Name: {candidate.name}")
        if candidate.email:
            parts.append(f"Email: {candidate.email}")
        
        # Skills and experience
        if candidate.skills:
            parts.append(f"Skills: {', '.join(candidate.skills)}")
        if candidate.experience_years:
            parts.append(f"Experience: {candidate.experience_years} years")
        
        # Work history
        if candidate.work_history:
            work_items = []
            for work in candidate.work_history:
                work_text = f"{work.get('position', 'Unknown')} at {work.get('company', 'Unknown')}"
                if work.get('duration'):
                    work_text += f" ({work['duration']})"
                if work.get('description'):
                    work_text += f" - {work['description']}"
                work_items.append(work_text)
            parts.append(f"Work History: {' | '.join(work_items)}")
        
        # Education
        if candidate.education:
            edu_items = []
            for edu in candidate.education:
                edu_text = f"{edu.get('degree', 'Unknown')} from {edu.get('institution', 'Unknown')}"
                if edu.get('year'):
                    edu_text += f" ({edu['year']})"
                edu_items.append(edu_text)
            parts.append(f"Education: {' | '.join(edu_items)}")
        
        # Additional info
        if candidate.location:
            parts.append(f"Location: {candidate.location}")
        if candidate.github_username:
            parts.append(f"GitHub: {candidate.github_username}")
        if candidate.linkedin_url:
            parts.append(f"LinkedIn: {candidate.linkedin_url}")
        
        return " | ".join(parts)
    
    def _create_job_text(self, job: JobPost) -> str:
        """Create a comprehensive text representation of job"""
        parts = []
        
        parts.append(f"Title: {job.title}")
        parts.append(f"Description: {job.description}")
        
        if job.required_skills:
            parts.append(f"Required Skills: {', '.join(job.required_skills)}")
        if job.preferred_skills:
            parts.append(f"Preferred Skills: {', '.join(job.preferred_skills)}")
        if job.experience_level:
            parts.append(f"Experience Level: {job.experience_level}")
        if job.location:
            parts.append(f"Location: {job.location}")
        if job.employment_type:
            parts.append(f"Employment Type: {job.employment_type}")
        
        return " | ".join(parts)
    
    def layer_a_filter(self, job: JobPost, candidates: List[Candidate]) -> List[Tuple[Candidate, float]]:
        """Layer A: Fast embedding-based similarity filtering"""
        if not candidates:
            return []
        
        # Create text representations
        job_text = self._create_job_text(job)
        candidate_texts = [self._create_candidate_text(candidate) for candidate in candidates]
        
        # Generate embeddings
        job_embedding = self.embedding_model.encode([job_text])
        candidate_embeddings = self.embedding_model.encode(candidate_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(job_embedding, candidate_embeddings)[0]
        
        # Filter candidates above threshold
        filtered_candidates = []
        for candidate, similarity in zip(candidates, similarities):
            if similarity >= self.embedding_threshold:
                filtered_candidates.append((candidate, float(similarity)))
        
        # Sort by similarity score (highest first)
        filtered_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_candidates
    
    async def layer_b_filter_single(self, job: JobPost, candidate: Candidate, embedding_score: float) -> Dict[str, Any]:
        """Layer B: LLM-based reasoning for a single candidate"""
        try:
            job_text = self._create_job_text(job)
            candidate_text = self._create_candidate_text(candidate)
            
            # Use LLM to analyze fit
            analysis = self.vertex_client.analyze_candidate_fit(job_text, candidate_text)
            
            # Combine embedding score with LLM analysis
            combined_score = (embedding_score * 0.3) + (analysis.get('overall_fit_score', 0) * 0.7)
            
            return {
                'candidate': candidate,
                'embedding_score': embedding_score,
                'llm_analysis': analysis,
                'combined_score': combined_score,
                'passed_filter': combined_score >= self.llm_threshold
            }
            
        except Exception as e:
            print(f"Error in Layer B filtering for {candidate.name}: {e}")
            return {
                'candidate': candidate,
                'embedding_score': embedding_score,
                'llm_analysis': {
                    'overall_fit_score': 0.0,
                    'reasoning': f"Analysis failed: {str(e)}",
                    'recommendation': 'NO_MATCH'
                },
                'combined_score': embedding_score * 0.3,  # Fallback to embedding only
                'passed_filter': False
            }
    
    async def layer_b_filter(self, job: JobPost, layer_a_results: List[Tuple[Candidate, float]], max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Layer B: LLM-based reasoning for multiple candidates with concurrency control"""
        if not layer_a_results:
            return []
        
        # Create semaphore to limit concurrent LLM calls
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(candidate, score):
            async with semaphore:
                return await self.layer_b_filter_single(job, candidate, score)
        
        # Process all candidates concurrently (with limit)
        tasks = [process_with_semaphore(candidate, score) for candidate, score in layer_a_results]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and sort by combined score
        valid_results = [r for r in results if isinstance(r, dict)]
        valid_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return valid_results
    
    def filter_candidates_sync(self, job: JobPost, candidates: List[Candidate], max_layer_b: int = 20) -> Dict[str, Any]:
        """Synchronous wrapper for the complete filtering process"""
        # Layer A: Fast embedding filtering
        layer_a_results = self.layer_a_filter(job, candidates)
        
        print(f"Layer A: {len(layer_a_results)}/{len(candidates)} candidates passed embedding filter")
        
        if not layer_a_results:
            return {
                'layer_a_count': 0,
                'layer_b_count': 0,
                'filtered_candidates': [],
                'all_results': []
            }
        
        # Limit Layer B processing to top candidates
        layer_a_limited = layer_a_results[:max_layer_b]
        
        # Layer B: LLM-based reasoning
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            layer_b_results = loop.run_until_complete(
                self.layer_b_filter(job, layer_a_limited)
            )
        finally:
            loop.close()
        
        # Filter candidates that passed Layer B
        passed_candidates = [r for r in layer_b_results if r['passed_filter']]
        
        print(f"Layer B: {len(passed_candidates)}/{len(layer_a_limited)} candidates passed LLM filter")
        
        return {
            'layer_a_count': len(layer_a_results),
            'layer_b_count': len(passed_candidates),
            'filtered_candidates': passed_candidates,
            'all_results': layer_b_results
        }
    
    async def filter_candidates_async(self, job: JobPost, candidates: List[Candidate], max_layer_b: int = 20) -> Dict[str, Any]:
        """Asynchronous version of the complete filtering process"""
        # Layer A: Fast embedding filtering
        layer_a_results = self.layer_a_filter(job, candidates)
        
        print(f"Layer A: {len(layer_a_results)}/{len(candidates)} candidates passed embedding filter")
        
        if not layer_a_results:
            return {
                'layer_a_count': 0,
                'layer_b_count': 0,
                'filtered_candidates': [],
                'all_results': []
            }
        
        # Limit Layer B processing to top candidates
        layer_a_limited = layer_a_results[:max_layer_b]
        
        # Layer B: LLM-based reasoning
        layer_b_results = await self.layer_b_filter(job, layer_a_limited)
        
        # Filter candidates that passed Layer B
        passed_candidates = [r for r in layer_b_results if r['passed_filter']]
        
        print(f"Layer B: {len(passed_candidates)}/{len(layer_a_limited)} candidates passed LLM filter")
        
        return {
            'layer_a_count': len(layer_a_results),
            'layer_b_count': len(passed_candidates),
            'filtered_candidates': passed_candidates,
            'all_results': layer_b_results
        }


# Global filter instance
_smart_filter = None

def get_smart_filter() -> SmartFilter:
    """Get or create the global smart filter"""
    global _smart_filter
    if _smart_filter is None:
        _smart_filter = SmartFilter()
    return _smart_filter 