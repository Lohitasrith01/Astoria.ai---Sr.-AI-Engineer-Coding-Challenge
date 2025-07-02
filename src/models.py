from pydantic import BaseModel, Field, validator
from typing import List, Optional


class JobPost(BaseModel):
    title: str
    description: str
    min_years_experience: int = Field(ge=0, description="Minimum years of relevant experience required")
    visa_requirements: List[str] = Field(default_factory=list, description="Allowed visa / work authorization statuses")
    num_candidates: int = Field(default=5, ge=1, description="How many candidates to shortlist")


class Candidate(BaseModel):
    name: str
    years_experience: int = Field(ge=0)
    visa_status: str
    resume_text: str
    github_handle: Optional[str] = None
    linkedin_url: Optional[str] = None

    @validator("resume_text")
    def resume_not_empty(cls, v: str):
        if not v or len(v.strip()) == 0:
            raise ValueError("resume_text cannot be empty")
        return v 