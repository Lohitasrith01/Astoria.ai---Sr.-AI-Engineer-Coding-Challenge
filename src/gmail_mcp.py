from typing import Optional, Dict

class GmailMCP:
    """
    Placeholder for a Gmail Machine Code Processing (MCP) or API integration.
    In production, this would connect to Gmail's API or SMTP for real email sending.
    All methods here are stubs for demonstration/testing purposes only.
    """
    def __init__(self):
        pass

    def send_interview_invite(
        self,
        candidate_email: str,
        candidate_name: str,
        interview_link: str,
        initial_analysis: Optional[Dict] = None,
        recruiter_name: Optional[str] = None,
        job_title: Optional[str] = None,
        interview_time: Optional[str] = None,
    ) -> bool:
        """
        Send an interview invitation email to the candidate.
        This is a stub. Replace with real Gmail API/SMTP logic in production.
        """
        print("\n--- SENDING EMAIL (STUB) ---")
        print(f"To: {candidate_email}")
        print(f"Subject: Interview Invitation for {job_title or 'the position'}")
        print(f"Hi {candidate_name},\n")
        print(f"You are invited to a video interview for the role of {job_title or 'the position'}.")
        if interview_time:
            print(f"Scheduled Time: {interview_time}")
        print(f"Interview Link: {interview_link}\n")
        if initial_analysis:
            print("Initial Analysis Summary:")
            for k, v in initial_analysis.items():
                print(f"- {k}: {v}")
        print("\nBest regards,")
        print(recruiter_name or "Recruiter Team")
        print("--- END EMAIL (STUB) ---\n")
        return True

_gmail_mcp = None

def get_gmail_mcp() -> GmailMCP:
    """
    Get or create the global Gmail MCP instance.
    """
    global _gmail_mcp
    if _gmail_mcp is None:
        _gmail_mcp = GmailMCP()
    return _gmail_mcp 