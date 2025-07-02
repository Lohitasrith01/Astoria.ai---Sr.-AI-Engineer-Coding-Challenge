from typing import Optional, Dict
import redis
import os
import json
import time

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

# Redis setup (reuse REDIS_URL from env or default)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL)

# --- Email/Calendar Deduplication and Logging ---
def email_already_sent(session_id, email_type, recipient):
    key = f"email_sent:{session_id}:{email_type}:{recipient}"
    return redis_client.exists(key)

def mark_email_sent(session_id, email_type, recipient):
    key = f"email_sent:{session_id}:{email_type}:{recipient}"
    redis_client.set(key, int(time.time()))

def log_email_action(session_id, email_type, recipient, status, extra=None):
    key = f"email_log:{session_id}"
    log_entry = {
        "timestamp": int(time.time()),
        "email_type": email_type,
        "recipient": recipient,
        "status": status,
        "extra": extra or {}
    }
    redis_client.rpush(key, json.dumps(log_entry))

# --- Example: Send Interview Invite (with deduplication) ---
def send_interview_invite(session_id, candidate_email, candidate_name, interview_link, initial_analysis, recruiter_name, job_title):
    email_type = "interview_invite"
    if email_already_sent(session_id, email_type, candidate_email):
        log_email_action(session_id, email_type, candidate_email, "skipped_duplicate")
        return False
    # ... existing email sending logic ...
    print(f"[EMAIL] Sent interview invite to {candidate_email} for session {session_id}")
    mark_email_sent(session_id, email_type, candidate_email)
    log_email_action(session_id, email_type, candidate_email, "sent", {"job_title": job_title, "recruiter": recruiter_name})
    return True

# --- Example: Send Recruiter Report (with deduplication) ---
def send_recruiter_report(session_id, recruiter_email, report):
    email_type = "recruiter_report"
    if email_already_sent(session_id, email_type, recruiter_email):
        log_email_action(session_id, email_type, recruiter_email, "skipped_duplicate")
        return False
    # ... existing email sending logic ...
    print(f"[EMAIL] Sent recruiter report to {recruiter_email} for session {session_id}")
    mark_email_sent(session_id, email_type, recruiter_email)
    log_email_action(session_id, email_type, recruiter_email, "sent", {"report_summary": report.get('summary', '')})
    return True

# --- Calendly MCP Integration (Stub) ---
class CalendlyMCP:
    """Calendly MCP for managing candidate and recruiter scheduling after video interviews."""
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("CALENDLY_API_KEY")
        # TODO: Initialize Calendly API client if available

    def create_event(self, session_id, candidate_email, recruiter_email, start_time, end_time, notes=None):
        """Create a Calendly event for a follow-up call."""
        event_type = "followup_call"
        if email_already_sent(session_id, event_type, candidate_email):
            log_email_action(session_id, event_type, candidate_email, "skipped_duplicate")
            return False
        # TODO: Integrate with Calendly API to create event and send invites
        print(f"[CALENDLY] Created event for {candidate_email} and {recruiter_email} from {start_time} to {end_time} (session {session_id})")
        mark_email_sent(session_id, event_type, candidate_email)
        log_email_action(session_id, event_type, candidate_email, "created", {"recruiter_email": recruiter_email, "start_time": start_time, "end_time": end_time, "notes": notes})
        return True

    def check_availability(self, recruiter_email, candidate_email=None):
        """Check availability for recruiter (and optionally candidate)."""
        # TODO: Integrate with Calendly API to fetch availability
        print(f"[CALENDLY] Checked availability for recruiter {recruiter_email} (candidate: {candidate_email})")
        return {"available_slots": []}  # Stub

# Singleton getter
_calendly_mcp = None

def get_calendly_mcp():
    global _calendly_mcp
    if _calendly_mcp is None:
        _calendly_mcp = CalendlyMCP()
    return _calendly_mcp 