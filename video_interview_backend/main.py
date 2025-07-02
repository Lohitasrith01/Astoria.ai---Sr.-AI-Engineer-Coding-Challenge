from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uuid
import cv2
import numpy as np
import mediapipe as mp
import opensmile
from google.cloud import speech
import os
import base64
import tempfile
import json
import websockets
from agents.interview_agent import get_interview_agent
from src.models import JobPost, Candidate
from src.scoring import overall_score
import redis
import boto3
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from pydantic import BaseModel
from src.smart_filter import SmartFilter
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict

app = FastAPI()

# Allow CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MediaPipe analysis implementation
# Requires: pip install mediapipe opencv-python numpy
# This function expects frame_bytes as raw image bytes (e.g., JPEG/PNG)
def analyze_frame_with_mediapipe(frame_bytes):
    nparr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    mp_pose = mp.solutions.pose
    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    results = {}
    with mp_pose.Pose(static_image_mode=True) as pose:
        pose_result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        results['pose'] = bool(pose_result.pose_landmarks)
    with mp_face.FaceMesh(static_image_mode=True) as face:
        face_result = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        results['face'] = bool(face_result.multi_face_landmarks)
    with mp_hands.Hands(static_image_mode=True) as hands:
        hands_result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        results['hands'] = bool(hands_result.multi_hand_landmarks)
    return results

# OpenSmile audio analysis implementation
# Requires: pip install opensmile
# This function expects audio_bytes as WAV bytes
# You must have OpenSmile installed and accessible
def analyze_audio_with_opensmile(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = smile.process_file(tmp_path)
    os.remove(tmp_path)
    return features.to_dict() if hasattr(features, 'to_dict') else {}

# Google STT implementation
# Requires: pip install google-cloud-speech
# Set GOOGLE_APPLICATION_CREDENTIALS env var to your service account JSON
# This function expects audio_bytes as WAV bytes (16kHz, LINEAR16)
def transcribe_audio_with_google(audio_bytes):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )
    response = client.recognize(config=config, audio=audio)
    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript + " "
    return transcript.strip()

# Deepgram TTS implementation
# Requires: pip install websockets
# Set DEEPGRAM_API_KEY env var to your Deepgram API key
# This function sends text to Deepgram TTS WebSocket and returns audio bytes (linear16, 24kHz)
def synthesize_speech_with_deepgram(text):
    async def _synthesize():
        api_key = os.getenv("DEEPGRAM_API_KEY")
        url = "wss://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate=24000"
        headers = {"Authorization": f"Token {api_key}"}
        audio_chunks = []
        async with websockets.connect(url, extra_headers=headers) as ws:
            await ws.send(json.dumps({"type": "Speak", "text": text}))
            await ws.send(json.dumps({"type": "Flush"}))
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    audio_chunks.append(msg)
                elif isinstance(msg, str) and '"type":"Close"' in msg:
                    break
        return b"".join(audio_chunks)
    return asyncio.run(_synthesize())

# Initialize InterviewAgent (singleton)
interview_agent = get_interview_agent()

# Redis setup (assume running locally or set REDIS_URL env)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL)

# S3 setup (assume AWS credentials in env)
s3 = boto3.client('s3')
S3_BUCKET = os.getenv("S3_BUCKET", "video-interview-videos")

# Per-session context management
def get_session_context(session_id):
    key = f"interview:context:{session_id}"
    data = redis_client.get(key)
    if data:
        # Deserialize context (could use pickle, json, or custom)
        import pickle
        return pickle.loads(data)
    else:
        # New session: create context
        job = JobPost(title="AI Engineer", description="Python, NLP, ML, LLM, cloud", min_years_experience=3, visa_requirements=["US Citizen"], num_candidates=1)
        candidate = Candidate(name="Alice Smith", years_experience=5, visa_status="US Citizen", resume_text="Experienced software engineer specialized in NLP, Python, and large language models. Led projects using GPT-3.5, vector databases, and MLOps on AWS.", github_handle="torvalds", linkedin_url="https://linkedin.com/in/alicesmith")
        context = interview_agent.start_interview(job, candidate)
        # Attach both buffer and summary memory
        context.conversation_memory = ConversationBufferWindowMemory(k=20, return_messages=True)
        context.summary_memory = ConversationSummaryBufferMemory(llm=None, memory_key="summary")  # LLM to be set if needed
        return context

def save_session_context(session_id, context):
    key = f"interview:context:{session_id}"
    import pickle
    redis_client.set(key, pickle.dumps(context))

# S3 video upload utility
def upload_video_frame_to_s3(session_id, frame_bytes, frame_idx):
    key = f"videos/{session_id}/frame_{frame_idx:06d}.jpg"
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=frame_bytes, ContentType="image/jpeg")
    return key

def delete_video_from_s3(session_id):
    # List and delete all frames for this session
    prefix = f"videos/{session_id}/"
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    if 'Contents' in response:
        for obj in response['Contents']:
            s3.delete_object(Bucket=S3_BUCKET, Key=obj['Key'])

# --- In-memory session management for WebRTC signaling ---
webrtc_sessions: Dict[str, Dict[str, WebSocket]] = {}  # session_id -> {peer_id: websocket}

@app.websocket("/ws/interview")
async def interview_ws(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "signaling":
                await websocket.send_json({"type": "signaling_ack", "data": data.get("data")})

            elif msg_type == "media":
                frame_bytes = base64.b64decode(data.get("video_frame")) if data.get("video_frame") else None
                audio_bytes = base64.b64decode(data.get("audio_frame")) if data.get("audio_frame") else None

                mediapipe_result = analyze_frame_with_mediapipe(frame_bytes) if frame_bytes else {}
                opensmile_result = analyze_audio_with_opensmile(audio_bytes) if audio_bytes else {}
                transcript = transcribe_audio_with_google(audio_bytes) if audio_bytes else ""

                # Use InterviewAgent to get next question and update context
                if transcript:
                    # Add candidate message to context
                    # The InterviewAgent expects InterviewContext and uses .messages as InterviewMessage objects
                    # For simplicity, we just call generate_response
                    next_question = await interview_agent.generate_response(get_session_context(session_id), transcript)
                else:
                    next_question = "Could you please repeat that?"

                # Score candidate using overall_score
                score = overall_score(get_session_context(session_id).job_post, get_session_context(session_id).candidate)

                tts_audio = synthesize_speech_with_deepgram(next_question)

                await websocket.send_json({
                    "type": "analysis",
                    "mediapipe": mediapipe_result,
                    "opensmile": opensmile_result,
                    "score": score,
                    "transcript": transcript,
                    "next_question": next_question
                })
                await websocket.send_bytes(tts_audio)

                # Store video frames to S3 as they arrive
                if frame_bytes:
                    upload_video_frame_to_s3(session_id, frame_bytes, data.get("frame_idx"))

            else:
                await websocket.send_json({"type": "error", "message": "Unknown message type."})

    except WebSocketDisconnect:
        print(f"Session {session_id} disconnected.")
        delete_video_from_s3(session_id)
        save_session_context(session_id, None)
    except Exception as e:
        print(f"Error in session {session_id}: {e}")
        delete_video_from_s3(session_id)
        save_session_context(session_id, None)
        await websocket.close()

# Placeholder REST endpoint for WebRTC signaling (if needed)
@app.post("/signaling")
async def signaling_endpoint(payload: dict):
    # TODO: Implement signaling logic for WebRTC
    return {"status": "ok", "payload": payload}

# Health check
@app.get("/")
def root():
    return {"status": "ok", "message": "Video Interview Backend Running"}

# --- Recruiter API Models ---
class ScheduleCallRequest(BaseModel):
    availability: str  # e.g., "2024-06-10T15:00:00Z"
    notes: str = ""

class ScheduleVideoInterviewRequest(BaseModel):
    job: OrchestrateJob
    candidate: OrchestrateCandidate
    initial_analysis: dict
    recruiter_questions: list = []
    recruiter_email: str = None

# --- Recruiter Endpoints ---

@app.get("/candidate/{session_id}/report")
def get_candidate_report(session_id: str):
    """Return the candidate's analysis/report for the recruiter UI."""
    context = get_session_context(session_id)
    if not context:
        raise HTTPException(status_code=404, detail="Session not found")
    summary = interview_agent.generate_interview_summary(context)
    return summary

@app.post("/candidate/{session_id}/reject")
def reject_candidate(session_id: str):
    """Mark candidate as rejected and delete their video from S3."""
    # Mark as rejected in Redis
    redis_client.set(f"interview:rejected:{session_id}", True)
    # Delete video from S3
    delete_video_from_s3(session_id)
    return {"status": "rejected", "session_id": session_id}

@app.post("/candidate/{session_id}/schedule_call")
def schedule_call(session_id: str, req: ScheduleCallRequest):
    """Store candidate's availability for recruiter follow-up."""
    # Store scheduling info in Redis
    redis_client.set(f"interview:schedule:{session_id}", req.json())
    return {"status": "scheduled", "session_id": session_id, "availability": req.availability, "notes": req.notes}

@app.post("/candidate/{candidate_name}/schedule_video_interview")
def schedule_video_interview(candidate_name: str, req: ScheduleVideoInterviewRequest):
    """Schedule a video interview, store all context, and return session link."""
    session_id = str(uuid.uuid4())
    # Store all context in Redis
    context = {
        "job": req.job.dict(),
        "candidate": req.candidate.dict(),
        "initial_analysis": req.initial_analysis,
        "recruiter_questions": req.recruiter_questions,
        "status": "scheduled",
        "session_id": session_id,
        "recruiter_email": req.recruiter_email
    }
    redis_client.set(f"video_interview:session:{session_id}", json.dumps(context))
    # Return session link (frontend should use this to connect candidate)
    session_link = f"/ws/interview?session_id={session_id}"
    return {"session_id": session_id, "session_link": session_link}

# --- On interview completion, send recruiter email (stub) ---
def notify_recruiter_on_completion(session_id):
    context_json = redis_client.get(f"video_interview:session:{session_id}")
    if not context_json:
        return
    context = json.loads(context_json)
    recruiter_email = context.get("recruiter_email")
    report = context.get("final_report")
    # TODO: Integrate with email system to send report to recruiter
    print(f"[NOTIFY] Email sent to {recruiter_email} with report: {report}")

# --- Orchestration Endpoints ---
from typing import List

class OrchestrateCandidate(BaseModel):
    name: str
    years_experience: int
    visa_status: str
    resume_text: str
    github_handle: str = None
    linkedin_url: str = None
    # Add other fields as needed

class OrchestrateJob(BaseModel):
    title: str
    description: str
    min_years_experience: int
    visa_requirements: List[str]
    num_candidates: int
    # Add other fields as needed

class OrchestrateRequest(BaseModel):
    job: OrchestrateJob
    candidates: List[OrchestrateCandidate]
    job_id: str

@app.post("/orchestrate/bulk_process")
def orchestrate_bulk_process(req: OrchestrateRequest):
    """Run ATS (SentenceTransformers), then scoring and video agents in parallel for the filtered pool."""
    job = JobPost(**req.job.dict())
    candidates = [Candidate(**c.dict()) for c in req.candidates]
    job_id = req.job_id
    smart_filter = SmartFilter()

    # 1. ATS Filtering (Layer A)
    layer_a_results = smart_filter.layer_a_filter(job, candidates)
    filtered_candidates = [c for c, score in layer_a_results]
    redis_client.set(f"orchestrate:{job_id}:ats_passed", json.dumps([c.dict() for c in filtered_candidates]))

    # 2. Scoring and Video Interview in parallel
    results = []
    def process_candidate(candidate):
        # Scoring
        score = overall_score(job, candidate)
        # Video interview context (simulate or trigger actual process)
        context = interview_agent.start_interview(job, candidate)
        # For demo, just generate summary (replace with real video agent logic)
        summary = interview_agent.generate_interview_summary(context)
        # Store per-candidate result in Redis
        redis_client.set(f"orchestrate:{job_id}:result:{candidate.name}", json.dumps({
            "score": score,
            "summary": summary
        }))
        return {"name": candidate.name, "score": score, "summary": summary}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_candidate, c) for c in filtered_candidates]
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"error": str(e)})

    # Store all results for job
    redis_client.set(f"orchestrate:{job_id}:all_results", json.dumps(results))
    return {"status": "completed", "job_id": job_id, "num_candidates": len(filtered_candidates)}

@app.get("/orchestrate/status/{job_id}")
def orchestrate_status(job_id: str):
    """Return orchestration status/results for recruiter UI polling."""
    ats_passed = redis_client.get(f"orchestrate:{job_id}:ats_passed")
    all_results = redis_client.get(f"orchestrate:{job_id}:all_results")
    return {
        "ats_passed": json.loads(ats_passed) if ats_passed else [],
        "all_results": json.loads(all_results) if all_results else []
    }

CLEANUP_THRESHOLD_DAYS = int(os.getenv("CLEANUP_THRESHOLD_DAYS", 7))

@app.post("/admin/cleanup_old_sessions")
def cleanup_old_sessions():
    """Delete Redis keys and S3 videos for sessions older than threshold days."""
    now = datetime.utcnow()
    threshold = now - timedelta(days=CLEANUP_THRESHOLD_DAYS)
    deleted_sessions = []
    # Scan Redis for session keys
    for key in redis_client.scan_iter("video_interview:session:*"):
        context_json = redis_client.get(key)
        if not context_json:
            continue
        context = json.loads(context_json)
        # Assume context has a 'created_at' or fallback to Redis TTL/creation
        created_at = context.get("created_at")
        if not created_at:
            # Fallback: skip if no timestamp
            continue
        created_dt = datetime.utcfromtimestamp(created_at)
        if created_dt < threshold:
            session_id = context.get("session_id")
            if session_id:
                delete_video_from_s3(session_id)
            redis_client.delete(key)
            deleted_sessions.append(session_id)
    return {"deleted_sessions": deleted_sessions, "threshold_days": CLEANUP_THRESHOLD_DAYS}

@app.websocket("/ws/signaling/{session_id}/{peer_id}")
async def webrtc_signaling_ws(websocket: WebSocket, session_id: str, peer_id: str):
    await websocket.accept()
    if session_id not in webrtc_sessions:
        webrtc_sessions[session_id] = {}
    webrtc_sessions[session_id][peer_id] = websocket
    try:
        while True:
            data = await websocket.receive_json()
            # Relay signaling message to the other peer in the session
            for other_peer_id, other_ws in webrtc_sessions[session_id].items():
                if other_peer_id != peer_id:
                    await other_ws.send_json({"from": peer_id, **data})
    except WebSocketDisconnect:
        # Remove peer from session on disconnect
        if session_id in webrtc_sessions and peer_id in webrtc_sessions[session_id]:
            del webrtc_sessions[session_id][peer_id]
        # Clean up session if empty
        if session_id in webrtc_sessions and not webrtc_sessions[session_id]:
            del webrtc_sessions[session_id] 