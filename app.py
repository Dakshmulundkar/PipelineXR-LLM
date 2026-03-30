"""
PipelineXR LLM Service — Hugging Face Space
Uses Qwen2.5-Coder-7B-Instruct (4-bit GGUF) via llama-cpp-python.
Model is downloaded at runtime to ephemeral disk — NOT stored in the repo.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipelinexr-llm")

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID    = os.getenv("MODEL_REPO",  "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF")
GGUF_FILE  = os.getenv("MODEL_FILE",  "qwen2.5-coder-7b-instruct-q4_k_m.gguf")
MAX_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))
CTX_SIZE   = int(os.getenv("CTX_SIZE", "4096"))
API_SECRET = os.getenv("API_SECRET", "")

llm = None

def load_model():
    global llm
    log.info(f"Downloading {GGUF_FILE} from {REPO_ID} ...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=GGUF_FILE)
    log.info(f"Model cached at {model_path}. Loading into llama-cpp ...")
    llm = Llama(
        model_path=model_path,
        n_ctx=CTX_SIZE,
        n_threads=os.cpu_count(),
        verbose=False,
    )
    log.info("Model ready.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="PipelineXR LLM", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────
def check_auth(request: Request):
    if not API_SECRET:
        return
    if request.headers.get("Authorization") != f"Bearer {API_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")

# ── Models ────────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    system: str = "You are a helpful DevOps and security assistant."
    max_tokens: int = MAX_TOKENS

class SecurityReviewRequest(BaseModel):
    repository: str
    vulnerabilities: list[dict]
    scan_engine: str = "trivy"

class PipelineEmailRequest(BaseModel):
    repository: str
    workflow_name: str
    conclusion: str
    duration_seconds: int = 0
    head_branch: str = "main"
    triggering_actor: str = ""
    failed_steps: list[str] = []
    commit_message: str = ""
    run_url: str = ""

class MonitorEmailRequest(BaseModel):
    url: str
    is_up: bool
    response_time_ms: int = 0
    consecutive_failures: int = 0
    incident_started_at: str = ""
    error: str = ""

class DoraInsightsRequest(BaseModel):
    repository: str
    time_range: str = "7d"
    avg_build_duration: float = 0
    success_rate: float = 0
    total_deployments: int = 0
    failed_deployments: int = 0

class IncidentRequest(BaseModel):
    title: str
    severity: str = "high"
    affected_service: str = ""
    symptoms: list[str] = []
    recent_changes: list[str] = []
    error_logs: list[str] = []

# ── Core inference ────────────────────────────────────────────────────────────
def infer(system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    out = llm(prompt, max_tokens=max_tokens, stop=["<|im_end|>"], echo=False)
    return out["choices"][0]["text"].strip()

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": GGUF_FILE, "device": "cpu"}

@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    check_auth(request)
    try:
        result = infer(req.system, req.prompt, req.max_tokens)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/security-review")
async def security_review(req: SecurityReviewRequest, request: Request):
    check_auth(request)
    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    top = sorted(req.vulnerabilities, key=lambda v: sev_order.get(v.get("severity","low").lower(), 4))[:10]
    vuln_str = "\n".join(
        f"- [{v.get('severity','?').upper()}] {v.get('id','N/A')} in {v.get('package_name','?')}"
        f" fix: {v.get('fixed_version','none')}"
        for v in top
    )
    system = "You are a senior application security engineer. Return only valid JSON, no markdown."
    user = f"Repository: {req.repository}\nFindings:\n{vuln_str}\n\nReturn JSON: {{\"risk_summary\":\"...\",\"critical_actions\":[],\"overall_posture\":\"secure|at-risk|critical\"}}"
    result = infer(system, user, 600)
    return {"ok": True, "data": result}

@app.post("/pipeline-email")
async def pipeline_email(req: PipelineEmailRequest, request: Request):
    check_auth(request)
    steps = "\n".join(f"  - {s}" for s in req.failed_steps) or "  (none)"
    system = "You are a DevOps engineer writing incident notification emails. Return only valid JSON."
    user = f"Repo: {req.repository}, Workflow: {req.workflow_name}, Branch: {req.head_branch}, Result: {req.conclusion}\nFailed steps:\n{steps}\nRun: {req.run_url}\n\nReturn JSON: {{\"subject\":\"...\",\"body_text\":\"...\",\"urgency\":\"low|medium|high|critical\"}}"
    result = infer(system, user, 700)
    return {"ok": True, "data": result}

@app.post("/monitor-email")
async def monitor_email(req: MonitorEmailRequest, request: Request):
    check_auth(request)
    status = "DOWN" if not req.is_up else "RECOVERED"
    system = "You are an SRE writing uptime alert emails. Return only valid JSON."
    user = f"URL: {req.url}, Status: {status}, Failures: {req.consecutive_failures}, Error: {req.error}\n\nReturn JSON: {{\"subject\":\"...\",\"body_text\":\"...\",\"severity\":\"warning|critical\"}}"
    result = infer(system, user, 500)
    return {"ok": True, "data": result}

@app.post("/dora-insights")
async def dora_insights(req: DoraInsightsRequest, request: Request):
    check_auth(request)
    system = "You are a DevOps performance analyst. Return only valid JSON."
    user = f"Repo: {req.repository}, Range: {req.time_range}, Success rate: {req.success_rate}%, Deployments: {req.total_deployments}, Failed: {req.failed_deployments}\n\nReturn JSON: {{\"executive_summary\":\"...\",\"performance_grade\":\"Elite|High|Medium|Low\",\"recommendations\":[]}}"
    result = infer(system, user, 600)
    return {"ok": True, "data": result}

@app.post("/incident-response")
async def incident_response(req: IncidentRequest, request: Request):
    check_auth(request)
    symptoms = "\n".join(f"  - {s}" for s in req.symptoms) or "  (none)"
    system = "You are an experienced SRE providing incident response guidance. Return only valid JSON."
    user = f"Incident: {req.title}, Severity: {req.severity}, Service: {req.affected_service}\nSymptoms:\n{symptoms}\n\nReturn JSON: {{\"immediate_actions\":[],\"likely_root_causes\":[],\"estimated_resolution_time\":\"...\"}}"
    result = infer(system, user, 600)
    return {"ok": True, "data": result}
