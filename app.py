"""
PipelineXR LLM Service — Hugging Face Space
Qwen2.5-Coder-7B-Instruct (4-bit GGUF) via llama-cpp-python.

Optimizations applied:
  - JSON priming: assistant turn starts with '{' — forces JSON from token 1
  - extract_json(): robust extraction even if model adds prose around JSON
  - Reduced max_tokens per endpoint — less generation = faster response
  - n_batch tuned for CPU throughput
  - temperature=0 for deterministic, structured output (no creative drift)
  - repeat_penalty to prevent the model looping on the same phrase
  - top_p / top_k disabled at temp=0 (greedy decoding is fastest + most consistent)
  - Concurrency lock — prevents two requests from running inference simultaneously
    (llama-cpp is not thread-safe; concurrent calls corrupt output)
"""

import os
import re
import json
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipelinexr-llm")

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID    = os.getenv("MODEL_REPO", os.getenv("MODEL_ID", "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"))
GGUF_FILE  = os.getenv("MODEL_FILE",  "qwen2.5-coder-7b-instruct-q4_k_m.gguf")
MAX_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
CTX_SIZE   = int(os.getenv("CTX_SIZE", "4096"))
API_SECRET = os.getenv("API_SECRET", "")

llm = None
# Inference lock — llama-cpp is not thread-safe
_infer_lock = asyncio.Lock()

def load_model():
    global llm
    log.info(f"Downloading {GGUF_FILE} from {REPO_ID} ...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=GGUF_FILE)
    log.info(f"Model cached at {model_path}. Loading into llama-cpp ...")
    llm = Llama(
        model_path=model_path,
        n_ctx=CTX_SIZE,
        n_threads=os.cpu_count(),   # use all CPU cores
        n_batch=512,                # larger batch = better CPU throughput
        verbose=False,
        use_mmap=True,              # memory-map the model file — faster load, less RAM copy
        use_mlock=False,            # don't lock pages — HF Space has limited RAM
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

# ── Pydantic models ───────────────────────────────────────────────────────────
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
async def infer(system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
    """
    Async inference with concurrency lock.

    Key techniques:
    - Primes assistant turn with '{' → model's first token is inside JSON
    - temperature=0 → greedy decoding, fastest + most deterministic
    - repeat_penalty=1.1 → prevents repetition loops
    - top_k=1 at temp=0 is equivalent to greedy (no sampling overhead)
    """
    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{{"
    )

    async with _infer_lock:
        # Run blocking llama-cpp call in thread pool so FastAPI stays responsive
        loop = asyncio.get_event_loop()
        out = await loop.run_in_executor(
            None,
            lambda: llm(
                prompt,
                max_tokens=max_tokens,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
                temperature=0.0,        # greedy — deterministic, no sampling overhead
                repeat_penalty=1.1,     # prevent repetition loops
                top_k=1,               # greedy equivalent
                top_p=1.0,
            )
        )

    raw = out["choices"][0]["text"].strip()
    return "{" + raw  # re-attach the priming brace


def extract_json(text: str) -> str:
    """
    Robustly extract the first valid JSON object from model output.

    Handles:
    - Leading prose before the JSON
    - Markdown code fences (```json ... ```)
    - Trailing text after the closing brace
    - Slightly malformed JSON (trailing commas)

    Returns the raw JSON string. If extraction fails, returns the original text
    so the client can still display something.
    """
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text.strip())

    # Find first '{' and walk to matching '}'
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                # Try to parse as-is
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    # Try fixing trailing commas (common model mistake)
                    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        json.loads(fixed)
                        return fixed
                    except json.JSONDecodeError:
                        pass
                break

    return text  # fallback: return whatever we got


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": GGUF_FILE, "device": "cpu"}


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    check_auth(request)
    try:
        result = await infer(req.system, req.prompt, req.max_tokens)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/security-review")
async def security_review(req: SecurityReviewRequest, request: Request):
    check_auth(request)

    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    top = sorted(
        req.vulnerabilities,
        key=lambda v: sev_order.get(v.get("severity", "low").lower(), 4)
    )[:8]

    vuln_lines = "\n".join(
        f"- [{v.get('severity', '?').upper()}] {v.get('id', 'N/A')} "
        f"in {v.get('package_name', '?')} "
        f"(fix: {v.get('fixed_version') or 'none'})"
        for v in top
    ) or "No vulnerabilities found."

    system = (
        "You are a senior application security engineer. "
        "Output ONLY a single valid JSON object. "
        "No text before or after the JSON. No markdown."
    )
    user = (
        f"Security findings for '{req.repository}':\n{vuln_lines}\n\n"
        f"Fill this JSON with real analysis:\n"
        f'{{"overall_posture":"critical|at-risk|secure",'
        f'"risk_summary":"2-3 sentences about the security posture",'
        f'"critical_actions":["specific action 1","specific action 2"],'
        f'"recommendations":["recommendation 1","recommendation 2"]}}'
    )

    raw = await infer(system, user, 400)
    return {"ok": True, "data": extract_json(raw)}


@app.post("/pipeline-email")
async def pipeline_email(req: PipelineEmailRequest, request: Request):
    check_auth(request)

    steps = "\n".join(f"- {s}" for s in req.failed_steps) or "- (none listed)"
    system = (
        "You are a DevOps engineer writing incident emails. "
        "Output ONLY a single valid JSON object. No markdown."
    )
    user = (
        f"Pipeline failure:\n"
        f"Repo: {req.repository} | Workflow: {req.workflow_name} | "
        f"Branch: {req.head_branch} | Result: {req.conclusion}\n"
        f"Failed steps:\n{steps}\n"
        f"Run: {req.run_url}\n\n"
        f"Fill this JSON:\n"
        f'{{"subject":"[FAILED] subject line","body_text":"email body",'
        f'"urgency":"low|medium|high|critical"}}'
    )

    raw = await infer(system, user, 500)
    return {"ok": True, "data": extract_json(raw)}


@app.post("/monitor-email")
async def monitor_email(req: MonitorEmailRequest, request: Request):
    check_auth(request)

    status = "DOWN" if not req.is_up else "RECOVERED"
    system = (
        "You are an SRE writing uptime alerts. "
        "Output ONLY a single valid JSON object. No markdown."
    )
    user = (
        f"URL: {req.url} is {status}. "
        f"Failures: {req.consecutive_failures}. "
        f"Error: {req.error or 'none'}.\n\n"
        f"Fill this JSON:\n"
        f'{{"subject":"alert subject","body_text":"alert body",'
        f'"severity":"warning|critical"}}'
    )

    raw = await infer(system, user, 350)
    return {"ok": True, "data": extract_json(raw)}


@app.post("/dora-insights")
async def dora_insights(req: DoraInsightsRequest, request: Request):
    check_auth(request)

    system = (
        "You are a DevOps performance analyst. "
        "Output ONLY a single valid JSON object. No markdown."
    )
    user = (
        f"DORA metrics for '{req.repository}' ({req.time_range}):\n"
        f"Success rate: {req.success_rate}% | "
        f"Deployments: {req.total_deployments} | "
        f"Failed: {req.failed_deployments} | "
        f"Avg build: {req.avg_build_duration}min\n\n"
        f"Fill this JSON:\n"
        f'{{"executive_summary":"2-3 sentence summary",'
        f'"performance_grade":"Elite|High|Medium|Low",'
        f'"recommendations":["rec 1","rec 2"]}}'
    )

    raw = await infer(system, user, 350)
    return {"ok": True, "data": extract_json(raw)}


@app.post("/incident-response")
async def incident_response(req: IncidentRequest, request: Request):
    check_auth(request)

    symptoms = "\n".join(f"- {s}" for s in req.symptoms[:5]) or "- (none)"
    logs_str = "\n".join(f"- {l[:150]}" for l in req.error_logs[:3]) or "- (none)"
    system = (
        "You are an experienced SRE. "
        "Output ONLY a single valid JSON object. No markdown."
    )
    user = (
        f"Incident: {req.title} | Severity: {req.severity} | "
        f"Service: {req.affected_service or 'unknown'}\n"
        f"Symptoms:\n{symptoms}\n"
        f"Logs:\n{logs_str}\n\n"
        f"Fill this JSON:\n"
        f'{{"immediate_actions":["action 1","action 2","action 3"],'
        f'"likely_root_causes":["cause 1","cause 2"],'
        f'"estimated_resolution_time":"X-Y minutes"}}'
    )

    raw = await infer(system, user, 350)
    return {"ok": True, "data": extract_json(raw)}
