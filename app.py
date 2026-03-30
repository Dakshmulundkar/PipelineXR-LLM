"""
PipelineXR LLM Service — Hugging Face Space
Serves Qwen-7B (4-bit quantized) via FastAPI for multiple DevOps use cases.

Endpoints:
  POST /security-review     — enhanced Trivy findings analysis
  POST /pipeline-email      — pipeline failure email generation
  POST /monitor-email       — uptime alert email generation
  POST /dora-insights       — DORA metrics executive summary
  POST /incident-response   — incident response guidance
  GET  /health              — liveness check
"""

import os
import json
import time
import logging
import traceback
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipelinexr-llm")

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_ID   = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))
DEVICE     = "cpu"  # Force CPU to avoid GPU memory issues
API_SECRET = os.getenv("API_SECRET", "")   # optional bearer token guard

log.info(f"Device: {DEVICE} | Model: {MODEL_ID}")

# ── Global model handles ──────────────────────────────────────────────────────
tokenizer = None
model     = None

def load_model():
    global tokenizer, model
    log.info("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ) if DEVICE == "cuda" else None

    if DEVICE == "cpu":
        os.makedirs("/tmp/model_offload", exist_ok=True)

    log.info("Loading model (4-bit quant on GPU, fp32 on CPU)…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_cfg,
        device_map="auto" if DEVICE == "cuda" else None,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_folder="/tmp/model_offload" if DEVICE == "cpu" else None,
    )
    model.eval()

    # Warmup run — compiles CUDA kernels so first real request isn't slow
    if DEVICE == "cuda":
        log.info("Running warmup inference…")
        _ = infer("You are a helpful assistant.", "Hello!", max_tokens=10)

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

# ── Auth guard ────────────────────────────────────────────────────────────────
def check_auth(request: Request):
    if not API_SECRET:
        return
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {API_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")

# ── Core inference ────────────────────────────────────────────────────────────
def infer(system_prompt: str, user_prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE if DEVICE == "cuda" else "cpu")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Strip the prompt tokens from output
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def safe_json(text: str) -> dict:
    """Extract JSON from model output — handles markdown code fences."""
    text = text.strip()
    # Strip ```json ... ``` fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                text = part
                break
    try:
        return json.loads(text)
    except Exception:
        # Return raw text wrapped in a standard envelope
        return {"raw": text, "parse_error": True}

# ── Request / Response models ─────────────────────────────────────────────────
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
    trend_data: list[dict] = []

class IncidentRequest(BaseModel):
    title: str
    severity: str = "high"
    affected_service: str = ""
    symptoms: list[str] = []
    recent_changes: list[str] = []
    error_logs: list[str] = []

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.post("/security-review")
async def security_review(req: SecurityReviewRequest, request: Request):
    check_auth(request)
    t0 = time.time()
    try:
        # Limit to top 10 by severity to stay within context window
        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        top_vulns = sorted(req.vulnerabilities, key=lambda v: sev_order.get(v.get("severity","low").lower(), 4))[:10]

        vuln_summary = "\n".join(
            f"- [{v.get('severity','?').upper()}] {v.get('id','N/A')} in {v.get('package_name','unknown')}"
            f" (installed: {v.get('installed_version','?')}, fix: {v.get('fixed_version','none')})"
            f": {(v.get('description') or v.get('title',''))[:120]}"
            for v in top_vulns
        )

        system = (
            "You are a senior application security engineer. "
            "Analyze vulnerability findings and return ONLY valid JSON — no markdown, no prose outside JSON."
        )
        user = f"""Repository: {req.repository}
Scanner: {req.scan_engine}
Findings ({len(req.vulnerabilities)} total, showing top {len(top_vulns)}):
{vuln_summary}

Return JSON with this exact structure:
{{
  "risk_summary": "2-3 sentence executive summary",
  "critical_actions": ["action1", "action2"],
  "per_vuln": [
    {{"id": "CVE-...", "fix": "specific fix command or config change", "priority": "immediate|soon|low"}}
  ],
  "overall_posture": "secure|at-risk|critical",
  "estimated_fix_time": "e.g. 2-4 hours"
}}"""

        raw = infer(system, user, max_tokens=800)
        result = safe_json(raw)
        return {"ok": True, "data": result, "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        log.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/pipeline-email")
async def pipeline_email(req: PipelineEmailRequest, request: Request):
    check_auth(request)
    t0 = time.time()
    try:
        steps_str = "\n".join(f"  - {s}" for s in req.failed_steps) if req.failed_steps else "  (no step details available)"
        duration_min = req.duration_seconds // 60
        duration_sec = req.duration_seconds % 60

        system = (
            "You are a DevOps engineer writing professional incident notification emails. "
            "Return ONLY valid JSON — no markdown outside JSON."
        )
        user = f"""Generate a pipeline failure notification email for:
Repository: {req.repository}
Workflow: {req.workflow_name}
Branch: {req.head_branch}
Result: {req.conclusion}
Duration: {duration_min}m {duration_sec}s
Triggered by: {req.triggering_actor or 'automated'}
Commit: {req.commit_message[:100] if req.commit_message else 'N/A'}
Failed steps:
{steps_str}
Run URL: {req.run_url or 'N/A'}

Return JSON:
{{
  "subject": "email subject line",
  "body_html": "full HTML email body with sections: Executive Summary, What Failed, Root Cause Analysis, Recommended Actions, Next Steps",
  "body_text": "plain text version",
  "urgency": "low|medium|high|critical"
}}"""

        raw = infer(system, user, max_tokens=900)
        result = safe_json(raw)
        return {"ok": True, "data": result, "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        log.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/monitor-email")
async def monitor_email(req: MonitorEmailRequest, request: Request):
    check_auth(request)
    t0 = time.time()
    try:
        status_str = "DOWN" if not req.is_up else "RECOVERED"
        system = (
            "You are a site reliability engineer writing uptime alert emails. "
            "Return ONLY valid JSON."
        )
        user = f"""Generate an uptime monitoring alert email:
Service URL: {req.url}
Status: {status_str}
Response time: {req.response_time_ms}ms
Consecutive failures: {req.consecutive_failures}
Incident started: {req.incident_started_at or 'unknown'}
Error: {req.error or 'timeout / no response'}

Return JSON:
{{
  "subject": "email subject",
  "body_html": "HTML email with: Service Status, Impact Assessment, Timeline, Suggested Mitigations",
  "body_text": "plain text version",
  "severity": "warning|critical"
}}"""

        raw = infer(system, user, max_tokens=700)
        result = safe_json(raw)
        return {"ok": True, "data": result, "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        log.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/dora-insights")
async def dora_insights(req: DoraInsightsRequest, request: Request):
    check_auth(request)
    t0 = time.time()
    try:
        trend_str = json.dumps(req.trend_data[-7:]) if req.trend_data else "[]"
        system = (
            "You are a DevOps performance analyst. "
            "Return ONLY valid JSON — no markdown outside JSON."
        )
        user = f"""Analyze DORA metrics for {req.repository} over {req.time_range}:
- Avg build duration: {req.avg_build_duration:.1f} minutes
- Success rate: {req.success_rate:.1f}%
- Total deployments: {req.total_deployments}
- Failed deployments: {req.failed_deployments}
- Recent trend (last 7 data points): {trend_str}

Return JSON:
{{
  "executive_summary": "3-4 sentence summary for non-technical stakeholders",
  "performance_grade": "Elite|High|Medium|Low",
  "key_insights": ["insight1", "insight2", "insight3"],
  "recommendations": ["rec1", "rec2"],
  "predicted_trend": "improving|stable|degrading",
  "benchmark_comparison": "how this compares to DORA industry benchmarks"
}}"""

        raw = infer(system, user, max_tokens=700)
        result = safe_json(raw)
        return {"ok": True, "data": result, "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        log.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/incident-response")
async def incident_response(req: IncidentRequest, request: Request):
    check_auth(request)
    t0 = time.time()
    try:
        symptoms_str  = "\n".join(f"  - {s}" for s in req.symptoms)
        changes_str   = "\n".join(f"  - {c}" for c in req.recent_changes)
        logs_str      = "\n".join(f"  - {l[:200]}" for l in req.error_logs[:5])

        system = (
            "You are an experienced SRE providing incident response guidance. "
            "Return ONLY valid JSON."
        )
        user = f"""Incident: {req.title}
Severity: {req.severity}
Affected service: {req.affected_service}
Symptoms:
{symptoms_str or '  (none provided)'}
Recent changes:
{changes_str or '  (none provided)'}
Error logs:
{logs_str or '  (none provided)'}

Return JSON:
{{
  "immediate_actions": ["step1", "step2", "step3"],
  "likely_root_causes": ["cause1", "cause2"],
  "diagnostic_commands": ["command1", "command2"],
  "escalation_path": "who to contact and when",
  "post_incident_tasks": ["task1", "task2"],
  "estimated_resolution_time": "e.g. 30-60 minutes"
}}"""

        raw = infer(system, user, max_tokens=700)
        result = safe_json(raw)
        return {"ok": True, "data": result, "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        log.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
