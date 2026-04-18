"""
PipelineXR LLM Service — Hugging Face Space
Qwen2.5-Coder-7B-Instruct (4-bit GGUF) via llama-cpp-python.
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
# llama-cpp-python is installed via Dockerfile
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipelinexr-llm")

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID    = os.getenv("MODEL_REPO", "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF")
GGUF_FILE  = os.getenv("MODEL_FILE", "qwen2.5-coder-7b-instruct-q4_k_m.gguf")
MAX_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
CTX_SIZE   = int(os.getenv("CTX_SIZE", "4096"))
API_SECRET = os.getenv("API_SECRET", "")

llm = None
_infer_lock = asyncio.Lock()

def load_model():
    global llm
    log.info(f"Downloading {GGUF_FILE} from {REPO_ID} ...")
    
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename=GGUF_FILE,
            cache_dir="/app/.cache"  # Explicitly set cache dir
        )
        log.info(f"Model cached at {model_path}. Loading into llama-cpp ...")
        
        llm = Llama(
            model_path=model_path,
            n_ctx=CTX_SIZE,
            n_threads=max(1, os.cpu_count() or 2), # Ensure at least 1 thread
            n_batch=512,
            verbose=False,
            use_mmap=True,
            use_mlock=False,
        )
        log.info("✅ Model ready.")
    except Exception as e:
        log.error(f"❌ Model loading failed: {e}")
        raise

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

# ... (Rest of your app.py code remains the same) ...

# ── Auth ──────────────────────────────────────────────────────────────────────
def check_auth(request: Request):
    if not API_SECRET:
        return
    if request.headers.get("Authorization") != f"Bearer {API_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")

# ... (Include the rest of your Pydantic models and endpoints from your previous file) ...
# (I have omitted them here for brevity, but you should keep them exactly as they were)
