---
title: PipelineXR LLM
emoji: 🔐
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# PipelineXR LLM Service

Multi-endpoint FastAPI service running Qwen2.5-7B-Instruct (4-bit quantized) for PipelineXR DevOps platform.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness check |
| POST | `/security-review` | Enhanced Trivy findings analysis |
| POST | `/pipeline-email` | Pipeline failure email generation |
| POST | `/monitor-email` | Uptime alert email generation |
| POST | `/dora-insights` | DORA metrics executive summary |
| POST | `/incident-response` | Incident response guidance |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_SECRET` | Optional bearer token for endpoint auth |
| `MODEL_ID` | HuggingFace model ID (default: `Qwen/Qwen2.5-7B-Instruct`) |
| `MAX_NEW_TOKENS` | Max tokens per response (default: `1024`) |
