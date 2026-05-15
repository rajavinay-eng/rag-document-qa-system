# project2/api.py
# Project 2 — RAG Document Q&A System
# Component: FastAPI Production Backend
# Built: Day 40

import os
import time
from dotenv import load_dotenv
load_dotenv()

import tempfile
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import Project 2 components
import sys
sys.path.append(os.path.dirname(__file__))
from rag_pipeline import query_rag, index_document
from evaluation  import evaluate_pipeline, measure_latency

# ── APP SETUP ─────────────────────────────────────────────
app = FastAPI(
    title="RAG Document Q&A API",
    description="Project 2 — AI Engineer Portfolio by Raja Vinay Kumar",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])

# ── STATE ─────────────────────────────────────────────────
request_counts = defaultdict(list)
request_log    = []

# ── REQUEST MODELS ────────────────────────────────────────
class QueryRequest(BaseModel):
    question:  str
    doc_id:    Optional[str] = None
    n_results: int           = 5

# ── MIDDLEWARE ────────────────────────────────────────────
@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    elapsed  = round((time.time()-start)*1000, 1)
    response.headers["X-Response-Time"] = f"{elapsed}ms"
    return response

# ── RATE LIMITER ──────────────────────────────────────────
def rate_limit(ip: str, limit=20, window=60):
    now   = time.time()
    times = [t for t in request_counts[ip] if now-t < window]
    request_counts[ip] = times
    if len(times) >= limit:
        raise HTTPException(429, f"Rate limit: {limit} req/{window}s")
    request_counts[ip].append(now)

# ── ENDPOINTS ─────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(400, "File too large — max 50MB")

    doc_id = file.filename.replace(".pdf","").replace(" ","_")
    start  = time.time()

    with tempfile.NamedTemporaryFile(suffix=".pdf",
                                     delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = index_document(tmp_path, doc_id, file.filename)
    finally:
        os.unlink(tmp_path)

    result["processing_ms"] = round((time.time()-start)*1000, 1)
    return result

@app.post("/query")
async def query(request: Request, body: QueryRequest):
    rate_limit(request.client.host)

    if not body.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    if len(body.question) > 1000:
        raise HTTPException(400, "Question too long — max 1000 chars")

    start = time.time()
    try:
        result = query_rag(body.question, body.doc_id)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {str(e)}")

    elapsed = round((time.time()-start)*1000, 1)
    request_log.append({
        "timestamp":   time.time(),
        "question":    body.question[:50],
        "latency_ms":  elapsed,
        "tokens_used": result.get("tokens_used", 0),
        "success":     result.get("status") == "success"
    })

    return JSONResponse(content=result)

@app.get("/stats")
async def stats():
    if not request_log:
        return {"message": "No requests yet"}
    total        = len(request_log)
    successful   = sum(1 for r in request_log if r["success"])
    avg_latency  = sum(r["latency_ms"] for r in request_log)/total
    total_tokens = sum(r["tokens_used"] for r in request_log)
    return {
        "total_requests":    total,
        "successful":        successful,
        "error_rate":        round((total-successful)/total, 3),
        "avg_latency_ms":    round(avg_latency, 1),
        "total_tokens_used": total_tokens
    }

# ── RUN ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)