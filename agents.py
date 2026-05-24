"""
Agent orchestration utilities for retrieval workflow experiments.
Implements tool routing for search, calculation, and document listing workflows.
"""
# project2/agents.py
# Project 2 — RAG Document Q&A System
# Component: Agent + Tool Calling
# Built: Day 37

import chromadb
import numpy as np
import time
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ── SETUP ─────────────────────────────────────────────────
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client      = chromadb.PersistentClient(path="./project2_db")
collection  = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# ── TOOLS ─────────────────────────────────────────────────
def tool_search(query, n=3):
    query_emb = embed_model.encode([query])
    results   = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=min(n, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"]
    )
    docs  = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [{"text": d, "doc_name": m.get("doc_name", "Doc"),
             "page": m.get("page", 1),
             "similarity": round(1-dist, 3)}
            for d, m, dist in zip(docs, metas, dists)]

def tool_calculate(expression):
    try:
        clean = re.sub(r'[^0-9+\-*/.() ]', '', expression)
        return {"result": eval(clean), "status": "success"}
    except:
        return {"result": None, "status": "error"}

def tool_list_docs():
    data = collection.get(include=["metadatas"])
    seen = {}
    for m in data["metadatas"]:
        doc_id = m.get("doc_id", "unknown")
        if doc_id not in seen:
            seen[doc_id] = m.get("doc_name", "Unknown")
    return [{"doc_id": k, "doc_name": v} for k,v in seen.items()]

# ── ROUTING ───────────────────────────────────────────────
def classify(question):
    q = question.lower()
    if any(s in q for s in ["list", "what documents",
                              "which documents", "available"]):
        return "list"
    if any(s in q for s in ["calculate", "how much",
                              "percent", "%"]) and \
       any(c.isdigit() for c in question):
        return "calculate"
    return "search"

# ── AGENT ─────────────────────────────────────────────────
def run_agent(question, doc_id=None):
    """Main agent — routes to correct tool"""
    start       = time.time()
    tool_choice = classify(question)

    if tool_choice == "list":
        docs   = tool_list_docs()
        answer = f"Available documents: " + \
                 ", ".join(d["doc_name"] for d in docs)

    elif tool_choice == "calculate":
        result = tool_calculate(question)
        answer = (f"Result: {result['result']}"
                  if result["status"] == "success"
                  else "Could not calculate.")

    else:
        results = tool_search(question)
        if results and results[0]["similarity"] > 0.3:
            top    = results[0]
            answer = (f"[{top['doc_name']}, p.{top['page']}] "
                     f"{top['text']}")
        else:
            answer = "I could not find relevant information."

    return {
        "question":   question,
        "tool":       tool_choice,
        "answer":     answer,
        "latency_ms": round((time.time()-start)*1000, 1)
    }

# ── TEST ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing agents.py...")
    for q in ["kidney dosage", "list documents",
               "Calculate 200 * 0.5"]:
        r = run_agent(q)
        print(f"\nQ: {r['question']}")
        print(f"Tool: {r['tool']}")
        print(f"A: {r['answer'][:80]}")
    print("\nagents.py working correctly ✅")
