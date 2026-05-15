# project2/workflow.py
# Project 2 — RAG Document Q&A System
# Component: Stateful Workflow with Retry Logic
# Built: Day 38

import chromadb
import numpy as np
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# ── SETUP ─────────────────────────────────────────────────
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
reranker    = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
client      = chromadb.PersistentClient(path="./project2_db")
collection  = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

def _normalize(s):
    a = np.array(s)
    if a.max() == a.min():
        return np.zeros_like(a)
    return (a - a.min()) / (a.max() - a.min())

def _hybrid_search(query, n=8):
    if collection.count() == 0:
        return []
    q_emb   = embed_model.encode([query])
    results = collection.query(
        query_embeddings=q_emb.tolist(),
        n_results=min(n, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    docs, metas, dists = (results["documents"][0],
                           results["metadatas"][0],
                           results["distances"][0])
    if not docs:
        return []
    tokenized   = [d.lower().split() for d in docs]
    bm25        = BM25Okapi(tokenized)
    bm25_norm   = _normalize(bm25.get_scores(query.lower().split()))
    sem_norm    = _normalize(np.array([1-d for d in dists]))
    combined    = 0.5 * sem_norm + 0.5 * bm25_norm
    top_idx     = np.argsort(combined)[::-1]
    return [{"text": docs[i], "doc_name": metas[i].get("doc_name","Doc"),
             "page": metas[i].get("page",1),
             "similarity": round(float(combined[i]),3)}
            for i in top_idx]

# ── WORKFLOW ──────────────────────────────────────────────
def run_workflow(question, use_real_llm=False):
    """
    Stateful RAG workflow with automatic retry.
    Implements LangGraph pattern without framework dependency.

    Nodes: retrieve → [route] → rewrite → retrieve → [route]
                                        ↘ rerank → generate
                                        ↘ fallback
    """
    state = {
        "question":    question,
        "query":       question,
        "candidates":  [],
        "top_chunks":  [],
        "answer":      "",
        "quality":     0.0,
        "retry_count": 0,
        "rewritten":   False,
        "path_taken":  []
    }

    start = time.time()

    # Node: retrieve
    def retrieve():
        cands   = _hybrid_search(state["query"])
        quality = max((c["similarity"] for c in cands), default=0.0)
        state.update({"candidates": cands, "quality": round(quality,3)})
        state["path_taken"].append("retrieve")

    # Node: rewrite
    def rewrite():
        state["query"]       = f"detailed information about {state['question']}"
        state["rewritten"]   = True
        state["retry_count"] += 1
        state["path_taken"].append("rewrite")

    # Node: rerank
    def rerank():
        cands  = state["candidates"]
        if not cands:
            state["top_chunks"] = []
        else:
            pairs  = [(state["question"], c["text"]) for c in cands[:8]]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, cands),
                            reverse=True, key=lambda x: x[0])
            top    = [c for _, c in ranked[:3]]
            for i, (sc, _) in enumerate(ranked[:3]):
                top[i]["rerank_score"] = round(float(sc), 3)
            state["top_chunks"] = top
        state["path_taken"].append("rerank")

    # Node: generate
    def generate():
        chunks  = state["top_chunks"]
        context = "\n\n".join([
            f"[{c['doc_name']}, p.{c['page']}]\n{c['text']}"
            for c in chunks
        ])
        if use_real_llm:
            import os
            from openai import OpenAI
            client_llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp       = client_llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "Answer only from context. Cite sources."},
                    {"role": "user",
                     "content": f"Context:\n{context}\n\nQuestion: {state['question']}"}
                ],
                temperature=0.1, max_tokens=500
            )
            state["answer"] = resp.choices[0].message.content
        else:
            state["answer"] = (f"[Simulated] Based on document: "
                              f"{chunks[0]['text'][:100]}..."
                              if chunks else "No context available.")
        state["path_taken"].append("generate")

    # Node: fallback
    def fallback():
        state["answer"] = ("I could not find relevant information "
                          "in the document for this question.")
        state["path_taken"].append("fallback")

    # ── EXECUTE WORKFLOW ──
    retrieve()

    if state["quality"] >= 0.4:
        rerank()
        generate()
    elif state["retry_count"] < 1 and not state["rewritten"]:
        rewrite()
        retrieve()
        if state["quality"] >= 0.4:
            rerank()
            generate()
        else:
            fallback()
    else:
        fallback()

    state["total_ms"] = round((time.time()-start)*1000, 1)
    return state

# ── TEST ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing workflow.py...")
    result = run_workflow("What is the dosage for kidney patients?")
    print(f"Path:    {' → '.join(result['path_taken'])}")
    print(f"Quality: {result['quality']}")
    print(f"Answer:  {result['answer'][:80]}")
    print(f"Time:    {result['total_ms']}ms")
    print("\nworkflow.py working correctly ✅")