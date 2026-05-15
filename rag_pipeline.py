# project2/rag_pipeline.py
# Project 2 — RAG Document Q&A System
# Component: Core RAG Pipeline — PDF to Answer
# Built: Day 36

import chromadb
import fitz
import numpy as np
import time
import os
import tiktoken
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.prompts import PromptTemplate

# ── MODELS ────────────────────────────────────────────────
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
reranker    = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
enc         = tiktoken.encoding_for_model("gpt-3.5-turbo")

# ── DATABASE ──────────────────────────────────────────────
client     = chromadb.PersistentClient(path="./project2_db")
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# ── PROMPT ────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a document assistant.
Answer ONLY from the provided context below.
If the answer is not in the context, say exactly:
'I cannot find this information in the provided document.'
Never use your training knowledge.
Always cite the source document and page number.

Context:
{context}

Question: {question}

Answer:"""
)

# ── PDF PROCESSING ────────────────────────────────────────
def process_pdf(pdf_path, doc_id, doc_name,
                chunk_size=400, overlap=50):
    """Parse PDF and create chunks"""
    doc    = fitz.open(pdf_path)
    chunks = []
    idx    = 0

    for page_num, page in enumerate(doc):
        text   = page.get_text().strip()
        if not text:
            continue

        tokens = enc.encode(text)
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            if not chunk_tokens:
                break
            chunks.append({
                "text":       enc.decode(chunk_tokens),
                "doc_id":     doc_id,
                "doc_name":   doc_name,
                "page":       page_num + 1,
                "chunk_idx":  idx,
                "token_count": len(chunk_tokens)
            })
            idx += 1
            if i + chunk_size >= len(tokens):
                break

    doc.close()
    return chunks

# ── INDEX DOCUMENT ────────────────────────────────────────
def index_document(pdf_path, doc_id, doc_name):
    """Process PDF and store all chunks in ChromaDB"""
    chunks = process_pdf(pdf_path, doc_id, doc_name)
    if not chunks:
        return {"status": "error", "message": "No text extracted"}

    try:
        collection.delete(where={"doc_id": doc_id})
    except:
        pass

    texts     = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts)
    metadatas = [{
        "doc_id":    c["doc_id"],
        "doc_name":  c["doc_name"],
        "page":      c["page"],
        "chunk_idx": c["chunk_idx"]
    } for c in chunks]
    ids = [f"{doc_id}_chunk_{c['chunk_idx']}" for c in chunks]

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )

    return {
        "status":      "success",
        "doc_id":      doc_id,
        "chunks_added": len(chunks),
        "total_in_db": collection.count()
    }

# ── HYBRID SEARCH ─────────────────────────────────────────
def hybrid_search(query, n_retrieve=10, doc_id=None):
    """BM25 + semantic search combined"""
    query_emb    = embed_model.encode([query])
    query_params = {
        "query_embeddings": query_emb.tolist(),
        "n_results":        min(n_retrieve, max(collection.count(), 1)),
        "include":          ["documents", "metadatas", "distances"]
    }
    if doc_id:
        query_params["where"] = {"doc_id": doc_id}

    results = collection.query(**query_params)
    docs    = results["documents"][0]
    metas   = results["metadatas"][0]
    dists   = results["distances"][0]

    if not docs:
        return []

    tokenized   = [d.lower().split() for d in docs]
    bm25        = BM25Okapi(tokenized)
    bm25_raw    = bm25.get_scores(query.lower().split())

    def normalize(s):
        a = np.array(s)
        if a.max() == a.min():
            return np.zeros_like(a)
        return (a - a.min()) / (a.max() - a.min())

    bm25_norm = normalize(bm25_raw)
    sem_norm  = normalize(np.array([1-d for d in dists]))
    combined  = 0.5 * sem_norm + 0.5 * bm25_norm
    top_idx   = np.argsort(combined)[::-1]

    return [{
        "text":      docs[i],
        "doc_name":  metas[i].get("doc_name", "Document"),
        "page":      metas[i].get("page", 1),
        "chunk_idx": metas[i].get("chunk_idx", 0),
        "similarity": round(float(combined[i]), 3)
    } for i in top_idx]

# ── QUERY ─────────────────────────────────────────────────
def query_rag(question, doc_id=None, use_real_llm=True):
    """
    Complete RAG query pipeline.
    Returns answer + sources + latency breakdown.
    """
    if not question.strip():
        return {"status": "error", "answer": "Empty question"}

    timings     = {}
    total_start = time.time()

    # Step 1: Hybrid retrieval
    t0         = time.time()
    candidates = hybrid_search(question, n_retrieve=15, doc_id=doc_id)
    timings["retrieval_ms"] = round((time.time()-t0)*1000, 1)

    if not candidates:
        return {
            "status": "error",
            "answer": "No documents indexed yet."
        }

    # Lower confidence threshold
    if candidates[0]["similarity"] < 0.1:
        return {
            "status": "low_confidence",
            "answer": "I could not find relevant information for this question.",
            "sources": []
        }

    # Step 2: CrossEncoder reranking
    t0     = time.time()

    pairs  = [(question, c["text"]) for c in candidates[:10]]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        reverse=True,
        key=lambda x: x[0]
    )

    # Use more chunks
    top = [c for _, c in ranked[:5]]

    for i, (sc, _) in enumerate(ranked[:5]):
        top[i]["rerank_score"] = round(float(sc), 3)

    timings["reranking_ms"] = round((time.time()-t0)*1000, 1)

    # Step 3: Build context
    context = "\n\n".join([
        f"""
        Source Document: {c['doc_name']}
        Page Number: {c['page']}

        Content:
        {c['text']}
        """
        for c in top
    ])

    # Better prompt
    prompt = f"""
You are a professional RAG assistant.

Rules:
- Answer ONLY using the provided context.
- If partial information exists, provide the best possible answer.
- Be concise and accurate.
- Mention source page numbers when possible.
- Do NOT say 'I cannot find this information'
unless absolutely nothing relevant exists.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    # Step 4: Token budget safety
    total_toks = len(enc.encode(prompt))

    if total_toks > 5000:
        context = "\n\n".join([
            f"[Page {c['page']}]\n{c['text']}"
            for c in top[:3]
        ])

        prompt = f"""
Use ONLY this context:

{context}

Question:
{question}

Answer:
"""

    # Step 5: LLM call
    t0 = time.time()

    if use_real_llm:
        from openai import OpenAI

        llm_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=500
        )

        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

    else:
        answer = "Simulated answer."
        tokens_used = total_toks + 100

    timings["llm_ms"] = round((time.time()-t0)*1000, 1)
    timings["total_ms"] = round((time.time()-total_start)*1000, 1)

    return {
        "status": "success",
        "question": question,
        "answer": answer,
        "sources": top,
        "timings": timings,
        "tokens_used": tokens_used
    }