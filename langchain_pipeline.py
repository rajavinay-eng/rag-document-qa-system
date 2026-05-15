# project2/langchain_pipeline.py
# Project 2 — RAG Document Q&A System
# Component: LangChain Prompt + Manual Retrieval Combined
# Built: Day 35

import chromadb
import time
import os
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate

# ── SETUP ─────────────────────────────────────────────────
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client      = chromadb.PersistentClient(path="./project2_db")
collection  = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# ── HALLUCINATION PREVENTION PROMPT ───────────────────────
# This exact prompt used in Project 2 final system
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a document assistant.
Answer ONLY from the provided context below.
If the answer is not in the context, say exactly:
'I cannot find this information in the provided document.'
Never use your training knowledge.
Always cite which part of the document supports your answer.

Context:
{context}

Question: {question}

Answer:"""
)

# ── RETRIEVE CHUNKS ───────────────────────────────────────
def retrieve_chunks(query, n_results=5, doc_id=None):
    """Retrieve relevant chunks from ChromaDB"""
    query_emb    = embed_model.encode([query])
    query_params = {
        "query_embeddings": query_emb.tolist(),
        "n_results":        min(n_results, collection.count()),
        "include":          ["documents", "metadatas", "distances"]
    }
    if doc_id:
        query_params["where"] = {"doc_id": doc_id}

    results   = collection.query(**query_params)
    formatted = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        formatted.append({
            "text":       doc,
            "doc_name":   meta.get("doc_name", "Document"),
            "page":       meta.get("page", 1),
            "chunk_idx":  meta.get("chunk_idx", 0),
            "similarity": round(1 - dist, 3)
        })

    return formatted

# ── FORMAT CONTEXT ────────────────────────────────────────
def format_context(chunks):
    """Format chunks for prompt — with source attribution"""
    return "\n\n".join([
        f"[Source: {c['doc_name']}, Page {c['page']}]\n{c['text']}"
        for c in chunks
    ])

# ── BUILD FINAL PROMPT ────────────────────────────────────
def build_rag_prompt(question, chunks):
    """
    Build complete prompt using LangChain template.
    Returns formatted string ready for LLM.
    """
    context = format_context(chunks)
    return RAG_PROMPT.format(
        context=context,
        question=question
    )

# ── COMPLETE PIPELINE ─────────────────────────────────────
def rag_pipeline(question, doc_id=None):
    """
    Manual retrieval + LangChain prompt formatting.
    Ready for LLM call in Day 36.
    """
    start = time.time()

    # Retrieve
    chunks  = retrieve_chunks(question,
                               n_results=5,
                               doc_id=doc_id)

    # Build prompt
    prompt  = build_rag_prompt(question, chunks)

    elapsed = (time.time() - start) * 1000

    return {
        "question":   question,
        "prompt":     prompt,
        "chunks":     chunks,
        "latency_ms": round(elapsed, 1)
    }

# ── TEST ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing langchain_pipeline.py...")

    result = rag_pipeline("What is the dosage for kidney patients?")
    print(f"Question: {result['question']}")
    print(f"Chunks found: {len(result['chunks'])}")
    print(f"Prompt length: {len(result['prompt'])} chars")
    print(f"Latency: {result['latency_ms']}ms")
    print()
    print("Prompt preview:")
    print(result["prompt"][:300])
    print("...")
    print("\nlangchain_pipeline.py working correctly ✅")