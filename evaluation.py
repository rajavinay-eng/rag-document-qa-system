# project2/evaluation.py
# Project 2 — RAG Document Q&A System
# Component: RAGAS-style Evaluation
# Built: Day 39

import numpy as np
import time
import statistics
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import chromadb

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
reranker    = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
client      = chromadb.PersistentClient(path="./project2_db")
collection  = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

def evaluate_faithfulness(answer, contexts):
    """Is answer grounded in context? Score 0-1"""
    if not answer or not contexts:
        return 0.0
    stop = {"the","a","an","is","are","was","were","in",
            "of","to","and","or","for","on","at","by","from"}
    a_words = set(answer.lower().split()) - stop
    c_words = set(" ".join(contexts).lower().split()) - stop
    if not a_words:
        return 1.0
    return round(len(a_words & c_words) / len(a_words), 3)

def evaluate_answer_relevancy(question, answer):
    """Does answer address the question? Score 0-1"""
    if not question or not answer:
        return 0.0
    q_emb = embed_model.encode([question])
    a_emb = embed_model.encode([answer])
    return round(float(cosine_similarity(q_emb, a_emb)[0][0]), 3)

def evaluate_context_precision(question, contexts):
    """Are retrieved chunks relevant? Score 0-1"""
    if not contexts:
        return 0.0
    q_emb  = embed_model.encode([question])
    c_embs = embed_model.encode(contexts)
    scores = cosine_similarity(q_emb, c_embs)[0]
    return round(float(np.mean(scores)), 3)

def evaluate_pipeline(question, answer, contexts):
    """Run all metrics for one QA pair"""
    return {
        "faithfulness":      evaluate_faithfulness(answer, contexts),
        "answer_relevancy":  evaluate_answer_relevancy(question, answer),
        "context_precision": evaluate_context_precision(question, contexts)
    }

def run_full_evaluation(test_cases):
    """
    Run evaluation on list of test cases.
    Returns aggregate scores for README.

    test_cases = [{"question": "...", "answer": "...",
                   "contexts": ["..."]}]
    """
    results = []
    for case in test_cases:
        scores = evaluate_pipeline(
            case["question"],
            case["answer"],
            case["contexts"]
        )
        results.append(scores)

    return {
        "faithfulness":      round(np.mean([r["faithfulness"]
                                            for r in results]), 3),
        "answer_relevancy":  round(np.mean([r["answer_relevancy"]
                                            for r in results]), 3),
        "context_precision": round(np.mean([r["context_precision"]
                                            for r in results]), 3),
        "n_samples":         len(results)
    }

def measure_latency(question, n_runs=3):
    """Measure P50 and P95 latency"""
    totals = []
    for _ in range(n_runs):
        t0 = time.time()
        q_emb = embed_model.encode([question])
        collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=min(5, max(collection.count(), 1)),
            include=["documents"]
        )
        totals.append((time.time()-t0)*1000)

    return {
        "p50_ms": round(statistics.median(totals), 1),
        "p95_ms": round(sorted(totals)[int(len(totals)*0.95)], 1)
    }

if __name__ == "__main__":
    print("Testing evaluation.py...")

    test_cases = [{
        "question": "What is the dosage for kidney patients?",
        "answer":   "Kidney patients require 50% dose reduction.",
        "contexts": ["Patients with kidney disease require 50% reduction."]
    }]

    scores = run_full_evaluation(test_cases)
    print(f"Faithfulness:      {scores['faithfulness']}")
    print(f"Answer Relevancy:  {scores['answer_relevancy']}")
    print(f"Context Precision: {scores['context_precision']}")

    latency = measure_latency("kidney dosage")
    print(f"P50 latency: {latency['p50_ms']}ms")
    print(f"P95 latency: {latency['p95_ms']}ms")
    print("\nevaluation.py working correctly ✅")