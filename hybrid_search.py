import chromadb
import numpy as np
import time
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ChromaDB persistent storage
client = chromadb.PersistentClient(path="./project2_db")

collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# Normalize scores to 0-1
def normalize_scores(scores):

    min_s = scores.min()
    max_s = scores.max()

    if max_s - min_s == 0:
        return np.zeros_like(scores)

    return (scores - min_s) / (max_s - min_s)

# Hybrid search function
def hybrid_search(
    query,
    n_retrieve=10,
    n_final=3,
    alpha=0.5
):

    start = time.time()

    # Semantic retrieval from ChromaDB
    query_emb = embed_model.encode([query])

    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=min(n_retrieve, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    # BM25 reranking
    tokenized_docs = [d.lower().split() for d in docs]

    bm25 = BM25Okapi(tokenized_docs)

    bm25_scores = bm25.get_scores(query.lower().split())

    bm25_norm = normalize_scores(np.array(bm25_scores))

    # Semantic scores
    semantic_scores = np.array([1 - d for d in distances])

    semantic_norm = normalize_scores(semantic_scores)

    # Combine
    combined_scores = (
        alpha * semantic_norm
        + (1 - alpha) * bm25_norm
    )

    # Top results
    top_indices = np.argsort(combined_scores)[::-1][:n_final]

    elapsed_ms = (time.time() - start) * 1000

    formatted = []

    for idx in top_indices:

        meta = metas[idx]

        formatted.append({
            "text": docs[idx],
            "doc_id": meta.get("doc_id", ""),
            "page": meta.get("page", 1),
            "similarity": round(float(combined_scores[idx]), 3)
        })

    return {
        "query": query,
        "results": formatted,
        "latency_ms": round(elapsed_ms, 1)
    }

# Test
if __name__ == "__main__":

    result = hybrid_search("kidney dosage risks")

    print(f"Query: {result['query']}")
    print(f"Latency: {result['latency_ms']} ms")

    for r in result["results"]:
        print(f"\nSimilarity: {r['similarity']}")
        print(r["text"])