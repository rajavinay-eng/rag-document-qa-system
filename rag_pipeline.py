import chromadb
import fitz
import numpy as np
import time
import os
import tiktoken

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate

# ── MODELS ────────────────────────────────────────────────

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

enc = tiktoken.encoding_for_model(
    "gpt-3.5-turbo"
)

# ── DATABASE ──────────────────────────────────────────────

client = chromadb.PersistentClient(
    path="./project2_db"
)

collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# ── PROMPT ────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a document assistant.

Answer ONLY from the provided context below.

If answer not found:
'I cannot find this information in the provided document.'

Context:
{context}

Question:
{question}

Answer:
"""
)

# ── PDF PROCESSING ────────────────────────────────────────

def process_pdf(
    pdf_path,
    doc_id,
    doc_name,
    chunk_size=400,
    overlap=50
):

    doc = fitz.open(pdf_path)

    chunks = []
    idx = 0

    for page_num, page in enumerate(doc):

        text = page.get_text().strip()

        if not text:
            continue

        tokens = enc.encode(text)

        for i in range(
            0,
            len(tokens),
            chunk_size - overlap
        ):

            chunk_tokens = tokens[
                i:i + chunk_size
            ]

            if not chunk_tokens:
                break

            chunks.append({
                "text": enc.decode(chunk_tokens),
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page": page_num + 1,
                "chunk_idx": idx
            })

            idx += 1

            if i + chunk_size >= len(tokens):
                break

    doc.close()

    return chunks

# ── INDEX DOCUMENT ────────────────────────────────────────

def index_document(
    pdf_path,
    doc_id,
    doc_name
):

    chunks = process_pdf(
        pdf_path,
        doc_id,
        doc_name
    )

    if not chunks:
        return {
            "status": "error",
            "message": "No text extracted"
        }

    try:
        collection.delete(
            where={"doc_id": doc_id}
        )
    except:
        pass

    texts = [c["text"] for c in chunks]

    embeddings = embed_model.encode(texts)

    metadatas = [
        {
            "doc_id": c["doc_id"],
            "doc_name": c["doc_name"],
            "page": c["page"],
            "chunk_idx": c["chunk_idx"]
        }
        for c in chunks
    ]

    ids = [
        f"{doc_id}_chunk_{c['chunk_idx']}"
        for c in chunks
    ]

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )

    return {
        "status": "success",
        "chunks_added": len(chunks)
    }

# ── HYBRID SEARCH ─────────────────────────────────────────

def hybrid_search(
    query,
    n_results=5,
    doc_id=None
):

    query_emb = embed_model.encode([query])

    query_params = {
        "query_embeddings": query_emb.tolist(),
        "n_results": n_results,
        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    if doc_id:
        query_params["where"] = {
            "doc_id": doc_id
        }

    results = collection.query(
        **query_params
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not docs:
        return []

    tokenized_docs = [
        d.lower().split()
        for d in docs
    ]

    bm25 = BM25Okapi(tokenized_docs)

    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    semantic_scores = [
        1 - d
        for d in dists
    ]

    combined = []

    for i in range(len(docs)):

        score = (
            0.5 * semantic_scores[i]
            + 0.5 * bm25_scores[i]
        )

        combined.append({
            "text": docs[i],
            "doc_name": metas[i]["doc_name"],
            "page": metas[i]["page"],
            "score": score
        })

    combined.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return combined

# ── QUERY RAG ─────────────────────────────────────────────

def query_rag(
    question,
    doc_id=None
):

    start = time.time()

    results = hybrid_search(
        question,
        n_results=5,
        doc_id=doc_id
    )

    if not results:
        return {
            "status": "error",
            "answer": "No matching documents found."
        }

    context = "\n\n".join([
        f"""
Source: {r['doc_name']}
Page: {r['page']}

{r['text']}
"""
        for r in results[:3]
    ])

    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    from openai import OpenAI

    client_openai = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    response = client_openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=400
    )

    answer = response.choices[0].message.content

    total_time = round(
        (time.time() - start) * 1000,
        1
    )

    return {
        "status": "success",
        "question": question,
        "answer": answer,
        "sources": results[:3],
        "latency_ms": total_time
    }