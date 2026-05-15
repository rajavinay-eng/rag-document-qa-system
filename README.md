# Production RAG Document Q&A System

Production-ready AI application for intelligent document question-answering using Retrieval-Augmented Generation (RAG), semantic search, reranking, and LLM-powered responses.

## Features

* PDF document upload
* Semantic search
* Hybrid retrieval pipeline
* GPT-powered question answering
* Source citations
* FastAPI backend
* Streamlit frontend
* ChromaDB vector database
* Docker support
* Retrieval latency tracking
* Reranking pipeline
* Production-oriented architecture

## Tech Stack

* Python
* FastAPI
* Streamlit
* LangChain
* ChromaDB
* OpenAI API
* Docker

## System Architecture

1. Upload PDF
2. Extract document chunks
3. Generate embeddings
4. Store vectors in ChromaDB
5. Retrieve relevant chunks
6. Rerank retrieved chunks
7. Generate final answer using GPT

## Run Locally

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start backend

```bash
uvicorn api:app --reload
```

### Start frontend

```bash
streamlit run app.py
```

## Example Questions

* Summarize this document
* What are the important topics?
* Explain the key concepts
* What does the document say about feature scaling?

## Future Improvements

* Multi-document support
* Authentication
* Cloud deployment
* Conversation memory
* Advanced reranking
* Monitoring dashboard

## Author

Raja Vinay Kumar Koppula
AI Engineer Portfolio Project
