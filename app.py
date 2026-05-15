# project2/app.py
import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG Document Q&A System")
st.caption("Upload a PDF. Ask questions. Get answers with sources.")

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose PDF", type="pdf")

    if uploaded_file:
        if st.button("Index Document", type="primary"):
            with st.spinner("Processing PDF..."):
                try:
                    response = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded_file.name,
                                        uploaded_file.getvalue(),
                                        "application/pdf")}
                    )
                    result = response.json()
                    if result.get("status") == "success":
                        st.success(
                            f"✅ Indexed {result['chunks_added']} chunks"
                        )
                        st.session_state["doc_id"] = result["doc_id"]
                        st.session_state["doc_name"] = uploaded_file.name
                    else:
                        st.error(f"Error: {result}")
                except Exception as e:
                    st.error(f"API error: {e}")

    if "doc_name" in st.session_state:
        st.success(f"Active: {st.session_state['doc_name']}")

    st.divider()
    st.header("System Stats")
    if st.button("Refresh Stats"):
        try:
            stats = requests.get(f"{API_URL}/stats").json()
            st.metric("Total Queries", stats.get("total_requests", 0))
            st.metric("Avg Latency",
                      f"{stats.get('avg_latency_ms', 0):.0f}ms")
            st.metric("Error Rate",
                      f"{stats.get('error_rate', 0)*100:.1f}%")
        except:
            st.warning("API not reachable")

# ── MAIN CHAT ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.caption(
                        f"**{s['doc_name']}** — Page {s['page']} "
                        f"(similarity: {s['similarity']})"
                    )
                    st.text(s["text"][:200])
        if msg.get("timings"):
            t = msg["timings"]
            cols = st.columns(4)
            cols[0].metric("Retrieval",
                           f"{t.get('retrieval_ms',0)}ms")
            cols[1].metric("Reranking",
                           f"{t.get('reranking_ms',0)}ms")
            cols[2].metric("LLM",
                           f"{t.get('llm_ms',0)}ms")
            cols[3].metric("Total",
                           f"{t.get('total_ms',0)}ms")

# Question input
if question := st.chat_input("Ask a question about your document..."):

    if "doc_id" not in st.session_state:
        st.warning("Please upload a document first")
        st.stop()

    # Show user message
    st.session_state.messages.append({
        "role": "user", "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        placeholder   = st.empty()
        full_response = ""

        try:
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "question": question,
                    "doc_id":   st.session_state["doc_id"]
                }
            )
            result = response.json()

            if result.get("status") == "success":
                answer = result.get("answer", "No answer returned")

                # Stream display
                for word in answer.split():
                    full_response += word + " "
                    placeholder.markdown(full_response + "▌")
                    time.sleep(0.02)
                placeholder.markdown(full_response)

                # Show sources
                sources = result.get("sources", [])
                if sources:
                    with st.expander("📚 Sources"):
                        for s in sources:
                            st.caption(
                                f"**{s.get('doc_name','Doc')}** — "
                                f"Page {s.get('page',1)} "
                                f"(similarity: {s.get('similarity',0)})"
                            )
                            st.text(s.get("text","")[:200])

                # Show latency
                timings = result.get("timings", {})
                if timings:
                    cols = st.columns(4)
                    cols[0].metric("Retrieval",
                                   f"{timings.get('retrieval_ms',0)}ms")
                    cols[1].metric("Reranking",
                                   f"{timings.get('reranking_ms',0)}ms")
                    cols[2].metric("LLM",
                                   f"{timings.get('llm_ms',0)}ms")
                    cols[3].metric("Total",
                                   f"{timings.get('total_ms',0)}ms")

                # Save to history
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": full_response,
                    "sources": sources,
                    "timings": timings
                })

            elif result.get("status") == "low_confidence":
                st.warning(result.get("answer"))
            else:
                st.error("Could not get answer")

        except Exception as e:
            st.error(f"Error: {e}")