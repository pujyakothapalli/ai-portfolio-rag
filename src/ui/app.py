import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
import tempfile
from src.rag.pipeline import ingest_pdf, ingest_text, query, reset_knowledge_base

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="Adaptive RAG System",
    page_icon="🔍",
    layout="wide"
)

# ── Header ────────────────────────────────────────────
st.title("🔍 Adaptive RAG System")
st.markdown("""
*Intelligent document Q&A with query routing, hybrid search, and self-evaluation*
""")

# ── Session state ─────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_ingested" not in st.session_state:
    st.session_state.docs_ingested = []

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.header("📄 Document Upload")
    
    upload_type = st.radio("Input type", ["Upload PDF", "Paste Text"])
    
    if upload_type == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
        if uploaded_file and st.button("Ingest Document"):
            with st.spinner("Processing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded_file.read())
                    tmp_path = f.name
                result = ingest_pdf(tmp_path)
                st.session_state.docs_ingested.append(uploaded_file.name)
                st.success(f"✅ Ingested {result['chunks_created']} chunks")
                os.unlink(tmp_path)
    
    else:
        pasted_text = st.text_area("Paste your text here", height=200)
        doc_name = st.text_input("Document name", value="pasted_document")
        if st.button("Ingest Text"):
            if pasted_text.strip():
                with st.spinner("Processing..."):
                    result = ingest_text(pasted_text, source=doc_name)
                    st.session_state.docs_ingested.append(doc_name)
                    st.success(f"✅ Ingested {result['chunks_created']} chunks")
            else:
                st.warning("Please paste some text first")
    
    # Show ingested docs
    if st.session_state.docs_ingested:
        st.divider()
        st.subheader("📚 Ingested Documents")
        for doc in st.session_state.docs_ingested:
            st.markdown(f"• {doc}")
    
    # Reset button
    st.divider()
    if st.button("🗑️ Reset Knowledge Base", type="secondary"):
        reset_knowledge_base()
        st.session_state.docs_ingested = []
        st.session_state.messages = []
        st.success("Knowledge base cleared")
        st.rerun()

# ── Main chat area ────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask Questions")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "metadata" in msg:
                meta = msg["metadata"]
                cols = st.columns(4)
                cols[0].metric("Confidence", f"{meta['confidence']:.0%}")
                cols[1].metric("Query Type", meta['query_type'])
                cols[2].metric("Strategy", meta['retrieval_strategy'])
                cols[3].metric("Chunks Used", meta['chunks_retrieved'])
                if meta.get("sources"):
                    st.caption(f"📎 Sources: {', '.join(meta['sources'])}")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.docs_ingested:
            st.warning("⚠️ Please upload a document first")
        else:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = query(prompt)
                
                st.markdown(result["answer"])
                
                # Show metadata
                cols = st.columns(4)
                cols[0].metric("Confidence", f"{result['confidence']:.0%}")
                cols[1].metric("Query Type", result['query_type'])
                cols[2].metric("Strategy", result['retrieval_strategy'])
                cols[3].metric("Chunks Used", result['chunks_retrieved'])
                
                if result.get("sources"):
                    st.caption(f"📎 Sources: {', '.join(result['sources'])}")
                
                if result.get("classifier_reasoning"):
                    with st.expander("🔍 Why this strategy?"):
                        st.write(result["classifier_reasoning"])
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "metadata": result
            })

with col2:
    st.subheader("⚙️ System Info")
    st.markdown("""
    **How it works:**
    
    1. **Query Classifier** routes your question to the right retrieval strategy
    
    2. **Hybrid Search** combines semantic (dense) + keyword (BM25) retrieval
    
    3. **Self-RAG Loop** scores confidence and retries if below 70%
    
    **Query Types:**
    - 🔵 FACTUAL → Dense, top-3
    - 🟢 ANALYTICAL → Hybrid, top-7  
    - 🟡 SUMMARIZATION → Hybrid, top-7
    - ⚪ CONVERSATIONAL → Dense, top-3
    """)
    
    st.divider()
    st.subheader("📊 Benchmark")
    st.markdown("""
    | Method | Faithfulness |
    |--------|-------------|
    | Naive RAG | ~0.65 |
    | This System | ~0.85 |
    
    *Measured with RAGAS on 20 Q&A pairs*
    """)