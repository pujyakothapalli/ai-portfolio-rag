import os
from pypdf import PdfReader
from typing import Optional
from .chunker import chunk_document
from .retriever import HybridRetriever
from .classifier import classify_query
from .generator import generate_answer

retriever = HybridRetriever()

def ingest_pdf(file_path: str) -> dict:
    """Extract text from PDF and ingest into vector store."""
    reader = PdfReader(file_path)
    text = "\n".join([
        page.extract_text() for page in reader.pages
        if page.extract_text()
    ])
    return ingest_text(text, source=os.path.basename(file_path))

def ingest_text(text: str, source: str = "document") -> dict:
    """Chunk and store text."""
    chunks = chunk_document(text, strategy="semantic")
    retriever.add_documents(chunks, source=source)
    return {"chunks_created": len(chunks), "source": source}

def query(question: str, verbose: bool = False) -> dict:
    """Full adaptive RAG pipeline."""

    # Step 1 — classify query
    classification = classify_query(question)
    strategy = classification.get("retrieval_strategy", "hybrid")
    top_k = classification.get("top_k", 5)

    if verbose:
        print(f"Query type: {classification['query_type']}")
        print(f"Strategy: {strategy} | top_k: {top_k}")

    # Step 2 — retrieve
    if strategy == "dense":
        chunks = retriever.dense_retrieve(question, top_k=top_k)
    else:
        chunks = retriever.hybrid_retrieve(question, top_k=top_k)

    if not chunks:
        return {"answer": "No documents ingested yet. Please upload a document first.",
                "confidence": 0, "query_type": classification["query_type"],
                "chunks_retrieved": 0}

    # Step 3 — generate with Self-RAG
    result = generate_answer(question, chunks)

    return {
        **result,
        "query_type": classification["query_type"],
        "retrieval_strategy": strategy,
        "chunks_retrieved": len(chunks),
        "sources": list(set(c["source"] for c in chunks)),
        "classifier_reasoning": classification.get("reasoning", "")
    }

def reset_knowledge_base():
    retriever.reset()